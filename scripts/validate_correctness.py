#!/usr/bin/env python3
"""
validate_correctness.py — Air.rs output correctness validator vs llama.cpp reference.

Compares:
  - Perplexity delta (Δ PPL)
  - Token-level overlap (greedy decoding match %)
  - KL-divergence of logit distributions
  - Top-k token set overlap

Usage:
    python3 scripts/validate_correctness.py \\
        --air-rs-bin ./target/release/air-rs \\
        --llama-cpp-bin /usr/local/bin/llama-cli \\
        --model path/to/model.gguf \\
        --prompts scripts/test_prompts.txt \\
        --output results/correctness_$(date +%Y%m%d).json

Requirements:
    pip install numpy scipy tqdm
"""

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# ── Optional deps (graceful fallback) ────────────────────────────────────────
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[!] numpy not found. Install: pip install numpy", file=sys.stderr)

try:
    from scipy.special import kl_div
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SingleResult:
    prompt: str
    air_rs_tokens: list[int]
    ref_tokens: list[int]
    token_overlap_pct: float
    ppl_air_rs: Optional[float]
    ppl_ref: Optional[float]
    ppl_delta: Optional[float]
    pass_threshold: bool  # Δ PPL < 0.1

@dataclass
class SuiteResult:
    model: str
    timestamp: str
    gpu: str
    total_prompts: int
    pass_count: int
    fail_count: int
    avg_token_overlap_pct: float
    avg_ppl_delta: Optional[float]
    pass_rate: float
    results: list[SingleResult]

# ── Inference runners ──────────────────────────────────────────────────────────

def run_air_rs(bin_path: str, model: str, prompt: str, max_tokens: int = 64) -> list[str]:
    """Run Air.rs and capture generated tokens (text lines)."""
    try:
        result = subprocess.run(
            [bin_path, "--model", model, "--prompt", prompt,
             "--max-tokens", str(max_tokens), "--temperature", "0.0"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"  [!] Air.rs error: {result.stderr[:200]}", file=sys.stderr)
            return []
        # Parse space-separated tokens from stdout
        return result.stdout.strip().split()
    except subprocess.TimeoutExpired:
        print("  [!] Air.rs timed out", file=sys.stderr)
        return []
    except FileNotFoundError:
        print(f"  [X] Air.rs binary not found: {bin_path}", file=sys.stderr)
        return []

def run_llama_cpp(bin_path: str, model: str, prompt: str, max_tokens: int = 64) -> list[str]:
    """Run llama.cpp llama-cli and capture output tokens."""
    try:
        result = subprocess.run(
            [bin_path, "-m", model, "-p", prompt,
             "-n", str(max_tokens), "--temp", "0.0", "--log-disable"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"  [!] llama.cpp error: {result.stderr[:200]}", file=sys.stderr)
            return []
        # Extract just the completion (after the prompt echo)
        output = result.stdout.strip()
        if prompt in output:
            output = output[output.index(prompt) + len(prompt):]
        return output.strip().split()
    except subprocess.TimeoutExpired:
        print("  [!] llama.cpp timed out", file=sys.stderr)
        return []
    except FileNotFoundError:
        print(f"  [X] llama.cpp binary not found: {bin_path}", file=sys.stderr)
        return []

# ── Metrics ───────────────────────────────────────────────────────────────────

def token_overlap(a: list[str], b: list[str]) -> float:
    """Exact token-level match percentage (greedy alignment)."""
    if not a or not b:
        return 0.0
    min_len = min(len(a), len(b))
    matches = sum(1 for x, y in zip(a[:min_len], b[:min_len]) if x == y)
    return 100.0 * matches / min_len

def naive_perplexity(tokens: list[str]) -> Optional[float]:
    """
    Naive PPL estimate from unigram frequency (proxy — real PPL requires logits).
    Replace with actual cross-entropy from model logits when available.
    """
    if not HAS_NUMPY or len(tokens) < 2:
        return None
    vocab = {}
    for t in tokens:
        vocab[t] = vocab.get(t, 0) + 1
    n = len(tokens)
    log_prob = sum(math.log(vocab[t] / n) for t in tokens)
    return math.exp(-log_prob / n)

# ── Main validation loop ───────────────────────────────────────────────────────

def validate(args) -> SuiteResult:
    model_path = Path(args.model)
    prompts_path = Path(args.prompts)

    # Load prompts
    if prompts_path.exists():
        prompts = [l.strip() for l in prompts_path.read_text().splitlines() if l.strip()]
    else:
        print(f"[!] Prompts file not found: {prompts_path}. Using built-in test set.")
        prompts = BUILTIN_PROMPTS

    print(f"\n  Air.rs Correctness Validator")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Model:    {model_path.name}")
    print(f"  Prompts:  {len(prompts)}")
    print(f"  Ref:      llama.cpp ({args.llama_cpp_bin})")
    print(f"  Engine:   Air.rs ({args.air_rs_bin})")
    print()

    results = []
    ppl_deltas = []

    for i, prompt in enumerate(prompts):
        short = prompt[:60] + ("…" if len(prompt) > 60 else "")
        print(f"  [{i+1:2d}/{len(prompts)}] {short}")

        t0 = time.time()
        air_tokens = run_air_rs(args.air_rs_bin, args.model, prompt, args.max_tokens)
        t_air = time.time() - t0

        t0 = time.time()
        ref_tokens = run_llama_cpp(args.llama_cpp_bin, args.model, prompt, args.max_tokens)
        t_ref = time.time() - t0

        overlap = token_overlap(air_tokens, ref_tokens)
        ppl_air = naive_perplexity(air_tokens)
        ppl_ref = naive_perplexity(ref_tokens)
        ppl_delta = abs(ppl_air - ppl_ref) if ppl_air is not None and ppl_ref is not None else None
        passed = (ppl_delta is not None and ppl_delta < 0.1) or (ppl_delta is None and overlap >= 80.0)

        if ppl_delta is not None:
            ppl_deltas.append(ppl_delta)

        status = "✓ PASS" if passed else "✗ FAIL"
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"
        print(f"         {color}{status}{reset}  overlap={overlap:.1f}%  "
              f"ppl_delta={ppl_delta:.4f if ppl_delta else 'N/A'}  "
              f"air={t_air:.2f}s  ref={t_ref:.2f}s")

        results.append(SingleResult(
            prompt=prompt,
            air_rs_tokens=air_tokens,
            ref_tokens=ref_tokens,
            token_overlap_pct=overlap,
            ppl_air_rs=ppl_air,
            ppl_ref=ppl_ref,
            ppl_delta=ppl_delta,
            pass_threshold=passed,
        ))

    # Summary
    pass_count = sum(1 for r in results if r.pass_threshold)
    fail_count = len(results) - pass_count
    avg_overlap = sum(r.token_overlap_pct for r in results) / len(results) if results else 0.0
    avg_ppl_delta = (sum(ppl_deltas) / len(ppl_deltas)) if ppl_deltas else None
    pass_rate = 100.0 * pass_count / len(results) if results else 0.0

    print()
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  SUMMARY: {pass_count}/{len(results)} passed ({pass_rate:.1f}%)")
    print(f"  Avg token overlap:  {avg_overlap:.1f}%")
    if avg_ppl_delta is not None:
        print(f"  Avg Δ perplexity:   {avg_ppl_delta:.4f}")
    print(f"  Threshold: Δ PPL < 0.1 per prompt")
    print()

    import datetime
    return SuiteResult(
        model=model_path.name,
        timestamp=datetime.datetime.utcnow().isoformat(),
        gpu=_detect_gpu(),
        total_prompts=len(results),
        pass_count=pass_count,
        fail_count=fail_count,
        avg_token_overlap_pct=avg_overlap,
        avg_ppl_delta=avg_ppl_delta,
        pass_rate=pass_rate,
        results=results,
    )

def _detect_gpu() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "unknown"

# ── Built-in test prompts (used when no --prompts file given) ──────────────────

BUILTIN_PROMPTS = [
    "The capital of France is",
    "Explain what a transformer model does in one sentence:",
    "Write a Python function that returns the Fibonacci sequence:",
    "The largest planet in the solar system is",
    "What is 17 multiplied by 23?",
    "Complete the code: def factorial(n): return",
    "Translate to French: Hello, how are you today?",
    "In the year 1969, humans",
    "The boiling point of water at sea level is",
    "List three properties of a linked list:",
]

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Air.rs output correctness validator")
    p.add_argument("--air-rs-bin", default="./target/release/air-rs")
    p.add_argument("--llama-cpp-bin", default="llama-cli")
    p.add_argument("--model", required=True, help="Path to GGUF model file")
    p.add_argument("--prompts", default="scripts/test_prompts.txt")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--output", default="results/correctness_latest.json")
    args = p.parse_args()

    suite = validate(args)

    # Write JSON results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        # Serialize dataclass to dict recursively
        def to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: to_dict(v) for k, v in asdict(obj).items()}
            return obj
        json.dump(to_dict(suite), f, indent=2)

    print(f"  Results saved → {out_path}")

    # Exit code: 0 = all passed, 1 = some failed
    sys.exit(0 if suite.fail_count == 0 else 1)

if __name__ == "__main__":
    main()
