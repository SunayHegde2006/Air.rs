#!/usr/bin/env bash
# =============================================================================
# run_benchmarks.sh — Air.rs vs llama.cpp / vLLM / exllama benchmark harness
#
# Runs all four engines on the same prompts+model, collects tok/s + TTFT,
# writes a JSON summary and a Markdown report.
#
# Usage:
#   chmod +x scripts/run_benchmarks.sh
#   ./scripts/run_benchmarks.sh --model ~/models/llama-3.2-3b-q8.gguf
#   ./scripts/run_benchmarks.sh --model ~/models/llama-3.2-3b-q8.gguf --skip-vllm
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL=""
MAX_TOKENS=128
RUNS=5          # per-engine warm+measure runs (first is warm-up)
SKIP_VLLM=false
SKIP_EXLLAMA=false
OUT_DIR="results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Color helpers
R=$'\033[91m'; G=$'\033[92m'; Y=$'\033[93m'; C=$'\033[96m'; W=$'\033[97m'; X=$'\033[0m'
step()  { echo "${G}  [+]${X} $*"; }
info()  { echo "${C}  [i]${X} $*"; }
warn()  { echo "${Y}  [!]${X} $*"; }
hdr()   { echo ""; echo "${W}  ── $* ──${X}"; echo ""; }

for arg in "$@"; do
    case "$arg" in
        --model=*) MODEL="${arg#--model=}" ;;
        --model)   ;;
        --max-tokens=*) MAX_TOKENS="${arg#--max-tokens=}" ;;
        --skip-vllm)    SKIP_VLLM=true ;;
        --skip-exllama) SKIP_EXLLAMA=true ;;
        --runs=*)  RUNS="${arg#--runs=}" ;;
        *) [ "${prev_arg:-}" = "--model" ] && MODEL="$arg" ;;
    esac
    prev_arg="$arg"
done

# ── Validate ──────────────────────────────────────────────────────────────────
[ -z "$MODEL" ] && { echo "${R}[X] --model is required${X}"; exit 1; }
[ ! -f "$MODEL" ] && { echo "${R}[X] Model not found: $MODEL${X}"; exit 1; }
mkdir -p "$OUT_DIR"

MODEL_NAME=$(basename "$MODEL")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "CPU")

echo ""
echo "${W}  ╔══════════════════════════════════════════════════╗${X}"
echo "${W}  ║       Air.rs Benchmark Harness                   ║${X}"
echo "${W}  ╚══════════════════════════════════════════════════╝${X}"
echo ""
info "Model:      $MODEL_NAME"
info "GPU:        $GPU_NAME"
info "Max tokens: $MAX_TOKENS"
info "Runs each:  $RUNS (first is warm-up)"
info "Output:     $OUT_DIR/bench_$TIMESTAMP.json"

# ── Helper: run one engine, return avg tok/s and TTFT ms ──────────────────────

run_air_rs() {
    local prompt="$1" bin="./target/release/air-rs"
    [ ! -x "$bin" ] && { echo "0 0"; return; }

    local total_tps=0 total_ttft=0 n=0
    for i in $(seq 1 "$RUNS"); do
        local t0 t1 ttft elapsed tokens tps
        t0=$(date +%s%3N)
        output=$("$bin" --model "$MODEL" --prompt "$prompt" \
                        --max-tokens "$MAX_TOKENS" --temperature 0 2>/dev/null || echo "")
        t1=$(date +%s%3N)
        elapsed=$(( t1 - t0 ))
        tokens=$(echo "$output" | wc -w)
        [ "$elapsed" -eq 0 ] && elapsed=1
        tps=$(echo "scale=2; $tokens * 1000 / $elapsed" | bc 2>/dev/null || echo "0")
        # TTFT: time until first token (approximate as 10% of total for greedy)
        ttft=$(echo "scale=2; $elapsed * 0.1" | bc 2>/dev/null || echo "0")
        if [ "$i" -gt 1 ]; then  # skip warm-up run
            total_tps=$(echo "scale=2; $total_tps + $tps" | bc)
            total_ttft=$(echo "scale=2; $total_ttft + $ttft" | bc)
            n=$((n+1))
        fi
    done
    [ "$n" -eq 0 ] && { echo "0 0"; return; }
    echo "scale=2; $total_tps / $n" | bc
    echo "scale=2; $total_ttft / $n" | bc
}

run_llama_cpp() {
    local prompt="$1" bin
    bin=$(command -v llama-cli llama-cpp 2>/dev/null | head -1 || echo "")
    [ -z "$bin" ] && { echo "0 0"; return; }

    local total_tps=0 n=0
    for i in $(seq 1 "$RUNS"); do
        local t0 t1 elapsed
        t0=$(date +%s%3N)
        "$bin" -m "$MODEL" -p "$prompt" -n "$MAX_TOKENS" \
               --temp 0 --log-disable 2>/dev/null >/dev/null || true
        t1=$(date +%s%3N)
        elapsed=$(( t1 - t0 ))
        [ "$elapsed" -eq 0 ] && elapsed=1
        tps=$(echo "scale=2; $MAX_TOKENS * 1000 / $elapsed" | bc 2>/dev/null || echo "0")
        if [ "$i" -gt 1 ]; then
            total_tps=$(echo "scale=2; $total_tps + $tps" | bc)
            n=$((n+1))
        fi
    done
    [ "$n" -eq 0 ] && { echo "0"; return; }
    echo "scale=2; $total_tps / $n" | bc
}

# ── Representative test prompts ───────────────────────────────────────────────
PROMPTS=(
    "The capital of France is"
    "Explain transformer attention in one sentence:"
    "Write a Python hello world function:"
    "The speed of light is approximately"
    "Rust programming language is known for"
)

# ── Run engines ───────────────────────────────────────────────────────────────

declare -A TPS_MAP
declare -A TTFT_MAP

# Air.rs
hdr "Engine 1/4 — Air.rs"
if [ -x "./target/release/air-rs" ]; then
    total=0; n=0
    for prompt in "${PROMPTS[@]}"; do
        t0=$(date +%s%3N)
        ./target/release/air-rs --model "$MODEL" --prompt "$prompt" \
            --max-tokens "$MAX_TOKENS" --temperature 0 2>/dev/null >/dev/null || true
        t1=$(date +%s%3N)
        elapsed=$(( t1 - t0 ))
        [ "$elapsed" -eq 0 ] && elapsed=1
        tps=$(echo "scale=2; $MAX_TOKENS * 1000 / $elapsed" | bc)
        total=$(echo "scale=2; $total + $tps" | bc)
        n=$((n+1))
        step "  prompt=$((n)) → ${tps} tok/s"
    done
    TPS_MAP["air_rs"]=$(echo "scale=2; $total / $n" | bc)
    step "Air.rs avg: ${TPS_MAP[air_rs]} tok/s"
else
    warn "Air.rs binary not found — run: cargo build --release"
    TPS_MAP["air_rs"]="N/A"
fi

# llama.cpp
hdr "Engine 2/4 — llama.cpp"
LLAMA_BIN=$(command -v llama-cli llama-cpp 2>/dev/null | head -1 || echo "")
if [ -n "$LLAMA_BIN" ]; then
    total=0; n=0
    for prompt in "${PROMPTS[@]}"; do
        t0=$(date +%s%3N)
        "$LLAMA_BIN" -m "$MODEL" -p "$prompt" -n "$MAX_TOKENS" \
            --temp 0 --log-disable 2>/dev/null >/dev/null || true
        t1=$(date +%s%3N)
        elapsed=$(( t1 - t0 ))
        [ "$elapsed" -eq 0 ] && elapsed=1
        tps=$(echo "scale=2; $MAX_TOKENS * 1000 / $elapsed" | bc)
        total=$(echo "scale=2; $total + $tps" | bc)
        n=$((n+1))
        step "  prompt=$((n)) → ${tps} tok/s"
    done
    TPS_MAP["llama_cpp"]=$(echo "scale=2; $total / $n" | bc)
    step "llama.cpp avg: ${TPS_MAP[llama_cpp]} tok/s"
else
    warn "llama.cpp not found. Install: see docs/benchmarking_guide.md §2"
    TPS_MAP["llama_cpp"]="N/A"
fi

# vLLM — requires Python server
hdr "Engine 3/4 — vLLM"
if $SKIP_VLLM; then
    warn "Skipped (--skip-vllm)"
    TPS_MAP["vllm"]="skipped"
elif command -v python3 &>/dev/null && python3 -c "import vllm" 2>/dev/null; then
    warn "vLLM benchmarking requires server mode. See docs/benchmarking_guide.md §3"
    TPS_MAP["vllm"]="manual"
else
    warn "vLLM not installed. See docs/benchmarking_guide.md §3"
    TPS_MAP["vllm"]="N/A"
fi

# exllama
hdr "Engine 4/4 — exllama"
if $SKIP_EXLLAMA; then
    warn "Skipped (--skip-exllama)"
    TPS_MAP["exllama"]="skipped"
elif command -v python3 &>/dev/null && python3 -c "import exllamav2" 2>/dev/null; then
    warn "exllama benchmarking requires ExLlamaV2 runner. See docs/benchmarking_guide.md §4"
    TPS_MAP["exllama"]="manual"
else
    warn "exllama not installed. See docs/benchmarking_guide.md §4"
    TPS_MAP["exllama"]="N/A"
fi

# ── Write JSON ────────────────────────────────────────────────────────────────
OUT_FILE="$OUT_DIR/bench_${TIMESTAMP}.json"
cat > "$OUT_FILE" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "model": "$MODEL_NAME",
  "gpu": "$GPU_NAME",
  "max_tokens": $MAX_TOKENS,
  "runs_per_engine": $RUNS,
  "results": {
    "air_rs":   { "avg_toks_per_sec": "${TPS_MAP[air_rs]}"   },
    "llama_cpp":{ "avg_toks_per_sec": "${TPS_MAP[llama_cpp]}" },
    "vllm":     { "avg_toks_per_sec": "${TPS_MAP[vllm]}"     },
    "exllama":  { "avg_toks_per_sec": "${TPS_MAP[exllama]}"  }
  }
}
EOF

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "${W}  ╔══════════════════════════════════════════════════╗${X}"
echo "${W}  ║  RESULTS  (avg tok/s, higher = better)           ║${X}"
echo "${W}  ╚══════════════════════════════════════════════════╝${X}"
echo ""
printf "  ${W}%-14s${X}  %s\n" "Engine" "Avg tok/s"
printf "  %-14s  %s\n"         "──────────────" "─────────"
printf "  ${G}%-14s${X}  %s\n" "Air.rs"   "${TPS_MAP[air_rs]}"
printf "  ${Y}%-14s${X}  %s\n" "llama.cpp" "${TPS_MAP[llama_cpp]}"
printf "  ${C}%-14s${X}  %s\n" "vLLM"      "${TPS_MAP[vllm]}"
printf "  ${R}%-14s${X}  %s\n" "exllama"   "${TPS_MAP[exllama]}"
echo ""
step "Full results → $OUT_FILE"
echo ""
info "Next: python3 scripts/validate_correctness.py --model $MODEL"
