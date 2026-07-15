#!/usr/bin/env python3
"""
Memory-Bandwidth Simulation for CDSC 100 tok/s Feasibility
===========================================================
Validates from first principles whether the Consensus-Driven Speculative
Council (CDSC) architecture can sustain ≥ 100 tok/s decode throughput on
a single RTX 3060 12 GB when running a 70B target model.

Physics Model
-------------
In S.L.I.P. (Streaming Layer Inference Pipeline), weights are not reloaded
per token during decode — the KV cache retains prior activations. For each
new token, only the *attention + FFN weight matrices* are streamed once per
layer. This is the standard derivation used in llama.cpp benchmarks.

    verify cost (per token) = Σ_layers [ layer_weight_bytes / bandwidth ]

For CDSC:
  - k tokens are drafted in VRAM via tiny LoRA voters (nanoseconds)
  - The target model does ONE batched forward pass across k+1 draft positions
    instead of k+1 individual ones. Batching k+1 tokens costs ~same as 1
    token in memory-bandwidth terms (the matmuls are bandwidth-bound, not
    compute-bound, for a single-batch decode).
  - Effective amortised cost per accepted token = (1 verify pass) / E[accept]

References
----------
- Leviathan et al. (2023) "Fast Inference from Transformers via Speculative Decoding"
- RTX 3060 spec sheet (NVIDIA, 2021)
- llama.cpp decode throughput analysis (ggerganov, 2024)
- PCIe 4.0 ×16 — ~14 GB/s host→device practical
- NVMe Gen4 (Samsung 990 Pro) — ~6.5 GB/s sequential read practical
"""

import math
from dataclasses import dataclass

# ─── Hardware Parameters ─────────────────────────────────────────────────────

@dataclass
class HardwareProfile:
    name: str
    vram_gb: float
    vram_bw_gbps: float       # GPU VRAM bandwidth
    pcie_bw_gbps: float       # Host→GPU (PCIe 4.0 ×16 practical)
    nvme_bw_gbps: float       # NVMe sequential read (practical)
    tflops_fp16: float

RTX_3060 = HardwareProfile(
    name="RTX 3060 12 GB",
    vram_gb=12.0,
    vram_bw_gbps=360.0,
    pcie_bw_gbps=14.0,
    nvme_bw_gbps=6.5,
    tflops_fp16=12.74,
)

# ─── Model Parameters ────────────────────────────────────────────────────────

@dataclass
class ModelProfile:
    name: str
    param_b: float
    quant_bits: int
    n_layers: int
    hidden_dim: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    context_len: int

LLAMA_70B_Q8 = ModelProfile(
    name="Llama-3.3-70B-Instruct-Q8_0",
    param_b=70.0,
    quant_bits=8,
    n_layers=80,
    hidden_dim=8192,
    n_kv_heads=8,
    head_dim=128,
    vocab_size=128256,
    context_len=4096,
)

# ─── CDSC Parameters ─────────────────────────────────────────────────────────

@dataclass
class CdscConfig:
    k_draft: int = 8
    epsilon: float = 0.15
    acceptance_rate: float = 0.85  # β per-token acceptance probability
    lora_rank: int = 16
    n_voters: int = 3
    voter_hidden_dim: int = 2048

# ─── Resource calculations ───────────────────────────────────────────────────

def model_size_gb(m: ModelProfile) -> float:
    return m.param_b * 1e9 * (m.quant_bits / 8) / 1e9

def kv_cache_size_gb(m: ModelProfile, batch: int = 1) -> float:
    # 2 (K,V) × n_layers × n_kv_heads × head_dim × context_len × 2 bytes (FP16)
    return 2 * m.n_layers * m.n_kv_heads * m.head_dim * m.context_len * batch * 2 / 1e9

def lora_voter_size_gb(c: CdscConfig, vocab_size: int) -> float:
    # A: (rank × in_dim)  +  B: (out_dim × rank)  per voter, FP16
    bytes_each = (c.lora_rank * c.voter_hidden_dim + vocab_size * c.lora_rank) * 2
    return c.n_voters * bytes_each / 1e9

# ─── Timing models (S.L.I.P. correct) ───────────────────────────────────────

def seconds_per_token_target(hw: HardwareProfile, m: ModelProfile) -> float:
    """
    Time to run one decode step through ALL layers of the target model.

    In S.L.I.P. the model streams from host storage. Each layer is:
      loaded → compute → freed.

    Bandwidth bottleneck during decode: PCIe (if weights are in host RAM)
    or NVMe (if mmap'd from disk). We model the slower NVMe case.

    Weight bytes per layer (approximate for Q8 transformer):
      attn: 4 × hidden² / 8   (Q,K,V,O projections)
      ffn:  3 × hidden × ffn_dim / 8   (gate, up, down)
    """
    ffn_dim = int(m.hidden_dim * 8 / 3)  # SwiGLU expansion ≈ 2.67×
    attn_bytes = 4 * m.hidden_dim ** 2 * m.quant_bits // 8
    ffn_bytes  = 3 * m.hidden_dim * ffn_dim * m.quant_bits // 8
    layer_bytes = attn_bytes + ffn_bytes

    # Practical bandwidth:
    # If model weights fit in host DRAM (70 GB needs 128+ GB server RAM),
    # use PCIe for uploads. On a consumer PC with 32–64 GB RAM, OS will
    # page-evict — effective bandwidth drops to NVMe speed.
    # We model NVMe (conservative / realistic for most RTX 3060 users).
    bw = hw.nvme_bw_gbps * 1e9
    per_layer_s = layer_bytes / bw
    return m.n_layers * per_layer_s

def seconds_council_draft(hw: HardwareProfile, m: ModelProfile, c: CdscConfig) -> float:
    """
    Time for one council draft pass producing k tokens.
    LoRA voters are fully VRAM-resident — cost is reading voter weights.
    With 3 voters × 2 matrices (A, B), each token needs:
      [rank × hidden] + [vocab × rank] elements read from VRAM.
    """
    voter_bytes = lora_voter_size_gb(c, m.vocab_size) * 1e9
    # All voters read once per draft token
    total_bytes = voter_bytes * c.k_draft
    return total_bytes / (hw.vram_bw_gbps * 1e9)

# ─── Expected tokens per round ───────────────────────────────────────────────

def expected_tokens_per_round(c: CdscConfig) -> float:
    """
    E[accepted] = Σ_{i=1}^{k} β^i  +  1 (unconditional bonus token)
    """
    beta = c.acceptance_rate
    return sum(beta ** i for i in range(1, c.k_draft + 1)) + 1.0

# ─── Throughput ──────────────────────────────────────────────────────────────

def throughput(hw: HardwareProfile, m: ModelProfile, c: CdscConfig, use_cdsc: bool) -> dict:
    t_ver = seconds_per_token_target(hw, m)
    t_dft = seconds_council_draft(hw, m, c)

    if use_cdsc:
        # ONE verify pass covers k+1 (draft+bonus) token positions.
        # Weight-bandwidth cost is ~identical to a single-token pass because
        # the matmuls are memory-bandwidth-bound, not compute-bound.
        # Draft pass overlaps GPU compute while verify runs on CPU/NVMe.
        # Model both steps fully sequential (conservative).
        round_s = t_dft + t_ver
        tpr = expected_tokens_per_round(c)
        tps = tpr / round_s
        mode = "CDSC (speculative council)"
    else:
        round_s = t_ver
        tpr = 1.0
        tps = 1.0 / t_ver
        mode = "Baseline (autoregressive)"

    baseline_tps = 1.0 / t_ver
    return dict(
        mode=mode, t_ver_ms=t_ver * 1000, t_dft_ms=t_dft * 1000,
        round_ms=round_s * 1000, tpr=tpr, tps=tps,
        speedup=tps / baseline_tps,
    )

# ─── Sensitivity sweep ────────────────────────────────────────────────────────

def sensitivity(hw: HardwareProfile, m: ModelProfile):
    print(f"\n  Sensitivity sweep: β × k  →  tok/s")
    print(f"  {'β':>6} {'k':>4}  {'tok/s':>10}  {'≥100?':>6}")
    print("  " + "─" * 34)
    for beta in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        for k in [4, 6, 8, 10, 12, 16]:
            cfg = CdscConfig(k_draft=k, acceptance_rate=beta)
            r = throughput(hw, m, cfg, use_cdsc=True)
            hit = "  ✓" if r["tps"] >= 100 else "  ✗"
            print(f"  {beta:>6.2f} {k:>4}  {r['tps']:>10.1f} {hit}")

# ─── What bandwidth do we *need* for 100 tok/s? ──────────────────────────────

def required_bandwidth_for_100_tps(m: ModelProfile, c: CdscConfig) -> float:
    """
    Solve for the NVMe/PCIe bandwidth that achieves 100 tok/s given CDSC.
    100 = E[accept] / (t_draft + t_verify)
    t_verify = n_layers * layer_bytes / bw
    → bw = n_layers * layer_bytes / (E[accept]/100 - t_draft)
    """
    ffn_dim = int(m.hidden_dim * 8 / 3)
    attn_bytes = 4 * m.hidden_dim ** 2 * m.quant_bits // 8
    ffn_bytes  = 3 * m.hidden_dim * ffn_dim * m.quant_bits // 8
    layer_bytes = attn_bytes + ffn_bytes
    total_weight_bytes = m.n_layers * layer_bytes

    tpr = expected_tokens_per_round(c)
    # t_draft is negligible vs t_verify; approximate as 0 for the bound
    # 100 = tpr / t_verify  →  t_verify = tpr / 100
    t_verify_needed = tpr / 100.0
    bw_needed = total_weight_bytes / t_verify_needed
    return bw_needed / 1e9  # GB/s

# ─── Memory budget ────────────────────────────────────────────────────────────

def memory_budget(hw: HardwareProfile, m: ModelProfile, c: CdscConfig):
    model_gb = model_size_gb(m)
    kv_gb    = kv_cache_size_gb(m)
    voter_gb = lora_voter_size_gb(c, m.vocab_size)

    print(f"\n  Memory Budget")
    print(f"  {'Component':<40} {'GB':>8}")
    print("  " + "─" * 50)
    print(f"  {'Target model Q8 (host RAM / NVMe)':<40} {model_gb:>8.1f}")
    print(f"  {'KV cache FP16 (ctx=' + str(m.context_len) + ')':<40} {kv_gb:>8.3f}")
    print(f"  {'LoRA voters ×3 (VRAM-resident, FP16)':<40} {voter_gb*1000:>7.1f}M")
    print(f"  {'VRAM available':<40} {hw.vram_gb:>8.1f}")
    total_vram = voter_gb + kv_gb
    fits = total_vram <= hw.vram_gb
    print(f"\n  Council VRAM usage : {total_vram:.3f} GB / {hw.vram_gb} GB  →  {'✓ FITS' if fits else '✗ OVERFLOW'}")
    return fits

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    hw = RTX_3060
    m  = LLAMA_70B_Q8
    c  = CdscConfig()
    bar = "═" * 62

    print(f"\n{bar}")
    print(f"  CDSC 100 tok/s — First-Principles Bandwidth Simulation")
    print(f"  Hardware : {hw.name}")
    print(f"  Model    : {m.name}")
    print(f"{bar}")

    memory_budget(hw, m, c)

    t_ver = seconds_per_token_target(hw, m)
    t_dft = seconds_council_draft(hw, m, c)
    tpr   = expected_tokens_per_round(c)
    bw_needed = required_bandwidth_for_100_tps(m, c)

    print(f"\n  Timing Breakdown  (k={c.k_draft}, ε={c.epsilon}, β={c.acceptance_rate})")
    print(f"  {'Metric':<44} {'Value':>14}")
    print("  " + "─" * 60)
    print(f"  {'Target verify pass (NVMe, 1 round)':<44} {t_ver*1000:>12.1f} ms")
    print(f"  {'Council draft pass (VRAM LoRA, k=' + str(c.k_draft) + ')':<44} {t_dft*1000:>12.3f} ms")
    print(f"  {'E[accepted tokens/round] (β=' + str(c.acceptance_rate) + ')':<44} {tpr:>14.2f}")
    print(f"  {'Bandwidth needed for 100 tok/s':<44} {bw_needed:>12.1f} GB/s")
    print(f"  {'RTX 3060 VRAM bandwidth':<44} {hw.vram_bw_gbps:>12.1f} GB/s")
    print(f"  {'PCIe 4.0 host→GPU bandwidth':<44} {hw.pcie_bw_gbps:>12.1f} GB/s")
    print(f"  {'NVMe Gen4 bandwidth (model source)':<44} {hw.nvme_bw_gbps:>12.1f} GB/s")

    print(f"\n  Throughput Comparison")
    print(f"  {'Mode':<44} {'tok/s':>8} {'Speedup':>9}")
    print("  " + "─" * 63)
    for use_cdsc in [False, True]:
        r = throughput(hw, m, c, use_cdsc)
        tag = " ← TARGET" if use_cdsc and r["tps"] >= 100 else ""
        print(f"  {r['mode']:<44} {r['tps']:>8.2f} {r['speedup']:>8.2f}×{tag}")

    sensitivity(hw, m)

    r = throughput(hw, m, c, use_cdsc=True)
    cdsc_tps = r["tps"]

    print(f"\n{bar}")
    print(f"  Results Summary")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  NVMe bottleneck : {hw.nvme_bw_gbps} GB/s  →  {cdsc_tps:.2f} tok/s (CDSC)")
    print(f"  PCIe model      : weights in host RAM (better consumer case)")
    bw_pcie = hw.pcie_bw_gbps
    # Recalculate with PCIe bandwidth
    layer_bytes = (4 * m.hidden_dim**2 + 3 * m.hidden_dim * int(m.hidden_dim*8/3)) * m.quant_bits // 8
    t_ver_pcie = m.n_layers * layer_bytes / (bw_pcie * 1e9)
    tps_pcie = tpr / (t_dft + t_ver_pcie)
    print(f"            PCIe BW = {bw_pcie} GB/s  →  {tps_pcie:.2f} tok/s (CDSC)")

    print(f"\n  To reach 100 tok/s you need ≥ {bw_needed:.1f} GB/s sustained bandwidth.")
    if bw_needed <= hw.vram_bw_gbps:
        print(f"  ✅ RTX 3060 VRAM bandwidth ({hw.vram_bw_gbps} GB/s) exceeds requirement — ")
        print(f"     achievable if the SPECULATIVE DRAFT keeps weights entirely in VRAM.")
    else:
        gap = bw_needed - max(hw.pcie_bw_gbps, hw.nvme_bw_gbps)
        print(f"  ⚠  Current I/O paths ({hw.nvme_bw_gbps} GB/s NVMe, {hw.pcie_bw_gbps} GB/s PCIe)")
        print(f"     fall {gap:.1f} GB/s short of the {bw_needed:.1f} GB/s needed.")
        print(f"  → CDSC achieves {cdsc_tps:.1f}× speedup vs baseline regardless.")
        print(f"  → On hardware with faster I/O (e.g. dual-GPU NVLink, or")
        print(f"     VRAM-fitting quantised model), the 100 tok/s target is reachable.")
    print(f"{bar}\n")

if __name__ == "__main__":
    main()
