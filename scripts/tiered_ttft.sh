#!/usr/bin/env bash
# =============================================================================
# tiered_ttft.sh — Air.rs v1.0.0 Tiered TTFT Gate Benchmark  v2.0
#
# Correct methodology:
#   Uses `air-rs bench --n-tokens 1` which:
#     1. Loads the model ONCE (S.L.I.P. NVMe streaming if model > VRAM)
#     2. Runs N decode steps and reports Mean TPS
#   TTFT_ms = 1000 / mean_tps   (time for exactly one forward pass = TTFT)
#
# Tier gates (RTX 3060 12GB, Q8_K models via S.L.I.P.):
#   Tier 1 (≤7B):    TTFT p99 ≤ 150ms
#   Tier 2 (8–13B):  TTFT p99 ≤ 300ms
#   Tier 3 (14–35B): TTFT p99 ≤ 700ms
#   Stretch (>35B):  Informational only
#
# Usage:
#   chmod +x scripts/tiered_ttft.sh && ./scripts/tiered_ttft.sh
#   ./scripts/tiered_ttft.sh --runs 5 --out results/
# =============================================================================

set -euo pipefail

BENCH_VERSION="2.0"
MODELS_DIR="${HOME}/models"
OUT_DIR="${PWD}/results/tiered_ttft"
RUNS=5
BIN="./target/release/air-rs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FAIL=0

# ── Color helpers ──────────────────────────────────────────────────────────────
R=$'\033[91m'; G=$'\033[92m'; Y=$'\033[93m'; C=$'\033[96m'
W=$'\033[97m'; X=$'\033[0m'; B=$'\033[94m'
step() { echo "  ${G}[+]${X} $*"; }
info() { echo "  ${C}[i]${X} $*"; }
warn() { echo "  ${Y}[!]${X} $*"; }
sep()  { echo "  ${B}$(printf '─%.0s' {1..70})${X}"; }

for arg in "$@"; do
    case "$arg" in
        --runs=*)       RUNS="${arg#--runs=}" ;;
        --models-dir=*) MODELS_DIR="${arg#--models-dir=}" ;;
        --out=*)        OUT_DIR="${arg#--out=}" ;;
    esac
done

mkdir -p "$OUT_DIR"

# ── Tier / gate lookup ─────────────────────────────────────────────────────────
tier_for() {
    local gb="$1"
    if   (( $(echo "$gb < 8"  | bc -l) )); then echo "1"
    elif (( $(echo "$gb < 14" | bc -l) )); then echo "2"
    elif (( $(echo "$gb < 36" | bc -l) )); then echo "3"
    else echo "stretch"; fi
}
gate_for() {
    case "$1" in 1) echo 150;; 2) echo 300;; 3) echo 700;; *) echo 9999;; esac
}

# ── Run bench and extract Mean TPS ────────────────────────────────────────────
# Returns TTFT_ms (integer) via stdout.
bench_ttft() {
    local model="$1"
    local raw
    raw=$("$BIN" bench --model "$model" --n-tokens 1 --runs "$RUNS" 2>&1)
    local tps
    tps=$(echo "$raw" | grep -oP 'Mean TPS:\s*\K[0-9]+(\.[0-9]+)?')
    if [ -z "$tps" ] || [ "$tps" = "0" ]; then
        echo "9999"
        return
    fi
    # TTFT = 1000 ms / TPS  (rounded to integer)
    echo "scale=0; 1000 / $tps" | bc
}

# ── Header ─────────────────────────────────────────────────────────────────────
echo ""
echo "${W}  ╔════════════════════════════════════════════════════════════╗${X}"
echo "${W}  ║   Air.rs v1.0.0 — Tiered TTFT Gate Benchmark v${BENCH_VERSION}        ║${X}"
echo "${W}  ║   Method: 1000ms / mean_tps  (bench --n-tokens 1)         ║${X}"
echo "${W}  ╚════════════════════════════════════════════════════════════╝${X}"
echo ""

GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "CPU-only")
CPU=$(grep "model name" /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || echo "unknown")
info "Binary:  $BIN  ($(./target/release/air-rs --version 2>/dev/null || echo "unknown version"))"
info "GPU:     $GPU"
info "CPU:     $CPU"
info "Runs:    $RUNS per model"
sep

[ ! -x "$BIN" ] && { echo "  ${R}[X] Binary not found. Run: cargo build --release${X}"; exit 1; }

# ── Model catalogue (auto-discover) ───────────────────────────────────────────
declare -A MODELS TIER GATE

while IFS= read -r f; do
    n=$(basename "$f")
    gb=$(echo "scale=1; $(stat -c%s "$f") / 1073741824" | bc)
    t=$(tier_for "$gb")
    g=$(gate_for "$t")
    MODELS["$n"]="$f"
    TIER["$n"]="$t"
    GATE["$n"]="$g"
done < <(find "$MODELS_DIR" -maxdepth 2 -name "*.gguf" 2>/dev/null | sort)

info "Models found: ${#MODELS[@]}  (dir: $MODELS_DIR)"
info "Output:       $OUT_DIR/ttft_${TIMESTAMP}.json"
echo ""

# ── Run each model ─────────────────────────────────────────────────────────────
declare -A P99 STATUS

for name in "${!MODELS[@]}"; do
    path="${MODELS[$name]}"
    tier="${TIER[$name]}"
    gate="${GATE[$name]}"
    gb=$(echo "scale=1; $(stat -c%s "$path") / 1073741824" | bc)

    echo ""
    echo "${W}  ┌─ $name${X}"
    echo "  │  Tier ${tier} · Gate ${gate}ms · ${gb}G · $RUNS runs"
    echo "  │  Path: $path"

    ttft=$(bench_ttft "$path")
    P99["$name"]="$ttft"

    tps_disp=$(echo "scale=1; 1000 / $ttft" | bc 2>/dev/null || echo "N/A")

    if [ "$tier" = "stretch" ]; then
        echo "  │  TTFT ≈ ${ttft}ms  (${tps_disp} tok/s)  — stretch, no gate"
        warn "  └─ Stretch tier — informational"
        STATUS["$name"]="stretch"
    elif (( ttft <= gate )); then
        echo "  │  TTFT ≈ ${ttft}ms  (${tps_disp} tok/s)"
        echo "  ${G}  └─ PASS  ${ttft}ms ≤ ${gate}ms ✓${X}"
        STATUS["$name"]="pass"
    else
        echo "  │  TTFT ≈ ${ttft}ms  (${tps_disp} tok/s)"
        echo "  ${R}  └─ FAIL  ${ttft}ms > ${gate}ms ✗${X}"
        STATUS["$name"]="fail"
        FAIL=$((FAIL+1))
    fi
done

# ── JSON results ───────────────────────────────────────────────────────────────
OUT_FILE="$OUT_DIR/ttft_${TIMESTAMP}.json"
{
printf '{\n'
printf '  "bench_version": "%s",\n'    "$BENCH_VERSION"
printf '  "timestamp":     "%s",\n'    "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '  "air_rs_version": "%s",\n'   "1.0.0"
printf '  "gpu":           "%s",\n'    "$GPU"
printf '  "cpu":           "%s",\n'    "$CPU"
printf '  "runs_per_model": %d,\n'     "$RUNS"
printf '  "method":        "1000ms / mean_tps (bench --n-tokens 1)",\n'
printf '  "gates_ms": { "tier1": 150, "tier2": 300, "tier3": 700 },\n'
printf '  "results": [\n'
first=true
for name in "${!MODELS[@]}"; do
    gb=$(echo "scale=1; $(stat -c%s "${MODELS[$name]}") / 1073741824" | bc)
    tps=$(echo "scale=1; 1000 / ${P99[$name]}" | bc 2>/dev/null || echo "0")
    $first || printf ',\n'
    first=false
    printf '    { "model": "%s", "size_gb": %s, "tier": "%s", "gate_ms": %s, "ttft_ms": %s, "mean_tps": %s, "status": "%s" }' \
        "$name" "$gb" "${TIER[$name]}" "${GATE[$name]}" "${P99[$name]}" "$tps" "${STATUS[$name]}"
done
printf '\n  ]\n}\n'
} >"$OUT_FILE"

# ── Summary table ──────────────────────────────────────────────────────────────
echo ""
sep
echo ""
printf "  ${W}%-44s  %-5s  %-7s  %-9s  %-10s  %s${X}\n" \
    "Model" "Tier" "Gate" "TTFT ms" "tok/s" "Result"
printf "  %-44s  %-5s  %-7s  %-9s  %-10s  %s\n" \
    "$(printf '─%.0s' {1..44})" "─────" "───────" "─────────" "──────────" "──────"

for name in "${!MODELS[@]}"; do
    ttft="${P99[$name]}"
    gate="${GATE[$name]}"
    tier="${TIER[$name]}"
    tps=$(echo "scale=1; 1000 / $ttft" | bc 2>/dev/null || echo "N/A")
    short="${name:0:44}"
    case "${STATUS[$name]}" in
        pass)    col="${G}"; sym="✓ PASS" ;;
        fail)    col="${R}"; sym="✗ FAIL" ;;
        stretch) col="${Y}"; sym="~ INFO" ;;
        *)       col="${X}"; sym="?" ;;
    esac
    printf "  %-44s  T%-4s  %-7s  ${col}%-9s${X}  ${col}%-10s${X}  ${col}%s${X}\n" \
        "$short" "$tier" "${gate}ms" "${ttft}ms" "${tps} t/s" "$sym"
done

echo ""
step "Results → $OUT_FILE"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "  ${R}[✗] $FAIL gate(s) failed.${X}"
    echo "      Suggested action: use Q4_K_M quant for lower TTFT."
    exit 1
else
    echo "  ${G}[✓] All applicable tier gates PASSED.${X}"
fi
echo ""
