#!/usr/bin/env bash
# =============================================================================
# test_all.sh — Air.rs Full Test Suite Runner
#
# Runs all Rust tests (with and without feature flags), Python binding tests,
# and a benchmark smoke test. Exit code propagates first failure.
#
# Usage:
#   chmod +x scripts/test_all.sh
#   ./scripts/test_all.sh
#   ./scripts/test_all.sh --fast         # skip slow integration tests
#   ./scripts/test_all.sh --features cuda # test with GPU features
# =============================================================================

set -euo pipefail

# ── Color helpers ─────────────────────────────────────────────────────────────
if [ -t 1 ] && command -v tput &>/dev/null && tput colors &>/dev/null 2>&1; then
    RED=$(tput setaf 1);   GREEN=$(tput setaf 2);  YELLOW=$(tput setaf 3)
    CYAN=$(tput setaf 6);  BOLD=$(tput bold);       RESET=$(tput sgr0)
    MAGENTA=$(tput setaf 5)
else
    RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; RESET=''; MAGENTA=''
fi

ok()   { echo "${GREEN}  [✓]${RESET} $*"; }
info() { echo "${CYAN}  [i]${RESET} $*"; }
warn() { echo "${YELLOW}  [!]${RESET} $*"; }
fail() { echo "${RED}  [✗]${RESET} $*" >&2; }
hdr()  { echo ""; echo "${BOLD}${MAGENTA}  ── $* ──${RESET}"; echo ""; }

# ── Arg parsing ───────────────────────────────────────────────────────────────
FAST=false
EXTRA_FEATURES=""

for arg in "$@"; do
    case "$arg" in
        --fast)          FAST=true ;;
        --features=*)    EXTRA_FEATURES="${arg#--features=}" ;;
        --features)      ;;
        --help|-h)
            echo "Usage: $0 [--fast] [--features=cuda,flash-attn]"
            exit 0
            ;;
        *)
            if [[ "${prev_arg:-}" == "--features" ]]; then
                EXTRA_FEATURES="$arg"
            fi
            ;;
    esac
    prev_arg="$arg"
done

# Build feature arg
FEATURE_ARG=""
if [ -n "$EXTRA_FEATURES" ]; then
    FEATURE_ARG="--features $EXTRA_FEATURES"
fi

echo ""
echo "${MAGENTA}  ╔══════════════════════════════════════════════════════╗${RESET}"
echo "${MAGENTA}  ║       Air.rs Test Suite                               ║${RESET}"
echo "${MAGENTA}  ╚══════════════════════════════════════════════════════╝${RESET}"
echo ""

PASS=0
FAIL=0
TOTAL_START=$(date +%s)

run_suite() {
    local name="$1"
    local cmd="$2"
    local start
    start=$(date +%s)
    info "Running: $cmd"
    if eval "$cmd"; then
        local end
        end=$(date +%s)
        ok "$name passed ($(( end - start ))s)"
        PASS=$((PASS + 1))
    else
        local end
        end=$(date +%s)
        fail "$name FAILED ($(( end - start ))s)"
        FAIL=$((FAIL + 1))
    fi
}

# =============================================================================
# SUITE 1: Rust unit tests (no features — always must pass)
# =============================================================================
hdr "Suite 1: Rust Unit Tests (CPU-only)"
run_suite "cargo test (CPU)" "cargo test --lib --quiet 2>&1 | tail -5"

# =============================================================================
# SUITE 2: Rust doc tests
# =============================================================================
hdr "Suite 2: Doc Tests"
run_suite "cargo test --doc" "cargo test --doc --quiet 2>&1 | tail -5"

# =============================================================================
# SUITE 3: Integration tests
# =============================================================================
if ! $FAST; then
    hdr "Suite 3: Integration Tests"
    run_suite "cargo test --tests" "cargo test --tests --quiet 2>&1 | tail -10"
else
    warn "Skipping integration tests (--fast)"
fi

# =============================================================================
# SUITE 4: Feature-gated tests (if requested)
# =============================================================================
if [ -n "$EXTRA_FEATURES" ]; then
    hdr "Suite 4: Feature Tests ($EXTRA_FEATURES)"
    run_suite "cargo test $FEATURE_ARG" "cargo test $FEATURE_ARG --quiet 2>&1 | tail -10"
fi

# =============================================================================
# SUITE 5: Clippy lints
# =============================================================================
hdr "Suite 5: Clippy (0 warnings policy)"
run_suite "cargo clippy" "cargo clippy -- -D warnings 2>&1 | tail -5"

# =============================================================================
# SUITE 6: Python tests (if venv exists)
# =============================================================================
hdr "Suite 6: Python Binding Tests"

PY_BIN=""
if [ -f ".venv/bin/python" ]; then
    PY_BIN=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PY_BIN="python3"
fi

if [ -n "$PY_BIN" ] && [ -d "python/tests" ]; then
    if "$PY_BIN" -c "import air_rs" 2>/dev/null; then
        run_suite "pytest python/tests/" "$PY_BIN -m pytest python/tests/ -q 2>&1 | tail -10"
    else
        warn "air_rs not installed in Python env — skipping Python tests"
        info "  Build first: maturin develop --features python"
    fi
else
    warn "No Python tests found (python/tests/ missing or no Python interpreter)"
fi

# =============================================================================
# SUITE 7: Benchmark smoke test (compile only, no run)
# =============================================================================
hdr "Suite 7: Benchmark Compile Check"
run_suite "cargo bench --no-run" "cargo bench --no-run --quiet 2>&1 | tail -3"

# =============================================================================
# Summary
# =============================================================================
TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo ""
echo "${BOLD}  ══════════════════════════════════════════════════════${RESET}"
printf  "  ${BOLD}Results:${RESET}  ${GREEN}%d passed${RESET}  ${RED}%d failed${RESET}  (${TOTAL_TIME}s total)\n" \
    "$PASS" "$FAIL"
echo "${BOLD}  ══════════════════════════════════════════════════════${RESET}"
echo ""

if [ "$FAIL" -gt 0 ]; then
    fail "$FAIL test suite(s) failed"
    exit 1
fi

ok "All suites passed!"
