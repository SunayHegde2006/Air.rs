#!/usr/bin/env bash
# =============================================================================
# build_wheel.sh — Air.rs Python Wheel Builder
#
# Builds the maturin wheel for the `air-rs` Python package.
# Outputs to dist/ and prints the install command.
#
# Usage:
#   chmod +x scripts/build_wheel.sh
#   ./scripts/build_wheel.sh                    # release wheel (current platform)
#   ./scripts/build_wheel.sh --debug            # debug wheel (faster compile)
#   ./scripts/build_wheel.sh --manylinux        # manylinux2014 wheel (for PyPI)
#   ./scripts/build_wheel.sh --features cuda    # include CUDA in wheel
#   ./scripts/build_wheel.sh --install          # build + pip install into venv
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
err()  { echo "${RED}  [✗]${RESET} $*" >&2; }
die()  { err "$*"; exit 1; }

# ── Defaults ──────────────────────────────────────────────────────────────────
DEBUG=false
MANYLINUX=false
INSTALL=false
EXTRA_FEATURES="python"   # always include python bindings

for arg in "$@"; do
    case "$arg" in
        --debug)         DEBUG=true ;;
        --manylinux)     MANYLINUX=true ;;
        --install)       INSTALL=true ;;
        --features=*)    EXTRA_FEATURES="python,${arg#--features=}" ;;
        --features)      ;;
        --help|-h)
            echo "Usage: $0 [--debug] [--manylinux] [--install] [--features=cuda]"
            exit 0
            ;;
        *)
            if [[ "${prev_arg:-}" == "--features" ]]; then
                EXTRA_FEATURES="python,$arg"
            fi
            ;;
    esac
    prev_arg="$arg"
done

echo ""
echo "${MAGENTA}  ╔══════════════════════════════════════════════════════╗${RESET}"
echo "${MAGENTA}  ║       Air.rs Python Wheel Builder                    ║${RESET}"
echo "${MAGENTA}  ╚══════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Find maturin ──────────────────────────────────────────────────────────────
MATURIN=""
if [ -f ".venv/bin/maturin" ]; then
    MATURIN=".venv/bin/maturin"
elif command -v maturin &>/dev/null; then
    MATURIN="maturin"
else
    die "maturin not found. Run: ./scripts/setup_env.sh  (or pip install maturin)"
fi

MATURIN_VER=$("$MATURIN" --version)
ok "maturin: $MATURIN_VER"

# ── Find pip for --install ────────────────────────────────────────────────────
PIP=""
if $INSTALL; then
    if [ -f ".venv/bin/pip" ]; then
        PIP=".venv/bin/pip"
    elif command -v pip3 &>/dev/null; then
        PIP="pip3"
    else
        warn "--install requested but pip not found — skipping install step"
        INSTALL=false
    fi
fi

# ── Build command ─────────────────────────────────────────────────────────────
mkdir -p dist

if $MANYLINUX; then
    BUILD_MODE="build --manylinux 2014"
    info "Mode: manylinux2014 (for PyPI upload)"
elif $DEBUG; then
    BUILD_MODE="develop"
    info "Mode: develop (debug — editable install in venv)"
else
    BUILD_MODE="build --release"
    info "Mode: release (optimized wheel → dist/)"
fi

CMD="$MATURIN $BUILD_MODE --features $EXTRA_FEATURES --out dist/"
info "Features: $EXTRA_FEATURES"
info "Running: $CMD"
echo ""

START=$(date +%s)
eval "$CMD"
END=$(date +%s)
BUILD_TIME=$((END - START))

echo ""
ok "Build succeeded in ${BUILD_TIME}s"

# ── Find the wheel ────────────────────────────────────────────────────────────
WHEEL=$(ls -t dist/air_rs-*.whl dist/air-rs-*.whl 2>/dev/null | head -1 || echo "")
if [ -n "$WHEEL" ]; then
    WHEEL_SIZE=$(du -sh "$WHEEL" | cut -f1)
    ok "Wheel: $WHEEL ($WHEEL_SIZE)"
fi

# ── Install if requested ──────────────────────────────────────────────────────
if $INSTALL && [ -n "$WHEEL" ]; then
    echo ""
    info "Installing wheel into Python environment..."
    "$PIP" install --force-reinstall "$WHEEL" --quiet
    ok "Installed!"

    echo ""
    info "Verify:"
    echo "  python3 -c \"import air_rs; print(air_rs.__version__)\""
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "${BOLD}  ══════════════════════════════════════════════════════${RESET}"
echo "${BOLD}  Output: dist/${RESET}"
if [ -n "$WHEEL" ]; then
    echo ""
    echo "  Install wheel:"
    echo "    ${GREEN}pip install $WHEEL${RESET}"
    echo ""
    echo "  Upload to PyPI (maintainers):"
    echo "    ${CYAN}twine upload dist/air_rs-*.whl${RESET}"
fi
echo "${BOLD}  ══════════════════════════════════════════════════════${RESET}"
echo ""
