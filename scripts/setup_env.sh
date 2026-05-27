#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — Air.rs Development Environment Setup
#
# One-command onboarding for contributors and users building from source.
# Checks all prerequisites, sets up a Python venv, and installs maturin.
#
# Usage:
#   chmod +x scripts/setup_env.sh
#   ./scripts/setup_env.sh
#   ./scripts/setup_env.sh --skip-python   # Rust/CUDA only
#   ./scripts/setup_env.sh --skip-venv     # no virtualenv (use system Python)
# =============================================================================

set -euo pipefail

# ── Color helpers ─────────────────────────────────────────────────────────────
if [ -t 1 ] && command -v tput &>/dev/null && tput colors &>/dev/null 2>&1; then
    RED=$(tput setaf 1);   GREEN=$(tput setaf 2);  YELLOW=$(tput setaf 3)
    BLUE=$(tput setaf 4);  CYAN=$(tput setaf 6);   BOLD=$(tput bold)
    RESET=$(tput sgr0);    MAGENTA=$(tput setaf 5)
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; CYAN=''; BOLD=''; RESET=''; MAGENTA=''
fi

ok()   { echo "${GREEN}  [✓]${RESET} $*"; }
info() { echo "${CYAN}  [i]${RESET} $*"; }
warn() { echo "${YELLOW}  [!]${RESET} $*"; }
err()  { echo "${RED}  [✗]${RESET} $*" >&2; }
die()  { err "$*"; exit 1; }
hdr()  { echo ""; echo "${BOLD}${MAGENTA}  ── $* ──${RESET}"; echo ""; }

SKIP_PYTHON=false
SKIP_VENV=false

for arg in "$@"; do
    case "$arg" in
        --skip-python) SKIP_PYTHON=true ;;
        --skip-venv)   SKIP_VENV=true   ;;
        --help|-h)
            echo "Usage: $0 [--skip-python] [--skip-venv]"
            exit 0
            ;;
    esac
done

echo ""
echo "${MAGENTA}  ╔══════════════════════════════════════════════════════╗${RESET}"
echo "${MAGENTA}  ║       Air.rs Dev Environment Setup                   ║${RESET}"
echo "${MAGENTA}  ╚══════════════════════════════════════════════════════╝${RESET}"
echo ""

ERRORS=0

# =============================================================================
# STEP 1: Rust toolchain
# =============================================================================
hdr "Step 1: Rust Toolchain"

if ! command -v cargo &>/dev/null; then
    err "cargo not found."
    echo "  Install Rust via rustup: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    ERRORS=$((ERRORS + 1))
else
    RUST_VER=$(rustc --version)
    ok "Rust: $RUST_VER"

    # Check minimum version (1.75)
    RUST_MINOR=$(rustc --version | grep -oP '1\.\K\d+' | head -1)
    if [ "${RUST_MINOR:-0}" -lt 75 ]; then
        warn "Rust 1.75+ required. Run: rustup update stable"
    fi
fi

# =============================================================================
# STEP 2: C/C++ toolchain
# =============================================================================
hdr "Step 2: C/C++ Toolchain"

OS="$(uname -s)"
if [ "$OS" = "Linux" ]; then
    if command -v cc &>/dev/null; then
        ok "C compiler: $(cc --version 2>&1 | head -1)"
    else
        warn "C compiler not found."
        echo "  Install: sudo apt install build-essential  (Debian/Ubuntu)"
        echo "           sudo dnf groupinstall 'Development Tools'  (Fedora/RHEL)"
        ERRORS=$((ERRORS + 1))
    fi
    if command -v pkg-config &>/dev/null; then
        ok "pkg-config: $(pkg-config --version)"
    else
        warn "pkg-config not found: sudo apt install pkg-config"
    fi
    if dpkg -s libssl-dev &>/dev/null 2>&1 || rpm -q openssl-devel &>/dev/null 2>&1; then
        ok "libssl-dev: present"
    else
        warn "libssl-dev not found: sudo apt install libssl-dev"
    fi
elif [ "$OS" = "Darwin" ]; then
    if xcode-select -p &>/dev/null; then
        ok "Xcode CLI Tools: $(xcode-select -p)"
    else
        warn "Xcode CLI Tools not found. Run: xcode-select --install"
        ERRORS=$((ERRORS + 1))
    fi
fi

# =============================================================================
# STEP 3: GPU backends (optional)
# =============================================================================
hdr "Step 3: GPU Backends (Optional)"

# NVIDIA CUDA
if command -v nvidia-smi &>/dev/null; then
    GPU_LINE=$(nvidia-smi -L 2>/dev/null | head -1 || echo "unknown")
    ok "NVIDIA GPU: ${GPU_LINE#GPU 0: }"
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version 2>&1 | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        ok "CUDA Toolkit: $CUDA_VER"
        info "  Build with: cargo build --release --features cuda,flash-attn"
    else
        warn "CUDA Toolkit not in PATH (NVIDIA GPU detected)"
        info "  export CUDA_HOME=/usr/local/cuda  # then re-run"
    fi
else
    info "No NVIDIA GPU detected — CPU build will work"
fi

# AMD ROCm
if command -v hipcc &>/dev/null || [ -d /opt/rocm ]; then
    ROCM_VER=$(cat /opt/rocm/.version 2>/dev/null || echo "unknown")
    ok "AMD ROCm: $ROCM_VER"
    info "  Build with: cargo build --release --features rocm"
else
    info "ROCm not detected"
fi

# Apple Metal
if [ "$OS" = "Darwin" ]; then
    if system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Metal"; then
        METAL_GPU=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Chipset Model:" | head -1 | sed 's/.*: //')
        ok "Apple Metal: ${METAL_GPU:-detected}"
        info "  Build with: cargo build --release --features metal"
    else
        info "Metal not detected (Intel Mac — use CPU build)"
    fi
fi

# Vulkan
if command -v vulkaninfo &>/dev/null 2>&1; then
    if vulkaninfo 2>/dev/null | grep -q "Vulkan Instance Version"; then
        VK_VER=$(vulkaninfo 2>/dev/null | grep "Vulkan Instance Version" | head -1 | sed 's/.*: //')
        ok "Vulkan: $VK_VER"
        info "  Build with: cargo build --release --features vulkan"
    fi
else
    info "Vulkan not detected"
fi

# =============================================================================
# STEP 4: Python environment
# =============================================================================
if ! $SKIP_PYTHON; then
    hdr "Step 4: Python Environment"

    if ! command -v python3 &>/dev/null; then
        warn "python3 not found — skipping Python setup"
        info "  Install Python 3.11+: https://python.org/downloads"
    else
        PY_VER=$(python3 --version)
        ok "Python: $PY_VER"

        PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
        if [ "${PY_MINOR:-0}" -lt 11 ]; then
            warn "Python 3.11+ required for air-rs wheels"
        fi

        if ! $SKIP_VENV; then
            if [ ! -d ".venv" ]; then
                info "Creating virtual environment: .venv"
                python3 -m venv .venv
                ok ".venv created"
            else
                ok ".venv already exists"
            fi

            VENV_PIP=".venv/bin/pip"
            VENV_PYTHON=".venv/bin/python"

            info "Upgrading pip..."
            "$VENV_PIP" install --upgrade pip --quiet

            # Install maturin (for building from source)
            if ! "$VENV_PIP" show maturin &>/dev/null 2>&1; then
                info "Installing maturin..."
                "$VENV_PIP" install "maturin>=1.7,<2" --quiet
                ok "maturin installed: $("$VENV_PYTHON" -m maturin --version)"
            else
                ok "maturin: $("$VENV_PYTHON" -m maturin --version)"
            fi

            # Install dev dependencies
            if [ -f "pyproject.toml" ]; then
                info "Installing Python dev dependencies..."
                "$VENV_PIP" install -e ".[dev]" --quiet 2>/dev/null || \
                    "$VENV_PIP" install pytest mypy ruff --quiet
                ok "Dev deps installed (pytest, mypy, ruff)"
            fi

            echo ""
            info "Activate venv: source .venv/bin/activate"
            info "Build Python extension: maturin develop --features python"
        else
            # Skip venv — check maturin globally
            if command -v maturin &>/dev/null; then
                ok "maturin: $(maturin --version)"
            else
                warn "maturin not found. Install: pip install maturin"
            fi
        fi
    fi
fi

# =============================================================================
# STEP 5: Verify cargo check passes
# =============================================================================
hdr "Step 5: Cargo Check (CPU build)"

info "Running: cargo check (CPU-only, no feature flags) ..."
if cargo check --quiet 2>&1; then
    ok "cargo check passed"
else
    err "cargo check failed — see errors above"
    ERRORS=$((ERRORS + 1))
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
if [ "$ERRORS" -eq 0 ]; then
    echo "${GREEN}  ╔══════════════════════════════════════════════════════╗${RESET}"
    echo "${GREEN}  ║  Environment setup complete — ready to build!        ║${RESET}"
    echo "${GREEN}  ╚══════════════════════════════════════════════════════╝${RESET}"
    echo ""
    echo "${BOLD}  Next steps:${RESET}"
    echo "  1.  ./build_air.sh --skip-prompt      # build with all detected features"
    echo "  2.  ./scripts/test_all.sh              # run full test suite"
    echo "  3.  ./scripts/build_wheel.sh           # build Python wheel (optional)"
else
    echo "${RED}  ╔══════════════════════════════════════════════════════╗${RESET}"
    echo "${RED}  ║  Setup completed with ${ERRORS} error(s) — see warnings above ║${RESET}"
    echo "${RED}  ╚══════════════════════════════════════════════════════╝${RESET}"
    exit 1
fi
