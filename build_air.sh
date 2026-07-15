#!/usr/bin/env bash
# =============================================================================
# build_air.sh — Air.rs Unified Build Script (macOS & Linux)
#
# USAGE:
#   ./build_air.sh              # Interactive feature selection
#   ./build_air.sh --release    # Release build (default)
#   ./build_air.sh --debug      # Debug build
#   ./build_air.sh --skip-prompt  # Use all available features silently
#   ./build_air.sh --features cuda,flash-attn  # Explicit feature list
#
# Mirrors build_air.ps1 behaviour exactly.
# =============================================================================

set -euo pipefail

# ── Color helpers ────────────────────────────────────────────────────────────
if [ -t 1 ] && command -v tput &>/dev/null && tput colors &>/dev/null; then
    RED=$(tput setaf 1 2>/dev/null || echo "")
    GREEN=$(tput setaf 2 2>/dev/null || echo "")
    YELLOW=$(tput setaf 3 2>/dev/null || echo "")
    BLUE=$(tput setaf 4 2>/dev/null || echo "")
    CYAN=$(tput setaf 6 2>/dev/null || echo "")
    BOLD=$(tput bold 2>/dev/null || echo "")
    RESET=$(tput sgr0 2>/dev/null || echo "")
    MAGENTA=$(tput setaf 5 2>/dev/null || echo "")
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; CYAN=''; BOLD=''; RESET=''; MAGENTA=''
fi

step()  { echo "${GREEN}  [+]${RESET} $*"; }
info()  { echo "${CYAN}  [i]${RESET} $*"; }
warn()  { echo "${YELLOW}  [!]${RESET} $*"; }
err()   { echo "${RED}  [X]${RESET} $*" >&2; }
die()   { err "$*"; exit 1; }

# ── Argument parsing ─────────────────────────────────────────────────────────
DEBUG_BUILD=false
SKIP_PROMPT=false
EXPLICIT_FEATURES=""

for arg in "$@"; do
    case "$arg" in
        --debug)                DEBUG_BUILD=true ;;
        --release)              ;;   # default
        --skip-prompt)          SKIP_PROMPT=true ;;
        --features=*)           EXPLICIT_FEATURES="${arg#--features=}" ;;
        --features)             ;;   # handled below (next arg)
        *)
            # handle: --features cuda,flash-attn (two args)
            if [[ "${prev_arg:-}" == "--features" ]]; then
                EXPLICIT_FEATURES="$arg"
            fi
            ;;
    esac
    prev_arg="$arg"
done

# ── Banner ───────────────────────────────────────────────────────────────────
echo ""
echo "${MAGENTA}  ======================================================${RESET}"
echo "${MAGENTA}       Air.rs Build System — v1.1.5 (Stable)            ${RESET}"
echo "${MAGENTA}  ======================================================${RESET}"
echo ""

# =============================================================================
# STEP 1: HARDWARE DETECTION
# =============================================================================
echo "${BOLD}  --- Step 1: Hardware Detection ---${RESET}"
echo ""

OS="$(uname -s)"
ARCH="$(uname -m)"
info "OS: $OS | Arch: $ARCH"

# ── NVIDIA GPU ────────────────────────────────────────────────────────────────
HAS_GPU=false
GPU_NAME=""
if command -v nvidia-smi &>/dev/null; then
    if GPU_LINE=$(nvidia-smi -L 2>/dev/null | head -1); then
        HAS_GPU=true
        GPU_NAME="${GPU_LINE#GPU 0: }"
        GPU_NAME="${GPU_NAME%% (UUID:*}"
        step "NVIDIA GPU: $GPU_NAME"
    fi
fi
if ! $HAS_GPU; then
    info "No NVIDIA GPU detected (CPU/Metal builds will work)"
fi

# ── GPU Architecture ──────────────────────────────────────────────────────────
GPU_ARCH=""
if $HAS_GPU && command -v nvidia-smi &>/dev/null; then
    # Query all GPU compute capabilities, strip dots, sort, and deduplicate
    if COMPUTE_CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '.' | sort -u); then
        CAPS_ARR=()
        for cap in $COMPUTE_CAPS; do
            if [[ "$cap" =~ ^[0-9]+$ ]]; then
                CAPS_ARR+=("sm_${cap}")
            fi
        done
        if [ ${#CAPS_ARR[@]} -gt 0 ]; then
            GPU_ARCH=$(IFS=','; echo "${CAPS_ARR[*]}")
            export NVCC_ARCH="${NVCC_ARCH:-$GPU_ARCH}"
            step "GPU arch targeting: ${NVCC_ARCH} (override via NVCC_ARCH=all or list)"
        fi
    fi
fi

# ── CUDA Toolkit ──────────────────────────────────────────────────────────────
HAS_CUDA=false
CUDA_VERSION=""
if command -v nvcc &>/dev/null; then
    if NVCC_OUT=$(nvcc --version 2>&1 | grep "release"); then
        HAS_CUDA=true
        CUDA_VERSION=$(echo "$NVCC_OUT" | sed 's/.*release //' | sed 's/,.*//')
        # CUDA 13.3+ Support Logic
        if [[ $CUDA_VERSION == 13.* ]]; then
            step "CUDA Toolkit: $CUDA_VERSION (High-Performance CUDA 13 Mode)"
        else
            step "CUDA Toolkit: $CUDA_VERSION"
        fi
    fi
elif [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
    HAS_CUDA=true
    CUDA_VERSION=$("$CUDA_HOME/bin/nvcc" --version 2>&1 | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    step "CUDA Toolkit (via CUDA_HOME): $CUDA_VERSION"
else
    if $HAS_GPU; then warn "NVIDIA GPU found but CUDA Toolkit not in PATH"; fi
fi

# ── Apple Metal ───────────────────────────────────────────────────────────────
HAS_METAL=false
if [ "$OS" = "Darwin" ]; then
    # Check for Apple Silicon or Metal-capable GPU
    if system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Metal"; then
        HAS_METAL=true
        METAL_GPU=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Chipset Model:" | head -1 | sed 's/.*: //')
        step "Apple Metal: ${METAL_GPU:-GPU detected}"
    else
        info "Metal not detected (GPU may not support it)"
    fi
fi

# ── ROCm ──────────────────────────────────────────────────────────────────────
HAS_ROCM=false
if command -v hipcc &>/dev/null || [ -d /opt/rocm ]; then
    HAS_ROCM=true
    ROCM_VERSION=$(cat /opt/rocm/.version 2>/dev/null || echo "unknown")
    step "AMD ROCm: $ROCM_VERSION"
fi

# ── Vulkan ────────────────────────────────────────────────────────────────────
HAS_VULKAN=false
if command -v vulkaninfo &>/dev/null 2>&1; then
    if vulkaninfo 2>/dev/null | grep -q "Vulkan Instance Version"; then
        HAS_VULKAN=true
        VK_VER=$(vulkaninfo 2>/dev/null | grep "Vulkan Instance Version" | head -1 | sed 's/.*: //')
        step "Vulkan: $VK_VER"
    fi
elif [ -f /usr/lib/x86_64-linux-gnu/libvulkan.so.1 ] || \
     [ -f /usr/lib/libvulkan.so.1 ] || \
     [ -f /usr/local/lib/libvulkan.dylib ]; then
    HAS_VULKAN=true
    step "Vulkan: runtime library found"
fi
if ! $HAS_VULKAN; then
    info "Vulkan not detected (install vulkan-tools to enable)"
fi

# =============================================================================
# STEP 2: ENVIRONMENT SETUP
# =============================================================================
echo ""
echo "${BOLD}  --- Step 2: Environment Setup ---${RESET}"
echo ""

# Linux: check build-essential
if [ "$OS" = "Linux" ]; then
    if ! command -v cc &>/dev/null; then
        warn "C compiler not found. Install: sudo apt install build-essential  (Debian/Ubuntu)"
        warn "                            or: sudo dnf groupinstall 'Development Tools'  (Fedora)"
    else
        CC_VERSION=$(cc --version 2>&1 | head -1)
        step "C toolchain: $CC_VERSION"
    fi
fi

# macOS: check Xcode CLI
if [ "$OS" = "Darwin" ]; then
    if ! xcode-select -p &>/dev/null; then
        warn "Xcode CLI Tools not found. Run: xcode-select --install"
    else
        step "Xcode CLI: $(xcode-select -p)"
    fi
fi

# Rust toolchain
if ! command -v cargo &>/dev/null; then
    die "cargo not found. Install Rust: https://rustup.rs"
fi
RUST_VERSION=$(rustc --version)
step "Rust: $RUST_VERSION"

# CUDA environment
if $HAS_CUDA; then
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    if [ -d "$CUDA_HOME" ]; then
        export CUDA_HOME
        step "CUDA_HOME: $CUDA_HOME"
    fi
fi

# =============================================================================
# STEP 3: FEATURE SELECTION
# =============================================================================
echo ""
echo "${BOLD}  --- Step 3: Feature Selection ---${RESET}"
echo ""

FEATURES=()

if [ -n "$EXPLICIT_FEATURES" ]; then
    # Explicit --features flag: split on comma, use as-is
    IFS=',' read -ra EXPLICIT_ARR <<< "$EXPLICIT_FEATURES"
    for f in "${EXPLICIT_ARR[@]}"; do
        FEATURES+=("$f")
    done
    info "Explicit features: ${FEATURES[*]}"

elif $SKIP_PROMPT; then
    # Auto-select everything available, but guard flash-attn on WSL low-memory systems.
    # candle-flash-attn spawns ~40 large cicc CUDA kernel compilations (~2.3 GB each).
    # On WSL2 with ≤16 GB RAM this OOM-kills the compiler mid-build.
    FLASH_ATTN_OK=true
    if grep -qi microsoft /proc/version 2>/dev/null && [ -f /proc/meminfo ]; then
        WSL_MEM_GB=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024 ))
        if [ "$WSL_MEM_GB" -lt 16 ]; then
            FLASH_ATTN_OK=false
            warn "WSL RAM is ${WSL_MEM_GB} GB (< 16 GB): skipping flash-attn to prevent OOM crashes."
            warn "To enable flash-attn: set WSL memory to 24GB+ in %USERPROFILE%\\.wslconfig, then wsl --shutdown."
        fi
    fi
    if $HAS_CUDA; then
        FEATURES+=("cuda")
        if $FLASH_ATTN_OK; then
            FEATURES+=("flash-attn")
        fi
    fi
    $HAS_METAL  && FEATURES+=("metal")
    $HAS_ROCM   && FEATURES+=("rocm")
    $HAS_VULKAN && FEATURES+=("vulkan")
    info "SkipPrompt: auto-selected: ${FEATURES[*]:-none (CPU)}"

else
    echo "  Available features:"
    echo ""

    if $HAS_CUDA; then
        echo "${GREEN}    [1] cuda         - NVIDIA GPU acceleration (CUDA $CUDA_VERSION)${RESET}"
        if [[ $CUDA_VERSION == 13.* ]]; then
            echo "${CYAN}                         -> ⚡ CUDA 13.3+ Optimizations Active (CompileIQ, TileProg)${RESET}"
        fi
        echo "${GREEN}    [2] flash-attn   - Flash Attention 2 (requires cuda)${RESET}"
    else
        echo "${BOLD}    [1] cuda         - NVIDIA GPU (not available — CUDA not detected)${RESET}"
        echo "${BOLD}    [2] flash-attn   - Flash Attention 2 (requires cuda)${RESET}"
    fi

    if $HAS_METAL; then
        echo "${GREEN}    [3] metal        - Apple Metal GPU (Apple Silicon)${RESET}"
    else
        echo "${BOLD}    [3] metal        - Apple Metal (not available on this system)${RESET}"
    fi

    if $HAS_ROCM; then
        echo "${GREEN}    [4] rocm         - AMD GPU via ROCm/HIP${RESET}"
    else
        echo "${BOLD}    [4] rocm         - AMD ROCm (not detected)${RESET}"
    fi

    if $HAS_VULKAN; then
        echo "${GREEN}    [5] vulkan       - Vulkan 1.2 GPU compute (STRIX VulkanHal)${RESET}"
    else
        echo "${BOLD}    [5] vulkan       - Vulkan (not detected — install vulkan-tools)${RESET}"
    fi

    echo "${GREEN}    [6] python       - PyO3 Python bindings${RESET}"
    echo "${GREEN}    [7] arb-heap     - O(log n) priority queue for ARB scheduler (W>512)${RESET}"
    echo "${GREEN}    [8] arb-lockfree - Lock-free enqueue via crossbeam (high-freq HTTP)${RESET}"
    echo "${YELLOW}    [0] (none)       - CPU-only build${RESET}"
    echo ""

    # Default suggestion
    DEFAULT=""
    $HAS_CUDA   && DEFAULT="1,2"
    $HAS_METAL  && DEFAULT="${DEFAULT:+$DEFAULT,}3"
    $HAS_ROCM   && DEFAULT="${DEFAULT:+$DEFAULT,}4"
    $HAS_VULKAN && DEFAULT="${DEFAULT:+$DEFAULT,}5"
    DEFAULT="${DEFAULT:-0}"

    read -rp "  Select features (comma-separated, e.g. 1,2) [default: $DEFAULT]: " CHOICE
    CHOICE="${CHOICE:-$DEFAULT}"

    IFS=',' read -ra SELECTIONS <<< "$CHOICE"
    for sel in "${SELECTIONS[@]}"; do
        sel="${sel// /}"
        case "$sel" in
            1)
                if $HAS_CUDA; then FEATURES+=("cuda")
                else warn "Skipping cuda — CUDA Toolkit not detected"; fi ;;
            2)
                if $HAS_CUDA; then FEATURES+=("flash-attn")
                else warn "Skipping flash-attn — requires CUDA"; fi ;;
            3)
                if $HAS_METAL; then FEATURES+=("metal")
                else warn "Skipping metal — not available on this system"; fi ;;
            4)
                if $HAS_ROCM; then FEATURES+=("rocm")
                else warn "Skipping rocm — ROCm not detected"; fi ;;
            5)
                if $HAS_VULKAN; then FEATURES+=("vulkan")
                else warn "Skipping vulkan — runtime not detected"; fi ;;
            6) FEATURES+=("python") ;;
            7) FEATURES+=("arb-heap") ;;
            8) FEATURES+=("arb-lockfree") ;;
            0) ;;
            *) warn "Unknown selection: $sel (ignored)" ;;
        esac
    done
fi

# Build profile
if $DEBUG_BUILD; then
    PROFILE_FLAG=""
    PROFILE_NAME="debug"
else
    PROFILE_FLAG="--release"
    PROFILE_NAME="release"
fi

# Feature argument
FEATURE_ARG=""
if [ ${#FEATURES[@]} -gt 0 ]; then
    FEATURE_STR=$(IFS=','; echo "${FEATURES[*]}")
    FEATURE_ARG="--features $FEATURE_STR"
fi

# Fail-Fast Multi-Vendor Guard
HAS_FEATURE_CUDA=false
HAS_FEATURE_ROCM=false
for f in "${FEATURES[@]}"; do
    if [[ "$f" == "cuda" ]]; then
        HAS_FEATURE_CUDA=true
    fi
    if [[ "$f" == "rocm" ]]; then
        HAS_FEATURE_ROCM=true
    fi
done

if $HAS_FEATURE_CUDA && $HAS_FEATURE_ROCM; then
    info "Multi-vendor build requested (CUDA + ROCm); verifying compiler toolchains..."
    if ! command -v nvcc &>/dev/null; then
        err "Multi-vendor build failure: 'nvcc' not found in PATH."
        err "For hybrid NVIDIA + AMD compilation, please install CUDA Toolkit and add nvcc to PATH."
        exit 1
    fi
    if ! command -v hipcc &>/dev/null; then
        err "Multi-vendor build failure: 'hipcc' not found in PATH."
        err "For hybrid NVIDIA + AMD compilation, please install ROCm/HIP Toolkit and add hipcc to PATH."
        exit 1
    fi
    ok "Both nvcc and hipcc compilers found."
fi

# =============================================================================
# STEP 5: BUILD
# =============================================================================
echo ""
echo "${BOLD}  --- Step 5: Building Air.rs ($PROFILE_NAME) ---${RESET}"
echo ""

# ── CUDA 13.3+ Self-Healing Logic ─────────────────────────────────────────────
if [[ $CUDA_VERSION == 13.* ]]; then
    # cudarc v0.19.7 does not yet support '13030' (CUDA 13.3) explicitly.
    # We inject CUDARC_CUDA_VERSION=13000 to use stable CUDA 13 bindings.
    export CUDARC_CUDA_VERSION="13000"
    info "Detected CUDA 13.3+; injecting CUDARC_CUDA_VERSION=13000 for compatibility"
fi

if [[ -n "${FEATURE_ARG}" && "${FEATURE_ARG}" == *"cuda"* ]]; then
    info "Refreshing cudarc lock entry..."
    cargo update cudarc 2>/dev/null || true
fi

# ── WSL Memory Exhaustion Protection ──────────────────────────────────────────
JOBS_ARG=""
if grep -qi microsoft /proc/version 2>/dev/null; then
    if [ -f /proc/meminfo ]; then
        MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        MEM_GB=$((MEM_KB / 1024 / 1024))
        CPU_CORES=$(nproc 2>/dev/null || echo 1)
        
        # Cap jobs using memory-safe formula (allocate ~4GB per job)
        CALCULATED_SAFE_JOBS=$((MEM_GB / 4))
        if [ "$CALCULATED_SAFE_JOBS" -lt 1 ]; then
            CALCULATED_SAFE_JOBS=1
        elif [ "$CALCULATED_SAFE_JOBS" -gt 4 ]; then
            # Cap at 4 to be conservative on memory pressure configurations
            CALCULATED_SAFE_JOBS=4
        fi
        
        if [ "$CALCULATED_SAFE_JOBS" -lt "$CPU_CORES" ]; then
            JOBS_ARG="-j $CALCULATED_SAFE_JOBS"
            warn "WSL Environment detected: Cap parallel jobs to $CALCULATED_SAFE_JOBS (out of $CPU_CORES CPUs) to prevent OOM system crashes."
        fi
    fi
fi

# Propagate arch to child processes if detected
if [[ -n "${NVCC_ARCH:-}" ]]; then
    info "GPU arch targeting: ${NVCC_ARCH} (injected into all CUDA kernel builds)"
fi

CMD="cargo build $PROFILE_FLAG $FEATURE_ARG $JOBS_ARG"
info "Running: $CMD"
echo ""

# ── Architectural Summary ──────────────────────────────────────────────
echo ""
echo "${BOLD}  --- Air.rs Consolidated Stack ---${RESET}"
echo ""
echo "${GREEN}  [✓] Actor-Based Threading              ${RESET}(RequestOrchestrator, scheduler.rs)"
echo "${GREEN}  [✓] S.L.I.P. Lazy Weight Streaming     ${RESET}(LayerUnit, layer_pipeline.rs)"
echo "${GREEN}  [✓] Flash-Attn 2 + cuBLAS DeltaNet     ${RESET}(fused kernels, ops.rs)"
echo "${GREEN}  [✓] Parallel Prefix-Scan (Rayon)       ${RESET}(recurrence, gated_deltanet.rs)"
echo "${GREEN}  [✓] STRIX Vulkan Buffer Pooling        ${RESET}(Managed pool, vulkan_hal.rs)"
echo "${GREEN}  [✓] Evaluation Gates (CI Guard)        ${RESET}(HellaSwag/MMLU, eval.rs)"
echo "${GREEN}  [✓] Whisper Production Pipeline        ${RESET}(Beam Search, whisper.rs)"
echo "${GREEN}  [✓] Self-Healing CUDA 13 Logic         ${RESET}(Transparent bindings, v1.1.4)"
echo "${GREEN}  [✓] GPU ISA Targeting (sm_XX)           ${RESET}(Arch-optimised kernels, v1.1.5)"
echo ""

BUILD_START=$(date +%s)
if eval "$CMD"; then
    BUILD_END=$(date +%s)
    BUILD_TIME=$((BUILD_END - BUILD_START))

    echo ""
    echo "${GREEN}  ======================================================${RESET}"
    echo "${GREEN}       BUILD SUCCEEDED                                   ${RESET}"
    echo "${GREEN}  ======================================================${RESET}"
    echo ""
    echo "${BOLD}  Profile:  ${RESET}$PROFILE_NAME"
    echo "${BOLD}  Features: ${RESET}${FEATURES[*]:-none (CPU only)}"
    printf "${BOLD}  Time:     ${RESET}%ds\n" "$BUILD_TIME"

    # Binary size
    BINARY="./target/$PROFILE_NAME/air-rs"
    if [ -f "$BINARY" ]; then
        if [ "$OS" = "Darwin" ]; then
            BINARY_SIZE=$(stat -f%z "$BINARY" 2>/dev/null || echo 0)
        else
            BINARY_SIZE=$(stat -c%s "$BINARY" 2>/dev/null || echo 0)
        fi
        BINARY_MB=$(echo "scale=1; $BINARY_SIZE / 1048576" | bc 2>/dev/null || echo "?")
        echo "${BOLD}  Binary:   ${RESET}$BINARY ($BINARY_MB MB)"
    fi
    echo ""
else
    echo ""
    echo "${RED}  ======================================================${RESET}"
    echo "${RED}       BUILD FAILED                                      ${RESET}"
    echo "${RED}  ======================================================${RESET}"
    echo ""
    echo "${YELLOW}  Common fixes:${RESET}"
    echo "${YELLOW}    - Missing build-essential:  sudo apt install build-essential${RESET}"
    echo "${YELLOW}    - CUDA not in PATH:          export CUDA_HOME=/usr/local/cuda${RESET}"
    echo "${YELLOW}    - Metal unavailable:         Build on macOS Apple Silicon${RESET}"
    echo "${YELLOW}    - Out of memory:             Use --debug or close other programs${RESET}"
    echo "${YELLOW}    - Linker errors:             cargo clean, then re-run this script${RESET}"
    echo ""
    exit 1
fi
