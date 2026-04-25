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
    RED=$(tput setaf 1);   GREEN=$(tput setaf 2);  YELLOW=$(tput setaf 3)
    BLUE=$(tput setaf 4);  CYAN=$(tput setaf 6);   BOLD=$(tput bold)
    RESET=$(tput sgr0);    MAGENTA=$(tput setaf 5)
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
echo "${MAGENTA}       Air.rs Build System (macOS / Linux)              ${RESET}"
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

# ── CUDA Toolkit ──────────────────────────────────────────────────────────────
HAS_CUDA=false
CUDA_VERSION=""
if command -v nvcc &>/dev/null; then
    if NVCC_OUT=$(nvcc --version 2>&1 | grep "release"); then
        HAS_CUDA=true
        CUDA_VERSION=$(echo "$NVCC_OUT" | sed 's/.*release //' | sed 's/,.*//')
        step "CUDA Toolkit: $CUDA_VERSION"
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
    # Auto-select everything available
    $HAS_CUDA   && { FEATURES+=("cuda"); FEATURES+=("flash-attn"); }
    $HAS_METAL  && FEATURES+=("metal")
    $HAS_ROCM   && FEATURES+=("rocm")
    $HAS_VULKAN && FEATURES+=("vulkan")
    info "SkipPrompt: auto-selected: ${FEATURES[*]:-none (CPU)}"

else
    echo "  Available features:"
    echo ""

    if $HAS_CUDA; then
        echo "${GREEN}    [1] cuda         - NVIDIA GPU acceleration (CUDA $CUDA_VERSION)${RESET}"
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

# =============================================================================
# STEP 4: PRE-BUILD CLEANUP
# =============================================================================
echo ""
echo "${BOLD}  --- Step 4: Pre-Build ---${RESET}"
echo ""

# Remove stale small stdc++.lib stubs (Linux cross-build artifacts)
if find ./target -name "stdc++.lib" -size -72c 2>/dev/null | grep -q .; then
    find ./target -name "stdc++.lib" -size -72c -print -delete 2>/dev/null
    info "Removed stale stdc++.lib stub(s)"
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

# =============================================================================
# STEP 5: BUILD
# =============================================================================
echo ""
echo "${BOLD}  --- Step 5: Building Air.rs ($PROFILE_NAME) ---${RESET}"
echo ""

CMD="cargo build $PROFILE_FLAG $FEATURE_ARG"
info "Running: $CMD"
echo ""

# ── OCS Algorithm Summary (always compiled-in) ────────────────────────────
echo ""
echo "${BOLD}  --- Optimal Compounding Stack (always enabled) ---${RESET}"
echo ""
echo "${GREEN}  [✓] SageAttention3 FP4 microscaling    ${RESET}(fp4_attention, ops.rs)"
echo "${GREEN}  [✓] KIMI Linear Attention O(N·D²)       ${RESET}(linear_attention_kimi, ops.rs)"
echo "${GREEN}  [✓] Gated Attention sink-suppression    ${RESET}(gated_attention, ops.rs)"
echo "${GREEN}  [✓] QJL 1-bit JL-transform KV keys      ${RESET}(QjlKey, kv_compress.rs)"
echo "${GREEN}  [✓] Fast KV Compaction (cosine merge)   ${RESET}(compact_kv_by_similarity, kv_compress.rs)"
echo "${GREEN}  [✓] HERMES importance-scored eviction   ${RESET}(HermesTierManager, kv_tier.rs)"
echo "${GREEN}  [✓] ConceptMoE adaptive token routing   ${RESET}(concept_moe_forward, moe.rs)"
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
