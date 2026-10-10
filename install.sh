#!/usr/bin/env bash
# Air.rs v2.0.0 — Unified OS & Hardware Agnostic One-Line Installer / Uninstaller
set -euo pipefail

REPO="SunayHegde2006/Air.rs"
DEFAULT_INSTALL_DIR="$HOME/.local/bin"
INSTALL_DIR="${AIR_INSTALL_DIR:-$DEFAULT_INSTALL_DIR}"
VERSION="${AIR_VERSION:-latest}"

if [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; NC=''
fi

info()  { echo -e "${GREEN}[air]${NC} $*"; }
warn()  { echo -e "${YELLOW}[air]${NC} $*"; }
error() { echo -e "${RED}[air-error]${NC} $*" >&2; exit 1; }

for arg in "$@"; do
    if [ "$arg" = "--uninstall" ] || [ "$arg" = "-u" ] || [ "$arg" = "uninstall" ]; then
        info "Uninstalling Air.rs..."
        rm -f "${INSTALL_DIR}/air-rs" "${INSTALL_DIR}/air" "$HOME/.local/bin/air-rs" "$HOME/.local/bin/air"
        rm -rf "${HOME}/.air-rs"
        info "✓ Air.rs successfully uninstalled from ${INSTALL_DIR} and ${HOME}/.air-rs"
        exit 0
    fi
done

detect_os() {
    local os
    os="$(uname -s 2>/dev/null || echo "unknown")"
    case "$os" in
        Linux*)             echo "linux" ;;
        Darwin*)            echo "macos" ;;
        CYGWIN*|MINGW*|MSYS*) echo "windows" ;;
        FreeBSD*|OpenBSD*|NetBSD*) echo "bsd" ;;
        *)                  echo "unknown" ;;
    esac
}

detect_arch() {
    local arch
    arch="$(uname -m 2>/dev/null || uname -p 2>/dev/null || echo "unknown")"
    case "$arch" in
        x86_64|amd64)         echo "x86_64" ;;
        aarch64|arm64|armv8*) echo "aarch64" ;;
        armv7*|arm)           echo "armv7" ;;
        riscv64)              echo "riscv64" ;;
        *)                    echo "$arch" ;;
    esac
}

OS="$(detect_os)"
ARCH="$(detect_arch)"

info "Air.rs Installer — Detecting environment..."
info "  OS: ${OS} ($(uname -s 2>/dev/null || echo 'unknown'))"
info "  Hardware Arch: ${ARCH}"

if command -v nvidia-smi &>/dev/null; then
    info "  Hardware Acceleration: NVIDIA CUDA GPU detected"
elif [ "$OS" = "macos" ]; then
    info "  Hardware Acceleration: Apple Silicon Metal / Accelerate detected"
elif command -v vulkaninfo &>/dev/null; then
    info "  Hardware Acceleration: Vulkan API detected"
else
    info "  Hardware Acceleration: CPU SIMD (AVX-512 / AVX2 / NEON) fallback"
fi

BINARY="air-rs-${OS}-${ARCH}"
if [ "$OS" = "windows" ]; then
    BINARY="${BINARY}.exe"
fi

INSTALLED=0

if command -v curl &>/dev/null || command -v wget &>/dev/null; then
    if [ "$VERSION" = "latest" ]; then
        RESOLVED_VERSION=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null \
            | grep '"tag_name"' | sed 's/.*"v\(.*\)".*/\1/' || echo "2.0.0")
    else
        RESOLVED_VERSION="${VERSION}"
    fi

    info "Fetching Air.rs v${RESOLVED_VERSION}..."
    DOWNLOAD_URL="https://github.com/${REPO}/releases/download/v${RESOLVED_VERSION}/${BINARY}.tar.gz"
    TEMP_DIR=$(mktemp -d 2>/dev/null || mktemp -d -t 'air_install')
    
    clean_temp() { rm -rf "$TEMP_DIR"; }
    trap clean_temp EXIT

    if curl -fsSL "$DOWNLOAD_URL" -o "${TEMP_DIR}/air-rs.tar.gz" 2>/dev/null; then
        mkdir -p "$INSTALL_DIR"
        tar -xzf "${TEMP_DIR}/air-rs.tar.gz" -C "$TEMP_DIR" 2>/dev/null || unzip -q "${TEMP_DIR}/air-rs.tar.gz" -d "$TEMP_DIR" 2>/dev/null || true
        if [ -f "${TEMP_DIR}/air-rs" ] || [ -f "${TEMP_DIR}/air-rs.exe" ]; then
            cp "${TEMP_DIR}/air-rs"* "${INSTALL_DIR}/"
            chmod +x "${INSTALL_DIR}/air-rs"* 2>/dev/null || true
            INSTALLED=1
            info "✓ Successfully installed pre-built binary to ${INSTALL_DIR}"
        fi
    fi
fi

if [ "$INSTALLED" -eq 0 ]; then
    if command -v cargo &>/dev/null; then
        info "Pre-built binary unavailable for ${OS}/${ARCH}. Compiling from source using Cargo..."
        if [ -f "Cargo.toml" ]; then
            cargo build --release --workspace
            mkdir -p "$INSTALL_DIR"
            cp target/release/air-cli "${INSTALL_DIR}/air-rs" 2>/dev/null || cp target/release/air-rs "${INSTALL_DIR}/air-rs" 2>/dev/null || true
            chmod +x "${INSTALL_DIR}/air-rs" 2>/dev/null || true
            INSTALLED=1
            info "✓ Successfully built from source with Cargo!"
        else
            info "Installing via cargo install from repository..."
            cargo install --git "https://github.com/${REPO}.git" air-cli --root "$HOME/.local" && INSTALLED=1 || true
        fi
    fi
fi

if [ "$INSTALLED" -eq 0 ]; then
    if [ -f "target/release/air-cli" ] || [ -f "target/release/air-rs" ]; then
        mkdir -p "$INSTALL_DIR"
        cp target/release/air-cli "${INSTALL_DIR}/air-rs" 2>/dev/null || cp target/release/air-rs "${INSTALL_DIR}/air-rs" 2>/dev/null || true
        chmod +x "${INSTALL_DIR}/air-rs" 2>/dev/null || true
        INSTALLED=1
        info "✓ Copied local release build to ${INSTALL_DIR}"
    fi
fi

if [ "$INSTALLED" -eq 0 ]; then
    warn "Could not download pre-built binary or compile via Cargo automatically."
    warn "To build Air.rs on your system, please install Rust:"
    warn "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    warn "Then run: cargo build --release"
    exit 1
fi

if ! echo "$PATH" | grep -q "$INSTALL_DIR"; then
    SHELL_NAME="$(basename "${SHELL:-bash}")"
    SHELL_RC=""
    case "$SHELL_NAME" in
        zsh)  SHELL_RC="$HOME/.zshrc" ;;
        fish) SHELL_RC="$HOME/.config/fish/config.fish" ;;
        bash) SHELL_RC="$HOME/.bashrc" ;;
        *)    SHELL_RC="$HOME/.profile" ;;
    esac
    if [ -n "$SHELL_RC" ] && [ -d "$(dirname "$SHELL_RC")" ]; then
        if [ "$SHELL_NAME" = "fish" ]; then
            echo "fish_add_path $INSTALL_DIR" >> "$SHELL_RC"
        else
            echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$SHELL_RC"
        fi
        info "Added ${INSTALL_DIR} to ${SHELL_RC}"
        warn "Run 'source ${SHELL_RC}' or open a new terminal session to update PATH."
    fi
fi

info "✓ Air.rs installation complete!"
info "  Binary location: ${INSTALL_DIR}/air-rs"
info "  Verify with: air-rs --help"
