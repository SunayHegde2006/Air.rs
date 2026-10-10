#!/usr/bin/env bash
# Air.rs v2.0.0 — Unified Uninstaller
set -euo pipefail

INSTALL_DIR="${AIR_INSTALL_DIR:-$HOME/.local/bin}"

if [ -t 1 ]; then
    GREEN='\033[0;32m'; NC='\033[0m'
else
    GREEN=''; NC=''
fi

echo -e "${GREEN}[air]${NC} Uninstalling Air.rs..."
rm -f "${INSTALL_DIR}/air-rs" "${INSTALL_DIR}/air" "$HOME/.local/bin/air-rs" "$HOME/.local/bin/air"
rm -rf "${HOME}/.air-rs"

echo -e "${GREEN}[air]${NC} ✓ Air.rs uninstalled successfully."
