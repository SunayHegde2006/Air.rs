set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    just --list

build:
    cargo build --release

dev *ARGS:
    cargo watch -x "run -- {{ARGS}}"

test:
    cargo nextest run --workspace

test-gpu:
    cargo nextest run --workspace --features gpu-integration

bench:
    cargo bench --workspace

lint:
    cargo clippy --all-targets --all-features -- -D warnings
    cargo fmt --all -- --check

fmt:
    cargo fmt --all

kernels:
    python scripts/build_kernels.py

clean:
    cargo clean
    rm -rf kernels/precompiled/*

install-local:
    cargo build --release
    cp target/release/air-rs ~/.local/bin/air-rs

check:
    cargo deny check
    cargo audit
