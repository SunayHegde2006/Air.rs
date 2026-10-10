#!/usr/bin/env python3
"""
Air.rs Kernel Build Script
Compiles CUDA kernels for RTX 3060 (sm_86) using nvcc & fatbinary.
Generates libair_kernels.so for Rust FFI.
Bundles WGSL shaders for WebGPU fallback.
"""

import subprocess
import sys
import os
from pathlib import Path

KERNEL_SRC = Path("kernels/src")
OUTPUT_DIR = Path("kernels/precompiled")

def run(cmd, desc):
    print(f"  [{desc}] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        sys.exit(1)
    return result.stdout

def build_cuda_kernels():
    """Compile CUDA kernels for sm_86 (RTX 3060)."""
    cuda_files = list(KERNEL_SRC.glob("*.cu"))
    if not cuda_files:
        print("  No CUDA kernels found.")
        return
    
    for cu in cuda_files:
        stem = cu.stem
        
        # Compile PTX
        ptx = OUTPUT_DIR / f"{stem}_sm86.ptx"
        run(["nvcc", "-arch=sm_86", "-ptx", str(cu), "-o", str(ptx),
             "-O3", "--use_fast_math"], f"PTX {stem}")
        
        # Compile CUBIN
        cubin = OUTPUT_DIR / f"{stem}_sm86.cubin"
        run(["nvcc", "-arch=sm_86", "-cubin", str(cu), "-o", str(cubin),
             "-O3", "--use_fast_math"], f"CUBIN {stem}")
        
        print(f"  ✓ {stem} compiled for sm_86")
    
    # Create FATBIN from all CUBINs
    cubins = list(OUTPUT_DIR.glob("*_sm86.cubin"))
    if cubins:
        fatbin = OUTPUT_DIR / "air_kernels.fatbin"
        cmd = ["fatbinary", "--create", str(fatbin)]
        for c in cubins:
            cmd.append(f"--image=profile=sm_86,file={c}")
        run(cmd, "FATBIN")
        print(f"  ✓ FATBIN created: {fatbin}")
    
    # Create libair_kernels.so shared library
    so = OUTPUT_DIR / "libair_kernels.so"
    cmd_so = ["nvcc", "-arch=sm_86", "--shared", "-Xcompiler", "-fPIC", "-o", str(so), "-O3", "--use_fast_math"] + [str(f) for f in cuda_files]
    run(cmd_so, "SHARED LIB")
    print(f"  ✓ Shared library created: {so}")

def build_wgsl_kernels():
    """Copy WGSL shaders for WebGPU backend."""
    wgsl_files = list(KERNEL_SRC.glob("*.wgsl"))
    import shutil
    for wgsl in wgsl_files:
        dest = OUTPUT_DIR / wgsl.name
        shutil.copy2(wgsl, dest)
        print(f"  ✓ {wgsl.name} copied for wgpu")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Building kernels for Air.rs v2.0.0...")
    print(f"  Source: {KERNEL_SRC.resolve()}")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print()
    
    print("CUDA kernels (sm_86 / RTX 3060):")
    build_cuda_kernels()
    print()
    
    print("WGSL kernels (wgpu fallback):")
    build_wgsl_kernels()
    print()
    
    print("Done. All kernels compiled.")

if __name__ == "__main__":
    main()
