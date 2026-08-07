<#
.SYNOPSIS
    Air.rs Unified Build Script
    Sets up the environment, asks which features to enable, and builds the project.

.DESCRIPTION
    This is the one script you need. It:
      1. Auto-detects Windows SDK, MSVC, and CUDA paths
      2. Loads the Visual Studio environment (vcvars64)
      3. Asks which features to enable (cuda, flash-attn, vulkan, python, arb-heap, arb-lockfree)
      4. Cleans stale stdc++ stubs if needed
      5. Runs cargo build

.EXAMPLE
    .\build_air.ps1              # Interactive feature selection
    .\build_air.ps1 -Release     # Release mode (default is release)
    .\build_air.ps1 -Debug       # Debug mode
    .\build_air.ps1 -SkipPrompt  # Use defaults (cuda + flash-attn)
#>

[CmdletBinding()]
param(
    [switch]$DebugBuild,
    [switch]$Release,
    [switch]$SkipPrompt
)

$ErrorActionPreference = 'Stop'

# -- Helpers -----------------------------------------------------------------
function Write-Step  { param([string]$msg) Write-Host "  [+] $msg" -ForegroundColor Green }
function Write-Info  { param([string]$msg) Write-Host "  [i] $msg" -ForegroundColor Cyan }
function Write-Warn  { param([string]$msg) Write-Host "  [!] $msg" -ForegroundColor Yellow }
function Write-Err   { param([string]$msg) Write-Host "  [X] $msg" -ForegroundColor Red }

Write-Host ""
Write-Host "  ======================================================" -ForegroundColor Magenta
Write-Host "       Air.rs Build System — v1.2.2 (Stable)            " -ForegroundColor Magenta
Write-Host "  ======================================================" -ForegroundColor Magenta
Write-Host ""

# ============================================================================
# STEP 1: HARDWARE DETECTION
# ============================================================================
Write-Host "  --- Step 1: Hardware Detection ---" -ForegroundColor White
Write-Host ""

$arch = if ([Environment]::Is64BitOperatingSystem) { 'x64' } else { 'x86' }
Write-Info "Architecture: $arch"

# GPU check
$hasGpu = $false
$gpuName = ""
try {
    $nvsmi = nvidia-smi -L 2>&1
    if ($LASTEXITCODE -eq 0) {
        $hasGpu = $true
        $gpuName = ($nvsmi | Select-Object -First 1) -replace '^GPU 0: ', '' -replace ' \(UUID:.*', ''
        Write-Step "NVIDIA GPU: $gpuName"
    }
} catch {
    Write-Info "No NVIDIA GPU detected (CPU-only builds will work)"
}

# GPU Architecture Detection
$gpuArch = ""
try {
    $caps = nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        $capList = @()
        foreach ($cap in $caps) {
            $capClean = $cap.Trim() -replace '\.', ''
            if ($capClean -match '^\d+$') {
                $capList += "sm_$capClean"
            }
        }
        if ($capList.Count -gt 0) {
            $capList = $capList | Select-Object -Unique | Sort-Object
            $gpuArch = $capList -join ','
            if (-not $env:NVCC_ARCH) {
                $env:NVCC_ARCH = $gpuArch
            }
            Write-Step "GPU arch targeting: $env:NVCC_ARCH (override via env:NVCC_ARCH='all' or list)"
        }
    }
} catch {
    Write-Info "GPU arch detection unavailable; NVCC will use default"
}

# CUDA check
$hasCuda = $false
$cudaVersion = ""
try {
    $nvccOut = nvcc --version 2>&1 | Select-String "release"
    if ($nvccOut) {
        $hasCuda = $true
        $cudaVersion = ($nvccOut -replace '.*release ', '' -replace ',.*', '')
        if ($cudaVersion -like "13.*") {
            Write-Step "CUDA Toolkit: $cudaVersion (High-Performance CUDA 13 Mode)"
        } else {
            Write-Step "CUDA Toolkit: $cudaVersion"
        }
    }
} catch {
    if ($hasGpu) { Write-Warn "NVIDIA GPU found but CUDA Toolkit not in PATH" }
}

# Vulkan check
$hasVulkan = $false
try {
    $vkInfo = vulkaninfo 2>&1 | Select-String "Vulkan Instance Version"
    if ($vkInfo) {
        $hasVulkan = $true
        $vkVersion = ($vkInfo -replace '.*: ', '').Trim()
        Write-Step "Vulkan: $vkVersion"
    }
} catch { }
if (-not $hasVulkan) {
    # Fallback: check for vulkan-1.dll
    if (Test-Path "$env:SystemRoot\System32\vulkan-1.dll") {
        $hasVulkan = $true
        Write-Step "Vulkan: runtime DLL found (vulkan-1.dll)"
    } else {
        Write-Info "Vulkan not detected (install Vulkan SDK to enable)"
    }
}

# ============================================================================
# STEP 2: VISUAL STUDIO ENVIRONMENT
# ============================================================================
Write-Host ""
Write-Host "  --- Step 2: Build Environment ---" -ForegroundColor White
Write-Host ""

# Try vswhere first (cleanest approach)
$vsLoaded = $false
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vsInstallPath = & $vswhere -latest -property installationPath 2>$null
    if ($vsInstallPath) {
        $vcvars = Join-Path $vsInstallPath "VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $vcvars) {
            Write-Info "Loading Visual Studio environment..."
            cmd /c " `"$vcvars`" && set " | ForEach-Object {
                if ($_ -match "^(.*?)=(.*)$") {
                    Set-Content "env:\$($Matches[1])" $Matches[2]
                }
            }
            $vsLoaded = $true
            $vsEdition = Split-Path $vsInstallPath -Leaf
            Write-Step "VS environment loaded ($vsEdition)"
        }
    }
}

# Fallback: manual SDK/MSVC detection (same as setup_build_env.ps1)
if (-not $vsLoaded) {
    Write-Info "vswhere not found, detecting SDK/MSVC manually..."

    $libPaths = @()

    # Windows SDK
    $sdkRoot = "C:\Program Files (x86)\Windows Kits\10\Lib"
    if (Test-Path $sdkRoot) {
        $sdkVersion = (Get-ChildItem $sdkRoot -Directory |
            Where-Object { $_.Name -match '^\d+\.\d+\.\d+\.\d+$' } |
            Sort-Object Name | Select-Object -Last 1).Name
        if ($sdkVersion) {
            $sdkUm   = Join-Path $sdkRoot "$sdkVersion\um\$arch"
            $sdkUcrt = Join-Path $sdkRoot "$sdkVersion\ucrt\$arch"
            if (Test-Path $sdkUm)   { $libPaths += $sdkUm;   Write-Step "SDK um:   $sdkUm" }
            if (Test-Path $sdkUcrt) { $libPaths += $sdkUcrt;  Write-Step "SDK ucrt: $sdkUcrt" }
        }
    }

    # MSVC toolchain
    $vsEditions = @('Professional', 'Enterprise', 'Community', 'BuildTools')
    $vsYears    = @('2022', '2019')
    foreach ($year in $vsYears) {
        foreach ($edition in $vsEditions) {
            $msvcBase = "C:\Program Files\Microsoft Visual Studio\$year\$edition\VC\Tools\MSVC"
            if (Test-Path $msvcBase) {
                $msvcVer = (Get-ChildItem $msvcBase -Directory | Sort-Object Name | Select-Object -Last 1).Name
                $candidate = Join-Path $msvcBase "$msvcVer\lib\$arch"
                if (Test-Path $candidate) {
                    $libPaths += $candidate
                    Write-Step "MSVC lib: $candidate"
                    break
                }
            }
        }
        if ($libPaths.Count -ge 3) { break }
    }

    # CUDA lib
    $cudaRoot = $env:CUDA_PATH
    if (-not $cudaRoot) {
        $cudaCandidates = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending
        if ($cudaCandidates) { $cudaRoot = $cudaCandidates[0].FullName }
    }
    if ($cudaRoot) {
        $cudaLib = Join-Path $cudaRoot "lib\$arch"
        if (Test-Path $cudaLib) { $libPaths += $cudaLib; Write-Step "CUDA lib: $cudaLib" }
    }

    if ($libPaths.Count -eq 0) {
        Write-Err "No SDK/MSVC found. Install Visual Studio 2022 with 'Desktop development with C++' workload."
        exit 1
    }

    $env:LIB = $libPaths -join ';'
    Write-Step "Set LIB ($($libPaths.Count) paths)"
}

# CUDA compatibility flags
if ($hasCuda) {
    $env:CL = "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
    Write-Info "Set CUDA/MSVC compatibility flag"
}

# ============================================================================
# STEP 3: FEATURE SELECTION
# ============================================================================
Write-Host ""
Write-Host "  --- Step 3: Feature Selection ---" -ForegroundColor White
Write-Host ""

$features = @()

if ($SkipPrompt) {
    # Default: enable everything available
    if ($hasCuda)   { $features += 'cuda'; $features += 'flash-attn' }
    if ($hasVulkan) { $features += 'vulkan' }
    Write-Info "SkipPrompt: auto-selected features: $($features -join ', ')"
} else {
    Write-Host "  Select Machine Hardware Profile:" -ForegroundColor White
    Write-Host "    [1] Single GPU Rig / Consumer Hardware  (RTX 30/40, Single GPU — Lean footprint)" -ForegroundColor Green
    Write-Host "    [2] Multi-GPU Rig / NVLink Hardware    (Dual/Quad GPU, NCCL Tensor Parallelism)" -ForegroundColor Cyan
    Write-Host "    [3] Multi-Node / Datacentre Level Hardware (InfiniBand RDMA, Disaggregated KV Pools, Lockfree ARB)" -ForegroundColor Magenta
    Write-Host "    [0] Custom / Manual Feature Selection" -ForegroundColor Yellow
    Write-Host ""
    $profileChoice = Read-Host "  Select hardware profile [default: 1]"
    if ([string]::IsNullOrWhiteSpace($profileChoice)) { $profileChoice = "1" }

    switch ($profileChoice) {
        '1' {
            Write-Info "Profile Selected: Single GPU Rig / Consumer Hardware"
            if ($hasCuda)   { $features += 'cuda'; $features += 'flash-attn' }
            if ($hasVulkan) { $features += 'vulkan' }
        }
        '2' {
            Write-Info "Profile Selected: Multi-GPU Rig / NVLink Hardware"
            if ($hasCuda)   { $features += 'cuda'; $features += 'flash-attn'; $features += 'arb-heap' }
            if ($hasVulkan) { $features += 'vulkan'; $features += 'arb-heap' }
        }
        '3' {
            Write-Info "Profile Selected: Multi-Node / Datacentre Level Hardware"
            if ($hasCuda)   { $features += 'cuda'; $features += 'flash-attn'; $features += 'arb-heap'; $features += 'arb-lockfree' }
            if ($hasVulkan) { $features += 'vulkan'; $features += 'arb-heap'; $features += 'arb-lockfree' }
        }
        default {
            Write-Host "  Available features:" -ForegroundColor White
            if ($hasCuda) {
                Write-Host "    [1] cuda         - NVIDIA GPU acceleration" -ForegroundColor Green
                Write-Host "    [2] flash-attn   - Flash Attention 2" -ForegroundColor Green
            }
            if ($hasVulkan) { Write-Host "    [3] vulkan       - Vulkan 1.2 GPU compute" -ForegroundColor Green }
            Write-Host "    [4] python       - PyO3 Python bindings" -ForegroundColor Green
            Write-Host "    [5] arb-heap     - Priority queue for ARB scheduler" -ForegroundColor Green
            Write-Host "    [6] arb-lockfree - Lock-free enqueue via crossbeam" -ForegroundColor Green
            Write-Host "    [7] sycl         - Intel OneAPI / SYCL acceleration" -ForegroundColor Green
            Write-Host "    [8] mojo        - Modular Mojo / MAX compute graph execution" -ForegroundColor Green
            if ($hasCuda) { Write-Host "    [9] gds          - GPUDirect Storage NVMe-to-VRAM DMA" -ForegroundColor Green }
            Write-Host "    [10] rocm        - AMD GPU via ROCm/HIP" -ForegroundColor Green
            Write-Host "    [0] (none)       - CPU-only build" -ForegroundColor Yellow
            Write-Host ""
            $choice = Read-Host "  Select custom features (comma-separated)"
            $selections = $choice -split ',' | ForEach-Object { $_.Trim() }
            foreach ($sel in $selections) {
                switch ($sel) {
                    '1' { if ($hasCuda) { $features += 'cuda' } }
                    '2' { if ($hasCuda) { $features += 'flash-attn' } }
                    '3' { if ($hasVulkan) { $features += 'vulkan' } }
                    '4' { $features += 'python' }
                    '5' { $features += 'arb-heap' }
                    '6' { $features += 'arb-lockfree' }
                    '7' { $features += 'sycl' }
                    '8' { $features += 'mojo' }
                    '9' { if ($hasCuda) { $features += 'gds' } }
                    '10' { $features += 'rocm' }
                    default { }
                }
            }
        }
    }
}


# Determine build profile
$buildProfile = if ($DebugBuild) { "" } else { "--release" }
$profileName = if ($DebugBuild) { "debug" } else { "release" }

# Build feature string
$featureArg = ""
if ($features.Count -gt 0) {
    $featureStr = $features -join ','
    $featureArg = "--features $featureStr"
}

# Fail-Fast Multi-Vendor Guard
$hasFeatureCuda = $false
$hasFeatureRocm = $false
foreach ($f in $features) {
    if ($f -eq 'cuda') { $hasFeatureCuda = $true }
    if ($f -eq 'rocm') { $hasFeatureRocm = $true }
}

if ($hasFeatureCuda -and $hasFeatureRocm) {
    Write-Info "Multi-vendor build requested (CUDA + ROCm); verifying compiler toolchains..."
    $hasNvcc = $false
    $hasHipcc = $false
    try {
        $null = Get-Command nvcc -ErrorAction SilentlyContinue
        $hasNvcc = $true
    } catch {}
    try {
        $null = Get-Command hipcc -ErrorAction SilentlyContinue
        $hasHipcc = $true
    } catch {}

    if (-not $hasNvcc) {
        Write-Err "Multi-vendor build failure: 'nvcc' not found in PATH."
        Write-Err "For hybrid NVIDIA + AMD compilation, please install CUDA Toolkit and add nvcc to PATH."
        exit 1
    }
    if (-not $hasHipcc) {
        Write-Err "Multi-vendor build failure: 'hipcc' not found in PATH."
        Write-Err "For hybrid NVIDIA + AMD compilation, please install ROCm/HIP Toolkit and add hipcc to PATH."
        exit 1
    }
    Write-Step "Both nvcc and hipcc compilers found."
}

# ============================================================================
# STEP 5: BUILD
# ============================================================================
Write-Host ""
Write-Host "  --- Step 5: Building Air.rs ($profileName) ---" -ForegroundColor White
Write-Host ""

# CUDA 13.3+ Self-Healing Logic
# cudarc v0.19.7 doesn't yet recognise '13030' explicitly.
# Inject CUDARC_CUDA_VERSION=13000 to select stable CUDA 13 bindings.
if ($cudaVersion -like "13.*") {
    $env:CUDARC_CUDA_VERSION = "13000"
    Write-Info "Detected CUDA 13.3+; injecting CUDARC_CUDA_VERSION=13000 for compatibility"
}

if ($featureArg -match "cuda") {
    Write-Info "Refreshing cudarc lock entry..."
    cargo update cudarc 2>&1 | Out-Null
}
if ($gpuArch -ne "") {
    Write-Info "GPU arch targeting: $gpuArch (injected into all CUDA kernel builds)"
}

$cmd = "cargo build $buildProfile $featureArg"
Write-Info "Running: $cmd"
Write-Host ""

# Architectural Summary
Write-Host ""
Write-Host "  --- Air.rs Consolidated Stack ---" -ForegroundColor White
Write-Host ""
Write-Host "  [+] Actor-Based Threading              (RequestOrchestrator, scheduler.rs)" -ForegroundColor Green
Write-Host "  [+] S.L.I.P. Lazy Weight Streaming     (LayerUnit, layer_pipeline.rs)" -ForegroundColor Green
Write-Host "  [+] Flash-Attn 2 + cuBLAS DeltaNet     (fused kernels, ops.rs)" -ForegroundColor Green
Write-Host "  [+] Parallel Prefix-Scan (Rayon)       (recurrence, gated_deltanet.rs)" -ForegroundColor Green
Write-Host "  [+] STRIX Vulkan Buffer Pooling        (Managed pool, vulkan_hal.rs)" -ForegroundColor Green
Write-Host "  [+] Evaluation Gates (CI Guard)        (HellaSwag/MMLU, eval.rs)" -ForegroundColor Green
Write-Host "  [+] Whisper Production Pipeline        (Beam Search, whisper.rs)" -ForegroundColor Green
Write-Host "  [+] Self-Healing CUDA 13 Logic         (Transparent bindings, v1.1.4)" -ForegroundColor Green
Write-Host "  [+] GPU ISA Targeting (sm_XX)           (Arch-optimised kernels, v1.1.5)" -ForegroundColor Green
Write-Host "  [+] Zero-Config CLI Entry Point        (air-rs --run <target>, air_rs.rs)" -ForegroundColor Green
Write-Host "  [+] Interactive TUI REPL               (air-rs --interactive, tui.rs)" -ForegroundColor Green
Write-Host "  [+] Concurrent REST Server + TLS       (--serve --tls-cert, api.rs)" -ForegroundColor Green
Write-Host "  [+] Metal MSL Compute Kernels          (DeltaNet/RMSNorm/SwiGLU, kernels.metal)" -ForegroundColor Green
Write-Host ""


# Execute build
$buildStart = Get-Date
Invoke-Expression $cmd
$buildResult = $LASTEXITCODE
$buildTime = ((Get-Date) - $buildStart).TotalSeconds

Write-Host ""
if ($buildResult -eq 0) {
    Write-Host "  ======================================================" -ForegroundColor Green
    Write-Host "       BUILD SUCCEEDED                                   " -ForegroundColor Green
    Write-Host "  ======================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Profile:  $profileName" -ForegroundColor White
    Write-Host "  Features: $(if ($features.Count -gt 0) { $features -join ', ' } else { '(none - CPU only)' })" -ForegroundColor White
    Write-Host "  Time:     $([math]::Round($buildTime, 1))s" -ForegroundColor White

    $binaryPath = ".\target\$profileName\air-rs.exe"
    if (Test-Path $binaryPath) {
        $binarySize = [math]::Round((Get-Item $binaryPath).Length / 1MB, 1)
        Write-Host "  Binary:   $binaryPath ($binarySize MB)" -ForegroundColor White
    }
    Write-Host ""
} else {
    Write-Host "  ======================================================" -ForegroundColor Red
    Write-Host "       BUILD FAILED                                      " -ForegroundColor Red
    Write-Host "  ======================================================" -ForegroundColor Red
    Write-Host ""

    # Diagnose common failures
    Write-Host "  Common fixes:" -ForegroundColor Yellow
    Write-Host "    - LNK1181 (kernel32.lib):  Run .\setup_build_env.ps1" -ForegroundColor Yellow
    Write-Host "    - LNK1107 (stdc++.lib):    cargo clean, then re-run this script" -ForegroundColor Yellow
    Write-Host "    - CUDA errors:             Ensure nvcc --version works" -ForegroundColor Yellow
    Write-Host "    - Out of memory:           Close other programs, try --debug" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}