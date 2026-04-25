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
Write-Host "       Air.rs Build System                               " -ForegroundColor Magenta
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

# CUDA check
$hasCuda = $false
$cudaVersion = ""
try {
    $nvccOut = nvcc --version 2>&1 | Select-String "release"
    if ($nvccOut) {
        $hasCuda = $true
        $cudaVersion = ($nvccOut -replace '.*release ', '' -replace ',.*', '')
        Write-Step "CUDA Toolkit: $cudaVersion"
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
    Write-Host "  Available features:" -ForegroundColor White
    Write-Host ""

    # CUDA
    if ($hasCuda) {
        Write-Host "    [1] cuda         - NVIDIA GPU acceleration (CUDA $cudaVersion detected)" -ForegroundColor Green
    } else {
        Write-Host "    [1] cuda         - NVIDIA GPU (not available - no CUDA)" -ForegroundColor DarkGray
    }

    # Flash Attention
    if ($hasCuda) {
        Write-Host "    [2] flash-attn   - Flash Attention 2 (requires CUDA)" -ForegroundColor Green
    } else {
        Write-Host "    [2] flash-attn   - Flash Attention 2 (requires CUDA)" -ForegroundColor DarkGray
    }

    # Vulkan
    if ($hasVulkan) {
        Write-Host "    [3] vulkan       - Vulkan 1.2 GPU compute (STRIX VulkanHal)" -ForegroundColor Green
    } else {
        Write-Host "    [3] vulkan       - Vulkan (not detected - install Vulkan SDK)" -ForegroundColor DarkGray
    }

    # Python
    Write-Host "    [4] python       - PyO3 Python bindings" -ForegroundColor Green

    # ARB optional deps
    Write-Host "    [5] arb-heap     - O(log n) priority queue for ARB scheduler (W>512)" -ForegroundColor Green
    Write-Host "    [6] arb-lockfree - Lock-free enqueue via crossbeam (high-freq HTTP)" -ForegroundColor Green

    # CPU only
    Write-Host "    [0] (none)       - CPU-only build" -ForegroundColor Yellow

    Write-Host ""
    $defaultChoice = if ($hasCuda) { "1,2" } else { "0" }
    $choice = Read-Host "  Select features (comma-separated, e.g. 1,2) [default: $defaultChoice]"
    if ([string]::IsNullOrWhiteSpace($choice)) { $choice = $defaultChoice }

    $selections = $choice -split ',' | ForEach-Object { $_.Trim() }

    foreach ($sel in $selections) {
        switch ($sel) {
            '1' {
                if ($hasCuda) { $features += 'cuda' }
                else { Write-Warn "Skipping cuda - CUDA Toolkit not detected" }
            }
            '2' {
                if ($hasCuda) { $features += 'flash-attn' }
                else { Write-Warn "Skipping flash-attn - requires CUDA" }
            }
            '3' {
                if ($hasVulkan) { $features += 'vulkan' }
                else { Write-Warn "Skipping vulkan - Vulkan runtime not detected" }
            }
            '4' { $features += 'python' }
            '5' { $features += 'arb-heap' }
            '6' { $features += 'arb-lockfree' }
            '0' { }
            default { Write-Warn "Unknown selection: $sel (ignored)" }
        }
    }
}

# ============================================================================
# STEP 4: PRE-BUILD CLEANUP
# ============================================================================
Write-Host ""
Write-Host "  --- Step 4: Pre-Build ---" -ForegroundColor White
Write-Host ""

# Clean stale stdc++.lib stubs (the old 8-byte ones that cause LNK1107)
$staleStubs = Get-ChildItem -Path ".\target" -Recurse -Filter "stdc++.lib" -ErrorAction SilentlyContinue
foreach ($stub in $staleStubs) {
    $size = $stub.Length
    if ($size -lt 72) {
        Remove-Item $stub.FullName -Force
        Write-Info "Removed invalid stdc++.lib stub ($size bytes) at $($stub.FullName)"
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

# ============================================================================
# STEP 5: BUILD
# ============================================================================
Write-Host ""
Write-Host "  --- Step 5: Building Air.rs ($profileName) ---" -ForegroundColor White
Write-Host ""

$cmd = "cargo build $buildProfile $featureArg"
Write-Info "Running: $cmd"
Write-Host ""

# OCS Algorithm Summary
Write-Host ""
Write-Host "  --- Optimal Compounding Stack (always enabled) ---" -ForegroundColor White
Write-Host ""
Write-Host "  [+] SageAttention3 FP4 microscaling    (fp4_attention, ops.rs)" -ForegroundColor Green
Write-Host "  [+] KIMI Linear Attention O(N*D2)       (linear_attention_kimi, ops.rs)" -ForegroundColor Green
Write-Host "  [+] Gated Attention sink-suppression    (gated_attention, ops.rs)" -ForegroundColor Green
Write-Host "  [+] QJL 1-bit JL-transform KV keys      (QjlKey, kv_compress.rs)" -ForegroundColor Green
Write-Host "  [+] Fast KV Compaction (cosine merge)   (compact_kv_by_similarity, kv_compress.rs)" -ForegroundColor Green
Write-Host "  [+] HERMES importance-scored eviction   (HermesTierManager, kv_tier.rs)" -ForegroundColor Green
Write-Host "  [+] ConceptMoE adaptive token routing   (concept_moe_forward, moe.rs)" -ForegroundColor Green
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