# Air.rs Build Script — Windows 11 + CUDA 12.1

# 1. GPU & CUDA Health Check
Write-Host "`n--- [ Step 1: Health Check ] ---" -ForegroundColor Cyan
Write-Host "Checking NVIDIA Driver..."
nvidia-smi -L
if ($LASTEXITCODE -ne 0) { Write-Error "NVIDIA Driver not found!"; exit 1 }

Write-Host "Checking NVCC..."
nvcc --version | Select-String "release"
if ($LASTEXITCODE -ne 0) { Write-Error "CUDA Toolkit (nvcc) not in PATH!"; exit 1 }

# 2. Automatically find Visual Studio (auto-detect toolset)
Write-Host "`n--- [ Step 2: Hydrating Environment ] ---" -ForegroundColor Cyan
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsInstallPath = & $vswhere -latest -property installationPath

if (-not $vsInstallPath) {
    Write-Error "Could not detect any Visual Studio installation. Ensure 'Desktop development with C++' workload is installed."
    exit 1
}

$vsPath = Join-Path $vsInstallPath "VC\Auxiliary\Build\vcvars64.bat"

if (Test-Path $vsPath) {
    Write-Host "Found vcvars64.bat at $vsPath"
    # Load MSVC environment variables into PowerShell (auto-detects latest toolset)
    cmd /c " `"$vsPath`" && set " | Foreach-Object {
        if ($_ -match "^(.*?)=(.*)$") {
            Set-Content "env:\$($Matches[1])" $Matches[2]
        }
    }
    Write-Host "MSVC environment variables loaded."
} else {
    Write-Error "Found VS at $vsInstallPath but could not find vcvars64.bat."
    exit 1
}

# 3. Set CUDA compute capability and build flags
$env:CUDA_COMPUTE_CAP = "86"  # RTX 3060

# CUDA 12.1's nvcc doesn't officially support MSVC 19.36+ — suppress the STL version check
# bindgen_cuda doesn't forward NVCC_APPEND_FLAGS, so we use the CL env var which cl.exe reads directly
$env:CL = "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"

# Clear any stale RUSTFLAGS that may contain broken paths with spaces
$env:RUSTFLAGS = $null

# 4. Build
Write-Host "`n--- [ Step 3: Building Air.rs ] ---" -ForegroundColor Cyan
cargo build 2>&1 | Tee-Object -Variable buildOutput
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n--- BUILD FAILED ---" -ForegroundColor Red
    exit 1
}
Write-Host "`n--- BUILD SUCCEEDED ---" -ForegroundColor Green