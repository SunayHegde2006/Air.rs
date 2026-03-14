# Run tests after loading MSVC environment
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsInstallPath = & $vswhere -latest -property installationPath
$vsPath = Join-Path $vsInstallPath "VC\Auxiliary\Build\vcvars64.bat"
cmd /c " `"$vsPath`" && set " | Foreach-Object {
    if ($_ -match "^(.*?)=(.*)$") {
        Set-Content "env:\$($Matches[1])" $Matches[2]
    }
}
$env:CL = "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
$env:CUDA_COMPUTE_CAP = "86"
$env:RUSTFLAGS = $null

cargo test --no-default-features -- --test-threads=1
