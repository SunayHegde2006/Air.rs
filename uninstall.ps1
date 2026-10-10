$ErrorActionPreference = "Stop"
$InstallDir = if ($env:AIR_INSTALL_DIR) { $env:AIR_INSTALL_DIR } else { Join-Path $env:USERPROFILE ".local\bin" }

Write-Host "[air] Uninstalling Air.rs..." -ForegroundColor Yellow
Remove-Item (Join-Path $InstallDir "air-rs.exe") -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $env:USERPROFILE ".air-rs") -Recurse -Force -ErrorAction SilentlyContinue

$CurrentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($CurrentPath -like "*$InstallDir*") {
    $NewPath = ($CurrentPath -split ';' | Where-Object { $_ -and $_ -ne $InstallDir -and $_ -notlike "*\.air-rs*" }) -join ';'
    [Environment]::SetEnvironmentVariable("Path", $NewPath, "User")
}

Write-Host "[air] ✓ Air.rs uninstalled successfully." -ForegroundColor Green
