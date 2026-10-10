$ErrorActionPreference = "Stop"
$Repo = "SunayHegde2006/Air.rs"
$InstallDir = if ($env:AIR_INSTALL_DIR) { $env:AIR_INSTALL_DIR } else { Join-Path $env:USERPROFILE ".local\bin" }

if ($args -contains "--uninstall" -or $args -contains "-u" -or $args -contains "uninstall") {
    Write-Host "[air] Uninstalling Air.rs..." -ForegroundColor Yellow
    Remove-Item (Join-Path $InstallDir "air-rs.exe") -Force -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $env:USERPROFILE ".air-rs") -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "[air] Air.rs uninstalled successfully." -ForegroundColor Green
    exit 0
}

Write-Host "[air] Installing Air.rs v2.0.0 on Windows..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

$Installed = $false
try {
    $Release = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest"
    $Version = $Release.tag_name -replace '^v', ''
    $Asset = $Release.assets | Where-Object { $_.name -like "*windows*" -or $_.name -like "*.exe*" } | Select-Object -First 1

    if ($Asset) {
        $ZipPath = Join-Path $env:TEMP "air-rs.zip"
        Invoke-WebRequest -Uri $Asset.browser_download_url -OutFile $ZipPath
        Expand-Archive -Path $ZipPath -DestinationPath $InstallDir -Force
        Remove-Item $ZipPath -Force -ErrorAction SilentlyContinue
        $Installed = $true
        Write-Host "[air] Downloaded pre-built binary to $InstallDir" -ForegroundColor Green
    }
} catch {
    Write-Host "[air] Release download skipped, checking local builds..." -ForegroundColor Yellow
}

if (-not $Installed) {
    if (Test-Path "target\release\air-cli.exe") {
        Copy-Item "target\release\air-cli.exe" "$InstallDir\air-rs.exe" -Force
        $Installed = $true
    } elseif (Test-Path "target\release\air-rs.exe") {
        Copy-Item "target\release\air-rs.exe" "$InstallDir\air-rs.exe" -Force
        $Installed = $true
    } elseif (Get-Command cargo -ErrorAction SilentlyContinue) {
        Write-Host "[air] Building from source with Cargo..." -ForegroundColor Cyan
        cargo build --release --workspace
        if (Test-Path "target\release\air-cli.exe") {
            Copy-Item "target\release\air-cli.exe" "$InstallDir\air-rs.exe" -Force
            $Installed = $true
        }
    }
}

if ($Installed) {
    $CurrentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($CurrentPath -notlike "*$InstallDir*") {
        [Environment]::SetEnvironmentVariable("Path", "$CurrentPath;$InstallDir", "User")
        Write-Host "[air] Added $InstallDir to User PATH environment variable." -ForegroundColor Green
    }
    Write-Host "[air] ✓ Air.rs installed successfully to $InstallDir\air-rs.exe" -ForegroundColor Green
} else {
    Write-Host "[air-error] Could not install pre-built binary or compile via Cargo." -ForegroundColor Red
    Write-Host "[air-error] Install Rust via https://rustup.rs and run 'cargo build --release'." -ForegroundColor Red
    exit 1
}
