# setup_env.ps1 — Configure environment for building Air.rs on Windows 11.
# Run once per terminal session, or add to your $PROFILE.
#
# Usage:
#   . .\setup_env.ps1      # dot-source to set env in current session
#   cargo build             # will now find all Windows SDK libs

$ErrorActionPreference = "Stop"

# ── Find Windows SDK ──────────────────────────────────────────────────
$sdkRoot = "C:\Program Files (x86)\Windows Kits\10\Lib"
$sdkVersion = Get-ChildItem $sdkRoot -Directory |
    Where-Object { $_.Name -match '^10\.0\.' } |
    Sort-Object Name |
    Select-Object -Last 1 -ExpandProperty Name

if (-not $sdkVersion) {
    Write-Error "Windows 10/11 SDK not found in $sdkRoot"
    exit 1
}

$arch = if ([Environment]::Is64BitOperatingSystem) { "x64" } else { "x86" }

$umLib   = "$sdkRoot\$sdkVersion\um\$arch"
$ucrtLib = "$sdkRoot\$sdkVersion\ucrt\$arch"

# ── Find MSVC toolchain ──────────────────────────────────────────────
$msvcLib = $null
foreach ($edition in @("Professional", "Enterprise", "Community", "BuildTools")) {
    $msvcBase = "C:\Program Files\Microsoft Visual Studio\2022\$edition\VC\Tools\MSVC"
    if (Test-Path $msvcBase) {
        $msvcVersion = Get-ChildItem $msvcBase -Directory |
            Sort-Object Name |
            Select-Object -Last 1 -ExpandProperty Name
        if ($msvcVersion) {
            $msvcLib = "$msvcBase\$msvcVersion\lib\$arch"
            break
        }
    }
}

if (-not $msvcLib) {
    Write-Error "MSVC toolchain not found. Install VS2022 C++ Build Tools."
    exit 1
}

# ── Set LIB environment variable ─────────────────────────────────────
$env:LIB = "$umLib;$ucrtLib;$msvcLib"
Write-Host "✅ LIB set for Air.rs build:" -ForegroundColor Green
Write-Host "   SDK:  $umLib" -ForegroundColor Cyan
Write-Host "   UCRT: $ucrtLib" -ForegroundColor Cyan
Write-Host "   MSVC: $msvcLib" -ForegroundColor Cyan
Write-Host ""
Write-Host "Run 'cargo build' now." -ForegroundColor Yellow
