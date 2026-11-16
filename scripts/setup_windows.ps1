<# Creates .venv (never overwrites an existing one) and installs core deps. Downloads NOTHING else.
   Usage:  powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 [-Profile core|dev|all] #>
param([ValidateSet("core","dev","all")][string]$Profile = "core")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (Test-Path ".venv") { Write-Host ".venv already exists - leaving it untouched." }
else {
  $py = (Get-Command py -ErrorAction SilentlyContinue)
  if ($py) { & py -3.10 -m venv .venv } else { python -m venv .venv }
}
$pip = ".\.venv\Scripts\python.exe"
& $pip -m pip install --upgrade pip
switch ($Profile) {
  "core" { & $pip -m pip install -r requirements.txt }
  "dev"  { & $pip -m pip install -r requirements-dev.txt }
  "all"  { Write-Host "Install CPU torch first (see requirements-all.txt header)."; & $pip -m pip install -r requirements-all.txt }
}
& $pip main.py verify
