<# Examples:  scripts\run_module.ps1 simulate   |   scripts\run_module.ps1 interview --resume cv.pdf --jd jd.txt --stt whisper #>
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = if (Test-Path ".venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
& $py main.py @args
