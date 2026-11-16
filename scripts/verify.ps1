Set-Location (Split-Path -Parent $PSScriptRoot)
if (Test-Path ".venv\Scripts\python.exe") { .\.venv\Scripts\python.exe main.py verify } else { python main.py verify }
