<# EXPLICIT, OPT-IN downloads. Example: scripts\download_models.ps1 -Groups core,nli  (add -DryRun first) #>
param([string[]]$Groups = @(), [switch]$DryRun, [switch]$List)
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = if (Test-Path ".venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
$a = @("scripts\download_models.py")
if ($List) { $a += "--list" } else { $a += "--group"; $a += $Groups }
if ($DryRun) { $a += "--dry-run" }
& $py @a
