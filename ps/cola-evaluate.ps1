## cola-coder: Run evaluation (HumanEval pass@k)
## Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$py = Join-Path $project ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv missing — run cola-setup.ps1 first." -ForegroundColor Red; exit 1 }
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\evaluate.py @args
} finally { Pop-Location }
