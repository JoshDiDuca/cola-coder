## cola-coder: Auto quality report for a checkpoint (syntax, types, tokens)
## Usage: .\cola-quality.ps1 --checkpoint checkpoints\small\latest --config configs\small.yaml
## Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$py = Join-Path $project ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv missing — run cola-setup.ps1 first." -ForegroundColor Red; exit 1 }
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\quality_report.py @args
} finally { Pop-Location }
