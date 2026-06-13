## cola-coder: Side-by-side comparison of two checkpoints/models
## Usage: .\cola-compare.ps1 --checkpoint-a checkpoints\small\latest --checkpoint-b checkpoints\small_react_best\latest --config configs\small.yaml
## Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$py = Join-Path $project ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv missing — run cola-setup.ps1 first." -ForegroundColor Red; exit 1 }
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\compare_models.py @args
} finally { Pop-Location }
