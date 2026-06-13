## cola-coder: Train the semantic domain router (stage 8)
## Usage: .\cola-router.ps1 --arch mlp --generate-data
##        .\cola-router.ps1 --data data\router_training_data.jsonl --arch mlp
## Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$py = Join-Path $project ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv missing — run cola-setup.ps1 first." -ForegroundColor Red; exit 1 }
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\train_router.py @args
} finally { Pop-Location }
