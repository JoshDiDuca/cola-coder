## cola-coder: Train a model (tiny, small, medium, 4080_max, large, or reasoning)
## Usage: .\cola-train.ps1 [tiny|small|medium|4080_max|large|reasoning] [-- extra args]
## Lives in <project>\ps\ — project root is the parent of this folder.
param(
    [ValidateSet("tiny","small","medium","4080_max","large","reasoning")]
    [string]$Size = "tiny"
)
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$py = Join-Path $project ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv missing — run cola-setup.ps1 first." -ForegroundColor Red; exit 1 }
Push-Location $project
try {
    if ($Size -eq "reasoning") {
        & .\.venv\Scripts\python scripts\train_reasoning.py @args
    } else {
        & .\.venv\Scripts\python scripts\train.py --config "configs\$Size.yaml" @args
    }
} finally { Pop-Location }
