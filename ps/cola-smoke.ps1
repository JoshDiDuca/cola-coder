## cola-coder: Smoke test — fast 8-check validation of a checkpoint (~30s)
## Usage:
##   .\cola-smoke.ps1                                   # auto-detect latest checkpoint
##   .\cola-smoke.ps1 --checkpoint checkpoints\tiny\latest --config configs\tiny.yaml
##   .\cola-smoke.ps1 --quick                           # subset of checks
## Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$py = Join-Path $project ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv missing — run cola-setup.ps1 first." -ForegroundColor Red; exit 1 }
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\smoke_test.py @args
} finally { Pop-Location }
