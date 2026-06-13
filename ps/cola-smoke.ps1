## cola-coder: Smoke test — fast 8-check validation of a checkpoint (~30s)
## Usage:
##   .\cola-smoke.ps1                                   # auto-detect latest checkpoint
##   .\cola-smoke.ps1 --checkpoint checkpoints\tiny\latest --config configs\tiny.yaml
##   .\cola-smoke.ps1 --quick                           # subset of checks
## Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\smoke_test.py @args
} finally { Pop-Location }
