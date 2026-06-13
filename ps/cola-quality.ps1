## cola-coder: Auto quality report for a checkpoint (syntax, types, tokens)
## Usage: .\cola-quality.ps1 --checkpoint checkpoints\small\latest --config configs\small.yaml
## Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\quality_report.py @args
} finally { Pop-Location }
