## cola-coder: Run the full evaluation suite in sequence
## Usage: .\cola-eval-suite.ps1 --checkpoint checkpoints\small\latest --config configs\small.yaml
## Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\run_eval_suite.py @args
} finally { Pop-Location }
