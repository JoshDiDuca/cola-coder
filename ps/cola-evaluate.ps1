## cola-coder: Run evaluation (HumanEval pass@k)
## Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\evaluate.py @args
} finally { Pop-Location }
