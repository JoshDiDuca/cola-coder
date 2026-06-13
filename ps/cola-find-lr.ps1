## cola-coder: Learning-rate range finder
## Usage: .\cola-find-lr.ps1 --config configs\small.yaml
## Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\find_lr.py @args
} finally { Pop-Location }
