## cola-coder: Create venv and install dependencies
## Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    python -m venv .venv
    & .\.venv\Scripts\python -m pip install --upgrade pip
    & .\.venv\Scripts\pip install -e ".[dev,logging]"
} finally { Pop-Location }
