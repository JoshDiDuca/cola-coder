## cola-coder: Unified Master Menu — single entry point for all operations
## Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\menu.py @args
} finally { Pop-Location }
