## cola-coder: Lint with ruff
## Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    & .\.venv\Scripts\ruff check src\ scripts\ tests\ @args
} finally { Pop-Location }
