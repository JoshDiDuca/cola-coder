## cola-coder: Start the inference HTTP server
## Lives in <project>\ps\ — project root is the parent of this folder.
## Tip: the VS Code extension needs the server started with --cors.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\serve.py @args
} finally { Pop-Location }
