## cola-coder: Start the local web UI (FastAPI backend + built React dashboard).
## Lives in <project>\ps\ — project root is the parent of this folder.
## Serves the API AND the built webui/dist at one URL (default http://127.0.0.1:8800)
## and opens it in your browser. Streams live training progress over SSE; safe to run
## alongside the live training run (the UI refuses to start a second trainer).
##
## Usage:
##   .\ps\cola-ui.ps1                 # open http://127.0.0.1:8800
##   .\ps\cola-ui.ps1 --port 9000     # any extra args pass through to ui_server.py
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$py = Join-Path $project ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv missing — run cola-setup.ps1 first." -ForegroundColor Red; exit 1 }
Push-Location $project
try {
    & $py scripts\ui_server.py --open @args
} finally { Pop-Location }
