# Cola-Coder: Quick Run — interactive code generation (REPL)
# Usage: .\cola-run.ps1 [--preset creative|balanced|precise]
# Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\run.py @args
} finally { Pop-Location }
