## cola-coder: Safety probes on generated code (secrets, dangerous patterns)
## Usage: .\cola-safety.ps1 --checkpoint checkpoints\small\latest --config configs\small.yaml --suite extended
## Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\safety_eval.py @args
} finally { Pop-Location }
