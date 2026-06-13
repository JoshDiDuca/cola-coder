## cola-coder: Environment validation — Python, PyTorch/CUDA, GPU, deps, config sanity
## Usage:
##   .\cola-env-check.ps1                # full check (incl. a quick internet probe)
##   .\cola-env-check.ps1 --no-internet  # skip the network probe (offline machines)
## Lives in <project>\ps\ — project root is the parent of this folder.
## Run this first on a new Windows machine to confirm the setup is good.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\env_check.py @args
} finally { Pop-Location }
