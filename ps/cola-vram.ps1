## cola-coder: Estimate VRAM for a config before training
## Usage: .\cola-vram.ps1 --config configs\small.yaml
## Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\vram_estimate.py @args
} finally { Pop-Location }
