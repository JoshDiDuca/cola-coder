## cola-coder: Train a model (tiny, small, medium, 4080_max, large, or reasoning)
## Usage: .\cola-train.ps1 [tiny|small|medium|4080_max|large|reasoning] [-- extra args]
## Lives in <project>\ps\ — project root is the parent of this folder.
param(
    [ValidateSet("tiny","small","medium","4080_max","large","reasoning")]
    [string]$Size = "tiny"
)
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    if ($Size -eq "reasoning") {
        & .\.venv\Scripts\python scripts\train_reasoning.py @args
    } else {
        & .\.venv\Scripts\python scripts\train.py --config "configs\$Size.yaml" @args
    }
} finally { Pop-Location }
