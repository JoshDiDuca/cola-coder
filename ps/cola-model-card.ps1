# Cola-Coder: Generate Model Card
# Usage: .\cola-model-card.ps1 [--checkpoint path\to\ckpt] [--output MODEL_CARD.md]
# Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\model_card.py @args
} finally { Pop-Location }
