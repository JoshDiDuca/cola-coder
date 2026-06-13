## cola-coder: Export a trained model (GGUF / Ollama / quantized)
## Usage:
##   .\cola-export.ps1 --action gguf --checkpoint checkpoints\tiny\latest --config configs\tiny.yaml
##   .\cola-export.ps1 --action ollama --checkpoint checkpoints\tiny_sft\latest --config configs\tiny.yaml
## Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$py = Join-Path $project ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv missing — run cola-setup.ps1 first." -ForegroundColor Red; exit 1 }
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\export_model.py @args
} finally { Pop-Location }
