## cola-coder: Export a trained model (GGUF / Ollama / quantized)
## Usage:
##   .\cola-export.ps1 --action gguf --checkpoint checkpoints\tiny\latest --config configs\tiny.yaml
##   .\cola-export.ps1 --action ollama --checkpoint checkpoints\tiny_sft\latest --config configs\tiny.yaml
## Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\export_model.py @args
} finally { Pop-Location }
