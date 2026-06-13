## cola-coder: Generate distillation SFT data from a teacher (local Qwen/DeepSeek or cloud).
## Untrusted teacher output is verified only inside the sandbox (SEC-014).
## Usage: .\cola-distill-data.ps1 --prompts data\sft\seed_prompts.jsonl --output data\sft\distilled.jsonl --language ts --verify
## Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$py = Join-Path $project ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv missing — run cola-setup.ps1 first." -ForegroundColor Red; exit 1 }
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\generate_distillation_data.py @args
} finally { Pop-Location }
