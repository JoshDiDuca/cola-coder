## cola-coder: Instruction tuning / SFT (stage 6) on ChatML instruction pairs
## Usage:
##   .\cola-sft.ps1 -Checkpoint checkpoints\tiny\latest -Config configs\tiny.yaml
##   .\cola-sft.ps1 -Checkpoint checkpoints\4080_max\latest -Config configs\4080_max.yaml -Data data\sft\instructions.jsonl -Epochs 2 -Lr 2e-5
## Lives in <project>\ps\ — project root is the parent of this folder.
param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [Parameter(Mandatory = $true)][string]$Config,
    [string]$Data = "data\sft\instructions.jsonl",
    [int]$Epochs = 2,
    [string]$Lr = "2e-5"
)
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$py = Join-Path $project ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv missing — run cola-setup.ps1 first." -ForegroundColor Red; exit 1 }
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\train_sft.py `
        --data $Data `
        --config $Config `
        --checkpoint $Checkpoint `
        --epochs $Epochs `
        --lr $Lr `
        @args
} finally { Pop-Location }
