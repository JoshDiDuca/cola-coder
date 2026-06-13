## cola-coder: Prepare data for the tiny model config
## Usage: .\cola-prepare-tiny.ps1 [-MaxTokens 50000000]
## Defaults to 50M tokens — enough for ~76k training steps with tiny config.
## Lives in <project>\ps\ — project root is the parent of this folder.
## Tokenizer auto-resolves from the data-sources/storage config (override with -Tokenizer).
param(
    [int]$MaxTokens = 50000000,
    [int]$Workers = 0,
    [int]$BatchSize = 256,
    [string]$Tokenizer = ""
)
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    if ($Workers -eq 0) {
        $Workers = [Math]::Min((Get-CimInstance Win32_Processor).NumberOfLogicalProcessors, 16)
    }
    Write-Host "Workers: $Workers | Batch size: $BatchSize | Max tokens: $($MaxTokens.ToString('N0'))" -ForegroundColor Cyan
    $cmdArgs = @(
        "scripts\prepare_data.py",
        "--config", "configs\tiny.yaml",
        "--max-tokens", $MaxTokens,
        "--workers", $Workers,
        "--batch-size", $BatchSize
    )
    if ($Tokenizer) { $cmdArgs += "--tokenizer", $Tokenizer }
    & .\.venv\Scripts\python @cmdArgs @args
} finally { Pop-Location }
