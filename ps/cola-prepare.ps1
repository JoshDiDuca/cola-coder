## cola-coder: Download and preprocess training data
## Usage: .\cola-prepare.ps1 [-Config configs\tiny.yaml] [-MaxTokens 1000000] [-Tokenizer path]
## Lives in <project>\ps\ — project root is the parent of this folder.
## Tokenizer: omit -Tokenizer to auto-resolve from the data-sources/storage config.
## Performance: uses all CPU cores for filtering, batch size 256.
param(
    [string]$Config,
    [string]$Tokenizer = "",
    [int]$MaxTokens = 0,
    [int]$Workers = 0,
    [int]$BatchSize = 256
)
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    # Default workers: all CPU cores (up to 16)
    if ($Workers -eq 0) {
        $Workers = [Math]::Min((Get-CimInstance Win32_Processor).NumberOfLogicalProcessors, 16)
    }
    $cmdArgs = @(
        "scripts\prepare_data.py",
        "--workers", $Workers,
        "--batch-size", $BatchSize
    )
    if ($Config)    { $cmdArgs += "--config", $Config }
    if ($Tokenizer) { $cmdArgs += "--tokenizer", $Tokenizer }
    if ($MaxTokens) { $cmdArgs += "--max-tokens", $MaxTokens }
    Write-Host "Workers: $Workers | Batch size: $BatchSize" -ForegroundColor Cyan
    & .\.venv\Scripts\python @cmdArgs @args
} finally { Pop-Location }
