## cola-coder: Guided interactive data preparation
## Usage: .\cola-prepare-menu.ps1 [-BatchSize 256] [-Tokenizer path]
## Lives in <project>\ps\ — project root is the parent of this folder.
## Tokenizer: omit -Tokenizer to auto-resolve from the data-sources/storage config.
param(
    [string]$Tokenizer = "",
    [int]$BatchSize = 256
)
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    $cmdArgs = @("scripts\prepare_data_interactive.py", "--batch-size", $BatchSize)
    if ($Tokenizer) { $cmdArgs += "--tokenizer", $Tokenizer }
    & .\.venv\Scripts\python @cmdArgs @args
} finally { Pop-Location }
