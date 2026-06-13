## cola-coder: Quick test run of data preparation — measures throughput and estimates total time
## Usage: .\cola-prepare-test.ps1 [-MaxTokens 1000000]
## Lives in <project>\ps\ — project root is the parent of this folder.
## Tokenizer auto-resolves from the data-sources/storage config (override with -Tokenizer).
param(
    [int]$MaxTokens = 1000000,
    [string]$Tokenizer = ""
)
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$py = Join-Path $project ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv missing — run cola-setup.ps1 first." -ForegroundColor Red; exit 1 }
Push-Location $project
try {
    $cmdArgs = @(
        "scripts\prepare_data.py",
        "--config", "configs\tiny.yaml",
        "--max-tokens", $MaxTokens
    )
    if ($Tokenizer) { $cmdArgs += "--tokenizer", $Tokenizer }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & .\.venv\Scripts\python @cmdArgs @args
    $sw.Stop()
    $elapsed = $sw.Elapsed
    $tokensPerSec = [math]::Round($MaxTokens / $elapsed.TotalSeconds)

    # tiny config: 20k steps * 32 batch * 1024 seq = 655M tokens
    $fullTokens = 655000000
    $fullSecs = $fullTokens / $tokensPerSec
    $fullTime = [TimeSpan]::FromSeconds($fullSecs)

    Write-Host ""
    Write-Host "=== Test Results ===" -ForegroundColor Cyan
    Write-Host "  Processed:    $($MaxTokens.ToString('N0')) tokens"
    Write-Host "  Elapsed:      $($elapsed.ToString('hh\:mm\:ss\.ff'))"
    Write-Host "  Throughput:   $($tokensPerSec.ToString('N0')) tokens/sec"
    Write-Host ""
    Write-Host "=== Estimate for full tiny training set (655M tokens) ===" -ForegroundColor Yellow
    Write-Host "  Estimated:    $($fullTime.ToString('hh\:mm\:ss'))"
} finally { Pop-Location }
