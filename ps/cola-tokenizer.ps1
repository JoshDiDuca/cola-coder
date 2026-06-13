## cola-coder: Train the BPE tokenizer on downloaded code data
## Usage: .\cola-tokenizer.ps1 [-VocabSize 32768] [-NumSamples 10000] [-Config configs\tiny.yaml] [-Output path]
## Lives in <project>\ps\ — project root is the parent of this folder.
## Output: when -Output is omitted, train_tokenizer.py auto-resolves it to the
## dataset's tokenizer.json (data/<dataset>/tokenizer.json) per storage.yaml.
param(
    [int]$VocabSize = 32768,
    [int]$NumSamples = 10000,
    [string]$Config = "",
    [string]$Output = ""
)
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$py = Join-Path $project ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv missing — run cola-setup.ps1 first." -ForegroundColor Red; exit 1 }
Push-Location $project
try {
    $cmdArgs = @(
        "scripts\train_tokenizer.py",
        "--vocab-size", $VocabSize,
        "--num-samples", $NumSamples
    )
    if ($Config) { $cmdArgs += "--config", $Config }
    if ($Output) { $cmdArgs += "--output", $Output }
    & .\.venv\Scripts\python @cmdArgs @args
} finally { Pop-Location }
