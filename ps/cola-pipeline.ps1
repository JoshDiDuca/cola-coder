## cola-coder: Full Auto Pipeline — detect hardware, pick the best config, run all stages
## Usage:
##   .\cola-pipeline.ps1                 # full run (prompts for confirmation)
##   .\cola-pipeline.ps1 -Smoke          # fast wiring check (~minutes, tiny steps)
##   .\cola-pipeline.ps1 -DryRun         # print the plan, run nothing
##   .\cola-pipeline.ps1 -Yes            # skip confirmation prompts
##   .\cola-pipeline.ps1 -BaseConfig configs\4080_max.yaml
## Lives in <project>\ps\ — project root is the parent of this folder.
param(
    [switch]$Smoke,
    [switch]$DryRun,
    [switch]$Yes,
    [string]$BaseConfig = ""
)
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$py = Join-Path $project ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv missing — run cola-setup.ps1 first." -ForegroundColor Red; exit 1 }
Push-Location $project
try {
    $cmdArgs = @("scripts\auto_pipeline.py")
    if ($Smoke)      { $cmdArgs += "--smoke" }
    if ($DryRun)     { $cmdArgs += "--dry-run" }
    if ($Yes)        { $cmdArgs += "--yes" }
    if ($BaseConfig) { $cmdArgs += "--base-config", $BaseConfig }
    & .\.venv\Scripts\python @cmdArgs @args
} finally { Pop-Location }
