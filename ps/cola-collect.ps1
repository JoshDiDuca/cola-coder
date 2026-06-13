## cola-coder: Multi-source data collection (code 70% + text 20% + math 10%)
## Usage:
##   .\cola-collect.ps1 -Config configs\tiny.yaml
##   .\cola-collect.ps1 -Config configs\4080_max.yaml -Score   # quality-weighted
## Lives in <project>\ps\ — project root is the parent of this folder.
## Ratios/sources come from configs\data_sources.yaml. Tokenizer auto-resolves.
param(
    [string]$Config = "configs\tiny.yaml",
    [switch]$Score
)
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    $cmdArgs = @("scripts\collect_data.py", "--config", $Config)
    if ($Score) { $cmdArgs += "--score" }
    & .\.venv\Scripts\python @cmdArgs @args
} finally { Pop-Location }
