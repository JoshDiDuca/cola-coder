# Cola-Coder: Quick Benchmark — test your trained model
# Usage: .\cola-benchmark.ps1 [--checkpoint path\to\ckpt]
# Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\benchmark.py @args
} finally { Pop-Location }
