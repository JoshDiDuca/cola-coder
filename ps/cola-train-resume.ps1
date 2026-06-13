## cola-coder: Resume (or start) the best-quality small TS/React training run,
## DETACHED so it survives this shell, resuming ONLY from its own checkpoints.
## Run this after a reboot to continue training where it left off.
##
## Usage:
##   .\cola-train-resume.ps1
##   .\cola-train-resume.ps1 -Config configs\auto\small_react_best.yaml -Data E:\...\code_data.npy
##
## Why not `train.py --auto-resume`? auto-resume scans globally and can latch
## onto an architecturally-INCOMPATIBLE checkpoint from a different run (e.g.
## checkpoints/small without qk_norm) and crash on load (BUG-118). This wrapper
## resumes strictly from the run's OWN output_dir, or starts fresh if empty.
param(
    [string]$Config = "configs\auto\small_react_best.yaml",
    [string]$Data   = "E:\cola-coder-data\data\typescript-text-math\code_data.npy",
    [string]$OutputDir = "checkpoints\small_react_best"
)
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$py = Join-Path $project ".venv\Scripts\python.exe"
# Triton (torch.compile backend) is not installed on this Windows box; without
# this, compile crashes at step 0 instead of falling back to eager (BUG-119).
$env:COLA_NO_COMPILE = "1"
if (-not (Test-Path $py))   { Write-Host "venv missing — run cola-setup.ps1 first." -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Data)) { Write-Host "training data not found: $Data" -ForegroundColor Red; exit 1 }

# Already running? (a python training process). Don't double-launch.
$running = Get-Process python* -ErrorAction SilentlyContinue
if ($running) { Write-Host "A python process is already running (PID $($running.Id -join ',')). Not relaunching." -ForegroundColor Yellow; exit 0 }

$cmd = @("scripts\train.py", "--config", $Config, "--data", $Data)
$ckpt = Join-Path $project $OutputDir
$own = @()
if (Test-Path $ckpt) {
    $own = Get-ChildItem $ckpt -Directory -Filter "step_*" -ErrorAction SilentlyContinue
}
if ($own.Count -gt 0) {
    $latest = ($own | Sort-Object { [int]($_.Name -replace 'step_','') })[-1].FullName
    $cmd += @("--resume", $latest)
    Write-Host "Resuming from $latest"
} else {
    Write-Host "No checkpoint in $OutputDir yet -> fresh start (step 0)."
}

$p = Start-Process -FilePath $py -ArgumentList $cmd -WorkingDirectory $project `
    -RedirectStandardOutput "$project\train_small_react_best.log" `
    -RedirectStandardError  "$project\train_small_react_best.err" `
    -WindowStyle Hidden -PassThru
Write-Host "Launched training PID $($p.Id) (detached). Monitor: .\cola-smoke.ps1 / Get-Content train_small_react_best.log -Tail 20 -Wait"
