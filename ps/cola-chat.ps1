## cola-coder: Multi-turn chat REPL (keeps conversation history)
## Usage:
##   .\cola-chat.ps1                                    # auto-detect latest checkpoint
##   .\cola-chat.ps1 --checkpoint checkpoints\tiny_sft\latest --config configs\tiny.yaml
##   .\cola-chat.ps1 --system "You are a helpful coding assistant." --temperature 0.7
## Lives in <project>\ps\ — project root is the parent of this folder.
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    & .\.venv\Scripts\python scripts\chat.py @args
} finally { Pop-Location }
