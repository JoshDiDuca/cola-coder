# Skill: Running Cola-Coder Models

## Quick Start
- Auto-detect checkpoint: `python scripts/run.py` (finds latest checkpoint automatically)
- Specific checkpoint: `python scripts/run.py --checkpoint checkpoints/tiny/step_00020000`
- PowerShell wrapper: `cola-run.ps1` from `~/ai research/`

## Key Files
- `scripts/run.py` — Interactive REPL with auto-detection
- `src/cola_coder/inference/generator.py` — KV-cache generation (generate, generate_stream, generate_batch)
- `src/cola_coder/inference/sampling.py` — Temperature, top-k, top-p, repetition penalty

## Sampling Presets
- Precise: temp=0.2, top_k=10, top_p=0.85
- Balanced: temp=0.7, top_k=50, top_p=0.9
- Creative: temp=1.0, top_k=100, top_p=0.95

## REPL Commands
/clear, /preset, /info, /smoke, /history, /save, /quit

## Common Issues
- CUDA OOM: reduce max_new_tokens or use CPU
- Garbage output: model may not be trained enough, check loss < 2.5
- Slow generation: ensure CUDA is being used, not CPU
