# Skill: Benchmarking Cola-Coder

## Quick Benchmark
- `python scripts/benchmark.py` — 5 test prompts, measures tok/s
- `python scripts/evaluate.py` — HumanEval pass@k evaluation
- `python scripts/compare_checkpoints.py` — compare two checkpoints

## Expected Performance (RTX 4080 SUPER)
- Tiny (50M): ~86 tok/s
- Small (125M): ~45 tok/s
- Medium (350M): ~22 tok/s

## Key Metrics
- Loss: cross-entropy loss (lower = better). Target: 1.5-2.5
- Perplexity: exp(loss). Target: 4-15
- tok/s: tokens per second during generation
- pass@k: probability of solving a problem in k attempts

## Files
- `scripts/benchmark.py` — Quick generation benchmark
- `scripts/evaluate.py` — HumanEval evaluation
- `scripts/compare_checkpoints.py` — Side-by-side checkpoint comparison
- `scripts/training_status.py` — CPU-only training progress check
