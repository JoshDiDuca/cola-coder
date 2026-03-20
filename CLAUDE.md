# Cola-Coder

A from-scratch code generation transformer. Human (Josh) + Claude collaboration.
Josh is an experienced TypeScript developer learning ML — frame explanations in TS analogies where helpful.

## Quick Reference

- **Language:** Python 3.10+, PyTorch 2.2+
- **Package manager:** pip with venv (`.venv/`)
- **Install:** `python -m venv .venv && .venv/Scripts/pip install -e ".[dev,logging]"`
- **Platform:** Windows 11 (no `make` — use PowerShell scripts in parent dir `~/ai research/cola-*.ps1`)
- **Configs:** `configs/tiny.yaml` (50M), `small.yaml` (125M), `medium.yaml` (350M), `large.yaml` (1B+)
- **GPU:** RTX 4080 (16GB, bf16) and RTX 3080 (10GB, fp16+GradScaler). Use `precision: "bf16"` for 4080, `precision: "fp16"` for 3080.
- **RAM:** 64GB system

## Architecture

Decoder-only transformer: RoPE positional encoding, Grouped Query Attention (GQA), SwiGLU activation, RMSNorm (pre-norm), AdamW optimizer, cosine LR with linear warmup. Same architecture as LLaMA 3 / Mistral / DeepSeek-Coder.

Checkpoints use safetensors format (not pickle). Tokenizer is HuggingFace BPE (Rust-backed).

## Project Layout

```
configs/              YAML model & training configs
docs/                 Educational guides (01-05 + deep-dives/)
src/cola_coder/
  model/              Transformer: attention, feedforward, normalization, rope, config
  tokenizer/          BPE tokenizer training & utilities
  data/               Download, preprocess, quality filter, dataset, collator
  training/           Trainer loop, optimizer, checkpoint, metrics
  inference/          KV-cache generator, sampling, FastAPI server
  evaluation/         HumanEval benchmark, pass@k metrics
  reasoning/          CoT thinking tokens, GRPO, reward functions
scripts/              CLI entry points (train.py, prepare_data.py, etc.)
tests/                Unit tests
```

## Training Pipeline (3 stages)

1. **Tokenizer** — `scripts/train_tokenizer.py` — downloads code, trains BPE tokenizer -> `tokenizer.json`
2. **Data prep** — `scripts/prepare_data.py` — streams from HuggingFace, quality-filters, tokenizes, chunks -> `data/processed/train_data.npy`
3. **Training** — `scripts/train.py --config configs/<size>.yaml` — the actual training loop

Data prep requires `--tokenizer tokenizer.json`. Use `--config configs/tiny.yaml` to read dataset/language settings. Use `--max-tokens N` to cap data size.

Data prep downloads parquet files in bulk first (cached in ~/.cache/huggingface), then processes locally at disk speed. Use `--stream` for the old slow HTTP-per-row mode. Quality filtering runs in parallel with `--workers N`. Chunks stream to a memmap file to cap RAM usage.

Key flags: `--workers N` (parallel filter processes, default: CPU cores), `--batch-size N` (files per tokenization batch, default: 256), `--stream` (slow HTTP mode), `--no-filter`, `--filter-strict`.

Prepared data is reusable across training runs. Only re-prepare if tokenizer, seq_len, dataset, languages, or filter mode changes.

## Key Training Concepts

- Loss starts ~10.4 (random), target 2.0-2.5 for small, 1.8-2.2 for medium
- Perplexity = exp(loss), target 8-15 for good code generation
- Gradient accumulation: effective_batch = batch_size * gradient_accumulation
- Gradient checkpointing: required for medium (350M) on 16GB, trades ~30% speed for ~50% VRAM
- bf16 on 4080 (no GradScaler), fp16 on 3080 (needs GradScaler)
- Total tokens = effective_batch * max_seq_len * max_steps

## Training Data

- Source: `bigcode/starcoderdata` (gated — needs HF_TOKEN with accepted access)
- Default languages: Python, TypeScript, JavaScript
- Quality filter at `src/cola_coder/data/quality_filter.py`: conservative (default, ~48% rejection on raw GitHub data) or strict (~65%)
- Filter checks: length, line length, char diversity, auto-generated headers, data files, comment ratio, test dumps, syntax parsing, brace balance

## Reasoning Module

- Thinking tokens: `<think>` / `</think>` for chain-of-thought
- GRPO (Group Relative Policy Optimization): generate multiple solutions, run tests, reinforce correct ones
- Config: `configs/reasoning.yaml`
- Script: `scripts/train_reasoning.py`

## Vision: Multi-Agent Specialization

End goal is a router model (125M) + domain-specific specialists (50M each: React, Next.js, GraphQL, Prisma, Zod, Testing) + general TS fallback (125M). Each specialist trains independently in 1-2 days. Active per request: ~175M params. See `docs/deep-dives/multi-agent-specialization.md`.

## Commands

```bash
# Windows — use .venv/Scripts/python explicitly (no make)
.venv/Scripts/python scripts/train_tokenizer.py
.venv/Scripts/python scripts/prepare_data.py --config configs/tiny.yaml --tokenizer tokenizer.json  # --workers 4 --batch-size 64
.venv/Scripts/python scripts/train.py --config configs/tiny.yaml
.venv/Scripts/python scripts/generate.py --checkpoint checkpoints/tiny/latest
.venv/Scripts/python scripts/evaluate.py --checkpoint checkpoints/tiny/latest
.venv/Scripts/python scripts/serve.py --checkpoint checkpoints/tiny/latest

# Or use PowerShell wrapper scripts from ~/ai research/
# cola-tokenizer.ps1, cola-prepare.ps1, cola-train.ps1, cola-generate.ps1, etc.

# Tests & lint
.venv/Scripts/pytest tests/ -v
.venv/Scripts/ruff check src/ scripts/ tests/
```

## Code Style

- Line length: 100 (ruff config in pyproject.toml)
- Linter: ruff
- Tests: pytest
- Type hints used but not strictly enforced

## Hardware Estimates (RTX 4080)

| Config | Params | VRAM | Throughput | Training Time |
|--------|--------|------|-----------|---------------|
| tiny   | 50M    | ~3.6 GB | ~45k tok/s | ~4 hours |
| small  | 125M   | ~6.5 GB | ~35k tok/s | ~2 days |
| medium | 350M   | ~8.2 GB | ~22k tok/s | ~7 days |
| large  | 1B+    | ~24 GB  | N/A        | cloud only |

## Important Notes

- HuggingFace dataset is gated: set `HF_TOKEN` env var and accept terms at huggingface.co/datasets/bigcode/starcoderdata
- Always verify GPU is being used: `nvidia-smi` should show >90% utilization during training
- Resume from checkpoint: `--resume checkpoints/<size>/latest`
- wandb logging: `--wandb` flag on train.py (needs `pip install wandb` + `wandb login`)
- Checkpoints save model (safetensors) + optimizer state (pt) + metadata (json)
- The project is written for learning — all docs are in `docs/` and written for TS developers
