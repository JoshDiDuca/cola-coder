# Skill: Project Architecture

## Layout
```
src/cola_coder/
  model/       — Transformer: attention (GQA), feedforward (SwiGLU), normalization (RMSNorm), rope, config
  tokenizer/   — HuggingFace BPE tokenizer (Rust-backed)
  data/        — Download, preprocess, quality filter, quality scorer, dataset
    filters/   — Modular filter plugins, ML quality classifier
    sources/   — Data source connectors (HF, GitHub, local)
    curation/  — Test execution scoring + Docker sandbox
  training/    — Trainer loop, optimizer, checkpoint, metrics
  inference/   — KV-cache generator, sampling, FastAPI server
  evaluation/  — HumanEval benchmark, pass@k
  reasoning/   — CoT thinking tokens, GRPO, reward functions
  features/    — 83 optional feature modules
  cli.py       — Shared CLI styling (rich + questionary arrow-key menus)
scripts/       — 22 CLI entry points (menu.py is the master menu)
configs/       — YAML model & training configs + features.yaml + storage.yaml
```

## Architecture
Decoder-only transformer (LLaMA 3 / Mistral style):
- RoPE positional encoding
- Grouped Query Attention (GQA)
- SwiGLU activation
- RMSNorm (pre-norm)
- Safetensors checkpoints

## Hardware
- RTX 4080 SUPER (16GB, bf16) — primary training GPU
- RTX 3080 (10GB, fp16) — secondary
- 64GB RAM, Windows 11

## Vision: Multi-Agent Specialization
Router model (125M) + domain-specific specialists (50M each):
React, Next.js, GraphQL, Prisma, Zod, Testing + general TS fallback (125M).
Active per request: ~175M params.

## Platform Notes
- Windows 11 — no `make`, use PowerShell scripts
- Use `.venv/Scripts/python` (not `.venv/bin/python`)
- python.exe required in path checks (not `python` without extension)
