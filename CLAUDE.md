# Cola-Coder

A from-scratch code generation transformer. Human (Josh) + Claude collaboration.
Josh is an experienced TypeScript developer learning ML — frame explanations in TS analogies where helpful.

## Quick Reference

- **Language:** Python 3.10+, PyTorch 2.2+
- **Package manager:** pip with venv (`.venv/`)
- **Install:** `python -m venv .venv && .venv/Scripts/pip install -e ".[dev,logging]"`
- **Platform:** Windows 11 — use `.venv/Scripts/python`, no `make`
- **Configs:** `configs/tiny.yaml` (50M), `small.yaml` (125M), `medium.yaml` (299M), `4080_max.yaml` (455M), `large.yaml` (1B+)
- **GPU/RAM:** See `CLAUDE.local.md` for hardware specifics

## Architecture

Decoder-only transformer (LLaMA 3 / Mistral / DeepSeek-Coder): RoPE, GQA, SwiGLU, RMSNorm, AdamW, cosine LR. Optional MoE layer. Safetensors checkpoints. HuggingFace BPE tokenizer.

## Project Layout

```
configs/              YAML model & training configs (+ features.yaml, storage.yaml, reasoning.yaml)
src/cola_coder/
  model/              Transformer: attention, feedforward, normalization, rope, config
  tokenizer/          BPE tokenizer training & utilities
  data/               Download, preprocess, quality filter, FIM, dataset, collator
  training/           Trainer loop, optimizer, checkpoint, metrics, early stopping
  inference/          KV-cache generator, sampling, batched generation, FastAPI server
  evaluation/         HumanEval (62 problems), completion benchmark, pass@k, smoke tests
  reasoning/          CoT thinking tokens, GRPO, SFT warmup, reward registry, curriculum
  features/           166 feature modules — all toggled via configs/features.yaml
  cli.py              Shared CLI styling (rich + questionary arrow-key menus)
scripts/              45 CLI entry points
tests/                122 test files (~2600 tests)
docs/                 Educational guides (01-05) + deep-dives/
```

## Training Pipeline

1. **Tokenizer** — `scripts/train_tokenizer.py` → `tokenizer.json`
2. **Data prep** — `scripts/prepare_data.py --config configs/<size>.yaml --tokenizer tokenizer.json` → `data/processed/train_data.npy`
3. **Training** — `scripts/train.py --config configs/<size>.yaml`
4. **Reasoning** (optional) — `scripts/train_reasoning.py` — SFT warmup + GRPO with test-based rewards

Re-prepare data only if tokenizer, seq_len, dataset, languages, or filter mode changes.

## Key Commands

```bash
.venv/Scripts/python scripts/menu.py                           # Master menu
.venv/Scripts/python scripts/train.py --config configs/4080_max.yaml
.venv/Scripts/python scripts/train.py --config configs/4080_max.yaml --auto-resume
.venv/Scripts/python scripts/generate.py --checkpoint checkpoints/4080_max/latest --config configs/4080_max.yaml
.venv/Scripts/python scripts/evaluate.py --checkpoint checkpoints/4080_max/latest --config configs/4080_max.yaml
.venv/Scripts/python scripts/train_reasoning.py --config configs/4080_max.yaml --sft-warmup --reward combined --problems all

# Tests & lint (run before any training)
.venv/Scripts/pytest tests/ -v
.venv/Scripts/pytest tests/test_checkpoint.py -v   # CRITICAL — must pass before training
.venv/Scripts/ruff check src/ scripts/ tests/
.venv/Scripts/ruff check --fix src/ scripts/ tests/
```

## Checkpoints (CRITICAL)

Breaking any of these crashes training:

1. **Weight tying**: `tok_emb.weight` and `output.weight` share the same tensor. `output.weight` EXCLUDED from saved state dict. Re-tied on load.
2. **torch.compile**: `_orig_mod.` prefix stripped on save, added on load.
3. **Atomic saves**: writes to temp file, then renames.
4. **Always run `pytest tests/test_checkpoint.py`** after changes to checkpoint.py, transformer.py, or model configs.

If checkpoint tests fail, DO NOT start training.

## Code Style

- Use ruff. Line length: 100 (pyproject.toml)
- Use pytest for tests. Type hints used but not strictly enforced
- Use `from cola_coder.cli import cli` for all CLI output — never raw Rich imports

## Important Notes

- HuggingFace dataset is gated: set `HF_TOKEN` env var
- Verify GPU utilization with `nvidia-smi` during training
- Resume: `--resume checkpoints/<size>/latest` or `--auto-resume`
- wandb: `--wandb` flag (needs `pip install wandb` + `wandb login`)
- Storage config: `configs/storage.yaml` for alternate data/checkpoint paths
- **DO NOT interrupt active training runs** — checkpoint corruption loses days of GPU time

## Vision

Router model (125M) + domain specialists (50M each: React, Next.js, GraphQL, Prisma, Zod, Testing) + general TS fallback (125M). Active per request: ~175M params.

## Claude Agent Workflow

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

### Task Management
1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

### Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards
