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

Decoder-only transformer (LLaMA 3 / Mistral / DeepSeek-Coder): RoPE, GQA, SwiGLU, RMSNorm, AdamW/Muon, cosine/WSD LR. Optional MoE FFN (config.model.moe.enabled; upcycled via stage 7, auto-detected on load). Safetensors checkpoints. HuggingFace byte-level BPE tokenizer with **digit splitting** (`Digits(individual_digits=True)` — one token per digit, LLaMA-3/Qwen style, for numeric handling; affects freshly-trained tokenizers only). YaRN context extension (optional).

## Project Layout

```
configs/              YAML model & training configs (+ features.yaml, storage.yaml, reasoning.yaml, data_sources.yaml)
pipeline_runs/        Named pipeline run state files — pipeline_runs/{name}.json
src/cola_coder/
  model/              Transformer: attention, feedforward, normalization, rope, config
  tokenizer/          BPE tokenizer training & utilities
  data/               Download, preprocess, quality filter, FIM, dataset, collator
    filters/          Modular filter plugins (15+ checks)
    sources/          Data sources (HuggingFace, GitHub, SWH, local, docs, mixed)
    curation/         Test execution scoring + Docker sandbox
  training/           Trainer loop, optimizer, checkpoint, metrics, early stopping, SFT
  inference/          KV-cache generator, sampling, batched generation, best-of-N verification, FastAPI server
  evaluation/         HumanEval (62 problems), completion benchmark, pass@k, smoke tests
  reasoning/          CoT thinking tokens, GRPO, SFT warmup, reward registry, curriculum
  pipeline/           Pipeline run manager: named runs, state persistence, artifact chains
  features/           175 feature modules — all toggled via configs/features.yaml
    menus/            Sub-menu modules: data, training, eval, tools, pipeline
  export/             GGUF, Ollama, quantization export
  tools/              Tool registry, agent executor
  memory/             Long-context memory management
  cli.py              Shared CLI styling (rich + questionary arrow-key menus, multi_select, weight_editor)
scripts/              62 CLI entry points — all use `from cola_coder.cli import cli`, never direct Rich
tests/                150 test files (~3350 tests)
docs/                 Educational guides (01-06) + deep-dives/
vscode-extension/     TypeScript VS Code extension (see below)
```

## VS Code Extension

### Layout
```
vscode-extension/
  src/client/           HTTP + SSE client to FastAPI server (ColaCoderClient)
  src/providers/        InlineCompletion, ChatParticipant, CodeAction, LanguageModel
  src/server/           ServerManager (process lifecycle) + HealthMonitor
  src/ui/               StatusBar + ThinkingRenderer (collapsible <think> blocks)
  src/utils/            config, logger
  src/context/          ContextAssembler, FimFormatter
  src/extension.ts      Activation, provider registration, commands
  package.json          Extension manifest (chatParticipants, commands, settings)
```

### Key Commands
```bash
cd vscode-extension
npm run build                          # esbuild bundle (~47KB)
npx tsc --noEmit                       # type-check (must pass before packaging)
npx vsce package --no-dependencies     # create .vsix
code --install-extension cola-coder-0.1.0.vsix --force  # install locally
```

### Extension ↔ Server Contract
- Extension talks to FastAPI server via HTTP (default: `http://localhost:8000`)
- Server must be started with `--cors` flag for the extension to connect
- Endpoints: `/v1/chat/completions` (SSE streaming), `/v1/fim`, `/v1/context`, `/v1/models`, `/health`
- FIM endpoint → inline completions (ghost text)
- Chat endpoint → chat participant + language model provider
- Default mode is `external` (user starts server manually)

### Important
- Always `npm run build` after TypeScript changes — extension runs from `dist/extension.js`
- Always `npx tsc --noEmit` before packaging to catch type errors
- Extension registers ALL providers on activation even when server is disconnected
- Chat participant has `baseModelMode` setting: `true` = raw code completion (base model), `false` = structured prompts (instruction-tuned model)
- Inline completions require server connection — check output channel for "server not connected" logs

## Training Pipeline

The full training pipeline is **10 stages**. Use the Pipeline Manager or `full_pipeline.py` to run it end-to-end.

1. **Collect Data** — `scripts/collect_data.py` → multi-source (code 70% + text 20% + math 10%)
2. **Prepare Data** — `scripts/prepare_data.py --config --score` → `data/processed/train_data.npy`
3. **Pretrain** — `scripts/train.py --config` → `checkpoints/{size}/`
4. **Extend Context** (optional) — `scripts/train.py --config` with `rope_scaling` in config
5. **Generate Instructions** — `scripts/generate_instructions.py` → `data/sft/instructions.jsonl`
6. **Instruction Tune** — `scripts/train_sft.py --data --config --checkpoint` → `checkpoints/{size}_sft/`
7. **Upcycle MoE** (optional) — `scripts/upcycle_to_moe.py` → `checkpoints/moe/`
8. **Train Router** — `scripts/train_router.py --arch mlp` → `checkpoints/router/`
9. **Train Reasoning** — `scripts/train_reasoning.py --sft-warmup --reward combined` → reasoning checkpoint
10. **Evaluate** — `scripts/smoke_test.py` + `scripts/evaluate.py` + `scripts/quality_report.py`

Re-prepare data only if tokenizer, seq_len, dataset, languages, or filter mode changes.

**Fill-in-the-Middle (FIM):** two options — (a) prep-time, `scripts/prepare_fim_data.py`
(static, baked into the .npy); or (b) dynamic train-time, set `data.fim_rate` (e.g. 0.1)
in the config — the dataloader rearranges that fraction of each batch into FIM on the fly
(StarCoder2-style, different splits each epoch). Dynamic FIM needs the tokenizer's
`<|fim_*|>` tokens; it auto-disables with a warning if they're absent. Default `fim_rate: 0.0` (off).

### Multi-Source Data (Qwen2.5-Coder ratios)
```bash
# Collect code + text + math with 70/20/10 weights
.venv/Scripts/python scripts/collect_data.py --config configs/small.yaml

# Sources defined in configs/data_sources.yaml
# Code: bigcode/the-stack-v2-dedup (70%)
# Text: HuggingFaceFW/fineweb-edu (20%)
# Math: open-web-math/open-web-math (10%)
```

## Key Commands

```bash
.venv/Scripts/python scripts/menu.py                           # Master menu
.venv/Scripts/python scripts/full_pipeline.py --config configs/small.yaml  # Full 10-stage pipeline
.venv/Scripts/python scripts/collect_data.py --config configs/4080_max.yaml  # Multi-source data
.venv/Scripts/python scripts/train.py --config configs/4080_max.yaml
.venv/Scripts/python scripts/train.py --config configs/4080_max.yaml --auto-resume
.venv/Scripts/python scripts/train_sft.py --data data/sft/instructions.jsonl --config configs/4080_max.yaml --checkpoint checkpoints/4080_max/latest --epochs 2 --lr 2e-5
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
- CLI methods: `cli.header()`, `cli.choose()`, `cli.confirm()`, `cli.kv_table()`, `cli.multi_select()`, `cli.weight_editor()`, `cli.pick_languages()`, `cli.info()`, `cli.success()`, `cli.error()`, `cli.warn()`, `cli.dim()`, `cli.done()`, `cli.print()`, `cli.step()`, `cli.rule()`
- Every new user-facing feature or script MUST have a menu entry. No orphan scripts.
- Use `cli.pick_languages()` for any language selection — never inline language loops.
- New feature modules must be in a `_FEATURE_CATEGORIES` group, not left in "Other".

## Menu Architecture

Master menu is split into sub-modules for maintainability:

```
src/cola_coder/features/
  master_menu.py              # Thin coordinator — shared helpers, generate, router
  menus/
    __init__.py               # Exports: DataMenu, TrainingMenu, EvalMenu, ToolsMenu, PipelineMenu
    data_menu.py              # 5 grouped sub-menus: Collect, Modify, Score, Inspect, Prepare
    training_menu.py          # 6 sub-menus: Pipeline Manager, Foundation, Pre-Training,
                              #   Post-Training, Alignment & Reasoning, Monitoring & Tools
    eval_menu.py              # HumanEval, benchmarks, comparisons, quality reports
    tools_menu.py             # Tests, linting, GPU, features, settings, export
    pipeline_menu.py          # Named pipeline runs, resume, stage override, state tracking
```

Sub-menus accept `master: MasterMenu` in constructor and call `self._master._run_script()` etc.
`PipelineMenu` uses `_run_stage_script()` (its own helper) — raises on non-zero exit so stages fail correctly.
Feature scanning functions stay in master_menu.py — ToolsMenu imports them from there.

### Training Menu Structure (6 groups)
```
1. Pipeline Manager     → pipeline_menu.py (create/resume/view/override named runs)
2. Foundation (1-2)     → Train tokenizer, Prepare data
3. Pre-Training (3)     → Train model, Resume, Background training
4. Post-Training (4-7)  → Extend context, Generate instructions, Instruction tuning, MoE upcycle, MoE fine-tune (7.5)
5. Alignment (8-9)      → Train semantic router, GRPO reasoning, Self-play
6. Monitoring           → VRAM estimation, LR finder, dashboard, eval history
```

### Pipeline Manager
- Creates named runs persisted to `pipeline_runs/{name}.json`
- 10-stage state machine: pending → running → completed / failed / skipped
- Artifact chain resolution: stage override → previous artifact → filesystem auto-detect
- `PipelineRunManager` in `src/cola_coder/pipeline/run_manager.py`
- **Full Auto Pipeline** (first option; also a top-level master-menu entry):
  detects hardware via `features/hardware_profiler.py`, recommends the largest
  config that fits VRAM (estimator-validated), writes a derived config to
  `configs/auto/`, creates a named run (`auto-{config}[-smoke]`) and executes it.
  Modes: Smoke (30 steps, skips stage 9, isolated `_smoke` checkpoint dirs),
  Full, Dry-run (delegates to `scripts/auto_pipeline.py --dry-run`)

### Data Sources
All data sources (GitHub, HuggingFace, SWH, Local) emit `pipeline.DataRecord(content=..., metadata={...})`. The `metadata` dict carries source-specific fields (e.g. `"source": "github"`, `"repo_name"`, `"file_path"`). Access via `record.metadata.get("field_name", "")`.

### Instruction Tuning (Stage 6)
- Script: `train_sft.py` (NOT `train.py`)
- Args: `--data`, `--config`, `--checkpoint`, `--epochs`, `--lr`
- Saves to: `checkpoints/{config_stem}_sft/`
- LR: 2e-5, Epochs: 2-3

### Context Extension (Stage 4)
- Uses `RoPEScalingConfig` in model config: `type` (default "none"), `factor` (default 1.0)
- Auto-skipped when `type == "none"` or `factor <= 1.0`
- Runs `train.py --auto-resume` with the yarn scaling config applied

## Important Notes

- HuggingFace dataset is gated: set `HF_TOKEN` env var
- Tokenizer path from `configs/storage.yaml` — pipeline scripts must pass `--tokenizer` from storage config
- Verify GPU utilization with `nvidia-smi` during training
- Resume: `--resume checkpoints/<size>/latest` or `--auto-resume`
- wandb: `--wandb` flag (needs `pip install wandb` + `wandb login`)
- Storage config: `configs/storage.yaml` for alternate data/checkpoint paths
- `_run_script()` in master_menu does NOT raise on non-zero exit (prints error, continues)
- `_run_stage_script()` in pipeline_menu DOES raise — required for pipeline stage failure tracking
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
