# Cola-Coder

A from-scratch code generation transformer. Human (Josh) + Claude collaboration.
Josh is an experienced TypeScript developer learning ML — frame explanations in TS analogies where helpful.

## Quick Reference

- **Language:** Python 3.10+, PyTorch 2.2+
- **Package manager:** pip with venv (`.venv/`)
- **Install:** `python -m venv .venv && .venv/Scripts/pip install -e ".[dev,logging]"`
- **Platform:** Windows 11 (no `make` — use PowerShell scripts in parent dir `~/ai research/cola-*.ps1`)
- **Configs:** `configs/tiny.yaml` (50M), `small.yaml` (125M), `medium.yaml` (299M), `4080_max.yaml` (455M), `large.yaml` (1B+)
- **GPU:** RTX 4080 Super (16GB, bf16) and RTX 3080 (10GB, fp16+GradScaler). Use `precision: "bf16"` for 4080, `precision: "fp16"` for 3080.
- **RAM:** 64GB system

## Architecture

Decoder-only transformer: RoPE positional encoding (theta configurable up to 500K), Grouped Query Attention (GQA), SwiGLU activation, RMSNorm (pre-norm), AdamW optimizer, cosine LR with linear warmup. Same architecture as LLaMA 3 / Mistral / DeepSeek-Coder.

Optional: Mixture of Experts (MoE) layer replacing standard FFN, with learned routing and load balancing.

Checkpoints use safetensors format (not pickle). Tokenizer is HuggingFace BPE (Rust-backed).

## Project Layout

```
configs/              YAML model & training configs (+ features.yaml, storage.yaml, reasoning.yaml)
docs/                 Educational guides (01-05 + deep-dives/)
src/cola_coder/
  model/              Transformer: attention, feedforward, normalization, rope, config
  tokenizer/          BPE tokenizer training & utilities
  data/               Download, preprocess, quality filter, FIM, dataset, collator
    filters/          Quality classifier, ML-based scorer, LLM annotator
    sources/          Dataset source connectors (HuggingFace, GitHub, local)
    curation/         Data curation utilities
  training/           Trainer loop, optimizer, checkpoint, metrics, early stopping
  inference/          KV-cache generator, sampling (top-k/p/temp), batched generation, FastAPI server
  evaluation/         HumanEval (62 problems), completion benchmark, pass@k, smoke tests
  reasoning/          CoT thinking tokens, GRPO, SFT warmup, reward registry, curriculum
  features/           166 feature modules (analysis, training tools, code quality, etc.)
  cli.py              Shared CLI styling (rich + questionary arrow-key menus)
scripts/              45 CLI entry points (see Scripts list below)
tests/                122 test files (~2600 tests)
```

## Training Pipeline (3 stages + optional reasoning)

1. **Tokenizer** — `scripts/train_tokenizer.py` — downloads code, trains BPE tokenizer -> `tokenizer.json`
2. **Data prep** — `scripts/prepare_data.py` — streams from HuggingFace, quality-filters, tokenizes, chunks -> `data/processed/train_data.npy`
3. **Training** — `scripts/train.py --config configs/<size>.yaml` — the actual training loop
4. **Reasoning** (optional) — `scripts/train_reasoning.py` — SFT warmup + GRPO fine-tuning with test-based rewards

Data prep requires `--tokenizer tokenizer.json`. Use `--config configs/tiny.yaml` to read dataset/language settings. Use `--max-tokens N` to cap data size.

Data prep downloads parquet files in bulk first (cached in ~/.cache/huggingface), then processes locally at disk speed. Use `--stream` for the old slow HTTP-per-row mode. Quality filtering runs in parallel with `--workers N`. Chunks stream to a memmap file to cap RAM usage.

Key flags: `--workers N` (parallel filter processes, default: CPU cores), `--batch-size N` (files per tokenization batch, default: 256), `--stream` (slow HTTP mode), `--no-filter`, `--filter-strict`, `--score` (compute quality weights for weighted training).

Prepared data is reusable across training runs. Only re-prepare if tokenizer, seq_len, dataset, languages, or filter mode changes.

## Fill-in-the-Middle (FIM) Training

FIM teaches the model to complete code at arbitrary cursor positions (not just at the end). Critical for IDE autocomplete.

- **PSM format** (Prefix-Suffix-Middle): `<|fim_prefix|>before<|fim_suffix|>after<|fim_middle|>middle`
- **SPM format** (Suffix-Prefix-Middle): `<|fim_suffix|>after<|fim_prefix|>before<|fim_middle|>middle`
- Implementation: `src/cola_coder/data/fim.py` + `src/cola_coder/data/collator.py`
- `fim_rate`: probability of applying FIM per sample (default 0.5, optimal 0.5–0.7)
- `psm_rate`: proportion of PSM vs SPM (mixing 50/50 gives +5 pts on FIM benchmarks)
- Line-boundary aware splits to avoid breaking mid-token
- FIM data prep: `scripts/prepare_fim_data.py`

See `docs/deep-dives/fill-in-the-middle.md` for full explanation.

## Quality Scoring Pipeline

`src/cola_coder/features/code_scorer.py` provides continuous 0.0–1.0 quality scores:
- 13 weighted signals: structure, naming, docs, complexity, syntax, modernness, etc.
- Tiers: excellent (0.8+), good (0.6–0.8), average (0.4–0.6), poor (0.2–0.4), reject (<0.2)
- `score_to_weight()` converts score to training weight (0.0–2.0 range)

When `--score` is passed to `prepare_data.py`, a `.weights.npy` sidecar file is created alongside the training data. The trainer auto-detects this file and uses per-example quality weights so high-quality code contributes more to the loss.

See `docs/deep-dives/quality-weighted-training.md` for the full pipeline explanation.

## Key Training Concepts

- Loss starts ~10.4 (random), target 2.0-2.5 for small, 1.5-2.0 for medium, 1.3-1.8 for 4080_max
- Perplexity = exp(loss), target 8-15 for good code generation
- Gradient accumulation: effective_batch = batch_size * gradient_accumulation
- Gradient checkpointing: optional for medium (299M, fits without at batch=5), required for 4080_max (455M), trades ~30% speed for ~50% VRAM
- bf16 on 4080 (no GradScaler), fp16 on 3080 (needs GradScaler)
- Total tokens = effective_batch * max_seq_len * max_steps
- torch.compile: fuses GPU kernels for ~20-40% speedup (handles `_orig_mod.` prefix in checkpoints)
- TF32 matmul: free Tensor Core acceleration on Ampere+ GPUs
- Flash Attention: via `F.scaled_dot_product_attention` with `is_causal=True` — 2-3x faster, halves VRAM

See `docs/deep-dives/torch-compile-and-cuda.md` for performance stack details.

## Training Data

- Source: `bigcode/starcoderdata` (gated — needs HF_TOKEN with accepted access)
- Default languages: Python, TypeScript, JavaScript (4080_max adds Java, Go, Rust)
- Quality filter at `src/cola_coder/data/quality_filter.py`: conservative (default, ~48% rejection on raw GitHub data) or strict (~65%)
- Filter checks: length, line length, char diversity, auto-generated headers, data files, comment ratio, test dumps, syntax parsing, brace balance

## Storage Config

`configs/storage.yaml` — configures alternative storage paths for large files:
- Override default locations for datasets, checkpoints, and tokenizer output
- Useful when the project drive is low on space (e.g., redirect data to a second drive)

## Reasoning Module

Multi-stage reasoning pipeline inspired by DeepSeek-R1:

1. **Thinking tokens**: `<think>` / `</think>` for chain-of-thought (vocabulary expansion, embedding resizing)
2. **SFT warmup** (optional): supervised fine-tuning on curated CoT examples before RL (DeepSeek-R1 approach)
3. **GRPO**: Group Relative Policy Optimization — generate G solutions per problem, run tests, reinforce correct ones
4. **Reward registry**: pluggable rewards — `python_exec`, `typescript` (compiler-based), `combined` (multi-signal)
5. **Parallel generation**: batched forward pass for all G completions with KV-cache expansion
6. **Parallel rewards**: `ProcessPoolExecutor` for concurrent test execution
7. **Curriculum learning**: easy → medium → hard problem ordering with per-difficulty temperature scaling
8. **Problem set**: 62 built-in problems (easy/medium/hard) + JSONL custom problems

Config: `configs/reasoning.yaml`. Script: `scripts/train_reasoning.py`.
Flags: `--sft-warmup`, `--reward {python_exec,typescript,combined}`, `--problems {builtin,extended,all,curriculum}`

Toggle features in `configs/features.yaml`: `sft_warmup`, `typescript_rewards`, `expanded_problems`, `parallel_generation`.

## Mixture of Experts (MoE)

Optional MoE layer replaces the standard FFN in each transformer block:

- **Expert router**: learned gating network assigns tokens to top-k experts
- **Sparse activation**: only k of N experts run per token (e.g., top-2 of 8)
- **Load balancing**: auxiliary loss prevents expert collapse (all tokens going to one expert)
- **Capacity factor**: limits max tokens per expert to prevent memory spikes
- Config: `configs/features.yaml` → `moe: true`
- Implementation: `src/cola_coder/features/moe_layer.py`

See `docs/deep-dives/mixture-of-experts.md` for full explanation.

## Router & Specialist System

Domain-aware routing for the multi-agent specialization vision:

- **Router model**: small model (<5M params) classifies code into 7 domains
- **Architectures**: MLP or Transformer router
- **Training**: `scripts/train_router.py` — generates labeled data from code, trains classifier
- **Data generation**: `scripts/generate_router_data.py` — from .npy, source dirs, or synthetic
- **Specialist registry**: domain-specific fine-tuned models managed through the CLI menu

See `docs/deep-dives/multi-agent-specialization.md`.

## Checkpoints (CRITICAL)

Checkpoints use safetensors format. Key invariants — **breaking any of these will crash training**:

1. **Weight tying**: `tok_emb.weight` and `output.weight` share the same tensor. `output.weight` is EXCLUDED from saved state dict. On load, the model constructor re-ties them.
2. **torch.compile**: Wraps state dict keys with `_orig_mod.` prefix. Checkpoint save STRIPS this prefix; load ADDS it back if model is compiled. This keeps checkpoint format portable.
3. **Atomic saves**: writes to temp file, then renames — protects against corruption from power loss mid-save.
4. **Always run `pytest tests/test_checkpoint.py`** before starting training or after any changes to checkpoint.py, transformer.py, or model configs.

If checkpoint tests fail, DO NOT start training — a broken checkpoint means losing hours of GPU time.

See `docs/deep-dives/checkpoint-safety.md` for the full explanation of safetensors, weight tying, and recovery scenarios.

## RoPE Configuration

Rotary Position Embeddings encode token position via 2D rotations per dimension pair:

- **theta** (base frequency): controls position encoding wavelength spread
- Default: `rope_theta: 10000.0` (original LLaMA)
- **4080_max**: `rope_theta: 500000.0` (LLaMA-3 / Yi-Coder style — dramatically better long-context)
- Higher theta = wider frequency spread = better extrapolation beyond training sequence length
- Implementation: `src/cola_coder/model/rope.py`

See `docs/deep-dives/rope-positional-encoding.md` for the intuition, math, and tuning guidance.

## Vision: Multi-Agent Specialization

End goal is a router model (125M) + domain-specific specialists (50M each: React, Next.js, GraphQL, Prisma, Zod, Testing) + general TS fallback (125M). Each specialist trains independently in 1-2 days. Active per request: ~175M params. See `docs/deep-dives/multi-agent-specialization.md`.

## Commands

```bash
# Master menu (arrow-key navigation, questionary-based)
.venv/Scripts/python scripts/menu.py

# Windows — use .venv/Scripts/python explicitly (no make)
.venv/Scripts/python scripts/train_tokenizer.py
.venv/Scripts/python scripts/prepare_data.py --config configs/tiny.yaml --tokenizer tokenizer.json  # --workers 4 --batch-size 64
.venv/Scripts/python scripts/prepare_data.py --config configs/tiny.yaml --tokenizer tokenizer.json --score  # quality-weighted
.venv/Scripts/python scripts/train.py --config configs/4080_max.yaml
.venv/Scripts/python scripts/train.py --config configs/4080_max.yaml --auto-resume  # finds latest checkpoint automatically
.venv/Scripts/python scripts/generate.py --checkpoint checkpoints/4080_max/latest --config configs/4080_max.yaml
.venv/Scripts/python scripts/evaluate.py --checkpoint checkpoints/4080_max/latest --config configs/4080_max.yaml
.venv/Scripts/python scripts/serve.py --checkpoint checkpoints/4080_max/latest --config configs/4080_max.yaml
.venv/Scripts/python scripts/train_reasoning.py --config configs/4080_max.yaml --sft-warmup --reward combined --problems all

# Or use PowerShell wrapper scripts from ~/ai research/
# cola-tokenizer.ps1, cola-prepare.ps1, cola-train.ps1, cola-generate.ps1, etc.

# Tests & lint
.venv/Scripts/pytest tests/ -v
.venv/Scripts/pytest tests/test_checkpoint.py -v  # MUST pass before any training run
.venv/Scripts/ruff check src/ scripts/ tests/
.venv/Scripts/ruff check --fix src/ scripts/ tests/  # auto-fix
```

## Scripts (45 total)

### Training & Data
| Script | Purpose |
|--------|---------|
| `menu.py` | Master arrow-key menu for all scripts |
| `train_tokenizer.py` | Train BPE tokenizer |
| `prepare_data.py` | Download, filter, tokenize training data |
| `prepare_data_interactive.py` | Guided interactive data preparation |
| `prepare_fim_data.py` | Prepare FIM-formatted training data |
| `train.py` | Main training loop |
| `train_reasoning.py` | SFT warmup + GRPO reasoning fine-tune |
| `train_quality_classifier.py` | Train ML-based quality scorer |
| `train_router.py` | Train domain router model |
| `find_lr.py` | Learning rate range finder |
| `combine_datasets.py` | Merge multiple datasets |

### Inference & Generation
| Script | Purpose |
|--------|---------|
| `run.py` | Interactive inference REPL |
| `generate.py` | One-shot generation |
| `generate_instructions.py` | Create instruction pairs from code |
| `generate_router_data.py` | Generate router training data |
| `serve.py` | FastAPI inference server |

### Evaluation & Benchmarking
| Script | Purpose |
|--------|---------|
| `evaluate.py` | HumanEval pass@k benchmark |
| `benchmark.py` | Quick tok/s benchmark |
| `nano_benchmark.py` | Fast generation speed test |
| `inference_benchmark.py` | Detailed inference profiling (temp, seq_len, precision) |
| `smoke_test.py` | 8-check quick validation |
| `ts_benchmark.py` | TypeScript-specific benchmark |
| `regression_test.py` | Track quality across versions |
| `quality_report.py` | Auto-generate quality report |
| `compare_models.py` | Side-by-side model comparison |
| `run_eval_suite.py` | Run all evaluations in sequence |
| `test_type_reward.py` | Test GRPO reward functions |

### Analysis & Tools
| Script | Purpose |
|--------|---------|
| `training_status.py` | CPU-only training progress check (supports model selection) |
| `training_dashboard.py` | Live training metrics dashboard |
| `training_eval_history.py` | Auto-eval history over training |
| `compare_checkpoints.py` | Side-by-side checkpoint comparison |
| `checkpoint_diff.py` | Detailed checkpoint diff (params, config, loss) |
| `checkpoint_info.py` | Display checkpoint metadata and file sizes |
| `average_checkpoints.py` | Checkpoint averaging (model soups) |
| `model_card.py` | Generate HuggingFace model card |
| `vram_estimate.py` | Estimate VRAM before training |
| `export_model.py` | Export to GGUF/Ollama/quantized formats |
| `run_pipeline.py` | Orchestrated multi-stage pipeline |
| `migrate_storage.py` | Migrate data/checkpoints to new storage paths |

### Data & Environment
| Script | Purpose |
|--------|---------|
| `scrape_github.py` | Crawl GitHub repos for training data |
| `score_repos.py` | Rank repos by code quality |
| `data_stats.py` | Training data statistics |
| `tokenizer_health.py` | Tokenizer health check |
| `env_check.py` | Environment validation (Python, CUDA, deps) |
| `project_health.py` | Overall project health score |

## Code Style

- Line length: 100 (ruff config in pyproject.toml)
- Linter: ruff
- Tests: pytest (122 test files, ~2600 tests)
- Type hints used but not strictly enforced

## Hardware Estimates (RTX 4080 Super)

| Config | Params | VRAM | Throughput | Training Time |
|--------|--------|------|-----------|---------------|
| tiny   | 50M    | ~3.6 GB | ~86 tok/s | ~4 hours |
| small  | 125M   | ~6.5 GB | ~45 tok/s | ~2 days |
| medium | 299M   | ~14.1 GB | ~5K tok/s | ~14 days |
| 4080_max | 455M | ~14.1 GB | ~16 tok/s | ~10 days |
| large  | 1B+    | ~24 GB  | N/A        | cloud only |

### 4080_max Config Highlights

The `4080_max` config is tuned to squeeze every GB out of 16GB VRAM:
- **dim=1280**, 24 layers, 20 query heads / 4 KV heads (head_dim=64, GPU-optimal)
- **seq_len=4096** (2x medium — critical for real-world code files)
- **RoPE theta=500K** (50x default — LLaMA-3/Yi-Coder style long-context)
- **dropout=0.0** (DeepSeek/Qwen approach for 400M+ models)
- Gradient checkpointing required, batch=4 with grad_accum=4 (effective=16)

## Feature System (166 modules)

Features live in `src/cola_coder/features/`. All toggled via `configs/features.yaml` with the standard `FEATURE_ENABLED` / `is_enabled()` pattern.

### Key Feature Categories

| Category | Examples | Count |
|----------|----------|-------|
| **Code Analysis** | `code_scorer`, `complexity_scorer`, `code_entropy`, `import_analyzer`, `repetition_detector`, `code_smell_detector` | ~20 |
| **Training Tools** | `grad_accum_calculator`, `activation_monitor`, `gradient_noise`, `plateau_detector`, `training_anomaly_detector`, `stability_monitor` | ~20 |
| **Model Analysis** | `architecture_visualizer`, `attention_analyzer`, `pruning_analyzer`, `model_fingerprint`, `param_counter`, `vram_estimator` | ~15 |
| **Data Quality** | `data_quality_report`, `data_leakage_detector`, `code_dedup_checker`, `tokenizer_coverage`, `data_balancer` | ~12 |
| **Evaluation** | `completion_benchmark`, `benchmark_store`, `safety_checker`, `output_diversity`, `confidence_calibrator` | ~10 |
| **Inference** | `inference_profiler`, `generation_cache`, `latency_optimizer`, `model_size_estimator` | ~8 |
| **Reasoning** | `reasoning_curriculum`, `moe_layer`, `sft_warmup`, `reward_registry` | ~6 |
| **Utilities** | `prompt_templates`, `code_normalizer`, `formatting_standardizer`, `checkpoint_converter`, `cost_estimator` | ~15 |
| **Advanced ML** | `distillation_helper`, `checkpoint_merger`, `lr_range_test`, `weight_init_analyzer`, `loss_landscape` | ~12 |
| **Experiment Tracking** | `hyperparam_logger`, `experiment_comparator`, `training_summary`, `progress_estimator`, `training_log_parser` | ~10 |

Notable modules:
- `code_scorer.py` — continuous quality scoring (0.0–1.0) for training weights
- `moe_layer.py` — Mixture of Experts layer with routing and load balancing
- `ollama_improver.py` — local AI code improvement via Ollama (disabled by default)
- `safety_checker.py` — detects dangerous patterns (eval, secrets, infinite loops)
- `checkpoint_merger.py` — merge checkpoints via linear, SLERP, or task arithmetic

## Documentation

### Guides (docs/)
| Doc | Topic |
|-----|-------|
| `01_python_for_ts_devs.md` | Python fundamentals for TypeScript developers |
| `02_how_transformers_work.md` | Transformer architecture from scratch |
| `03_training_pipeline.md` | Training loop, optimizer, scheduling, mixed precision |
| `04_reasoning_experiments.md` | CoT thinking tokens, GRPO, reward functions |
| `05_hardware_guide.md` | GPU specs, VRAM budgets, cloud scaling |

### Deep Dives (docs/deep-dives/)
| Doc | Topic |
|-----|-------|
| `custom-data-competitive-edge.md` | Phi-1 style synthetic data, distillation strategies |
| `data-refinement.md` | Quality filtering, scoring, curriculum ordering |
| `multi-agent-specialization.md` | Router + specialist architecture, MoE comparison |
| `single-language-specialization.md` | TypeScript-only training, type-aware data, AST augmentation |
| `fill-in-the-middle.md` | FIM training: PSM/SPM formats, special tokens, IDE autocomplete |
| `mixture-of-experts.md` | MoE layer, expert routing, load balancing, capacity factor |
| `rope-positional-encoding.md` | RoPE math, theta tuning (10K→500K), long-context |
| `torch-compile-and-cuda.md` | torch.compile, TF32, Flash Attention, fused ops, performance stack |
| `quality-weighted-training.md` | 13-signal scorer, weighted loss, the full pipeline |
| `checkpoint-safety.md` | Safetensors, weight tying, atomic saves, recovery scenarios |

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

## Important Notes

- HuggingFace dataset is gated: set `HF_TOKEN` env var and accept terms at huggingface.co/datasets/bigcode/starcoderdata
- Always verify GPU is being used: `nvidia-smi` should show >90% utilization during training
- Resume from checkpoint: `--resume checkpoints/<size>/latest` or `--auto-resume` (finds latest automatically)
- wandb logging: `--wandb` flag on train.py (needs `pip install wandb` + `wandb login`)
- Checkpoints save model (safetensors) + optimizer state (pt) + metadata (json)
- The project is written for learning — all docs are in `docs/` and written for TS developers
- Storage config: `configs/storage.yaml` for redirecting data/checkpoint paths to alternate drives
- **DO NOT interrupt active training runs** — checkpoint corruption can lose days of GPU time
