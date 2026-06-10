# Architecture Reference

Full technical reference for Cola-Coder. For an overview, see [README.md](../README.md).

---

## Table of Contents

- [Training Pipeline](#training-pipeline)
- [Understanding the Training Screen](#understanding-the-training-screen)
- [Data Pipeline](#data-pipeline)
- [Data Quality Scoring Pipeline](#data-quality-scoring-pipeline)
- [Malware Scanning](#malware-scanning)
- [Security Architecture](#security-architecture)
- [Reasoning Module](#reasoning-module)
- [Mixture of Experts](#mixture-of-experts)
- [Performance Stack](#performance-stack)
- [Feature Modules](#feature-modules)
- [Scripts](#scripts)
- [Training Data](#training-data)
- [Disclaimer](#disclaimer)

---

## Training Pipeline

### 4 Stages (3 core + 1 optional)

```
Stage 1: Tokenizer       Train BPE tokenizer on code corpus → tokenizer.json
              ↓
Stage 2: Data Prep        Download, filter, quality-score, tokenize → train_data.npy
              ↓
Stage 3: Training         Main training loop with mixed precision → checkpoints/
              ↓
Stage 4: Reasoning        (Optional) SFT warmup + GRPO fine-tuning → reasoning checkpoints/
```

### Architecture Components

| Component | Implementation | Why This Choice |
|-----------|---------------|-----------------|
| **Architecture** | Decoder-only transformer | The standard for all modern code LLMs |
| **Positional Encoding** | Rotary Position Embeddings (RoPE) | Generalizes to unseen sequence lengths, zero learned parameters. Theta configurable up to 500K for long-context |
| **Attention** | Grouped Query Attention (GQA) | 3-4x smaller KV-cache than standard MHA — critical for consumer GPU inference |
| **Activation** | SwiGLU (Sigmoid Linear Unit + Gated Linear Unit) | Outperforms GELU/ReLU in every published ablation study |
| **Normalization** | RMSNorm (pre-norm) | Simpler and faster than LayerNorm, no centering bias, equally effective |
| **Optimizer** | AdamW with cosine LR schedule + linear warmup | The battle-tested recipe from GPT-2 through LLaMA 3 |
| **Precision** | bf16 / fp16 mixed precision | Half the VRAM, 2x throughput, zero quality loss |
| **Tokenizer** | Byte-Pair Encoding (BPE) via HuggingFace Tokenizers | Rust-backed, handles any encoding, code-aware pre-tokenization |
| **Checkpoints** | Safetensors format | No arbitrary code execution on load (unlike pickle) |
| **Reasoning** | Chain-of-thought + GRPO reinforcement learning | Same approach as DeepSeek-R1 — verifiable rewards from code execution |
| **MoE** (optional) | Mixture of Experts layer | Sparse expert routing replaces standard FFN — more params, same compute |
| **FIM** | Fill-in-the-Middle training (PSM + SPM) | Enables IDE autocomplete at arbitrary cursor positions |
| **Performance** | torch.compile + Flash Attention + TF32 | 2-4x combined speedup from GPU kernel optimizations |

### Model Configurations

| Config | Parameters | Layers | Dim | Heads (Q/KV) | FFN Hidden | Max Seq | VRAM (train) |
|--------|-----------|--------|-----|-------------|-----------|---------|-------------|
| **Tiny** | ~50M | 8 | 512 | 8 / 4 | 896 | 1024 | ~3.6 GB |
| **Small** | ~125M | 12 | 768 | 12 / 4 | 1344 | 2048 | ~6.5 GB |
| **Medium** | ~350M | 24 | 1024 | 16 / 4 | 1792 | 2048 | ~8.2 GB |
| **4080 Max** | ~455M | 24 | 1280 | 20 / 4 | 3456 | 4096 | ~14.1 GB |
| **Large** | ~1B+ | 32 | 2048 | 32 / 8 | 3584 | 4096 | ~24 GB |

The **4080 Max** config is tuned to squeeze every GB from a 16GB GPU: wider model (dim=1280), double the context length (4096), RoPE theta=500K for long-range position encoding, and zero dropout (regularized by data quality instead).

---

## Understanding the Training Screen

During training, you'll see output like this every 100 steps:

```
14:32:07 step   1,200 (12.0%) loss 4.2341 ppl    68.8 lr 5.87e-04  142,830 tok/s | ETA 2h 15m (16:47)
14:33:19 step   1,300 (13.0%) loss 3.8917 ppl    48.9 lr 5.94e-04  145,102 tok/s | ETA 2h 13m (16:46)
14:34:31 step   1,400 (14.0%) loss 3.5204 ppl    33.8 lr 6.00e-04  143,567 tok/s | ETA 2h 11m (16:45)
```

| Column | Example | What It Means |
|--------|---------|--------------|
| **Timestamp** | `14:32:07` | Wall clock time |
| **Step** | `step 1,200` | Current optimizer step. Each processes `batch_size × grad_accum × seq_len` tokens |
| **Progress** | `(12.0%)` | Percentage of `max_steps` completed |
| **Loss** | `loss 4.2341` | **Cross-entropy loss** — lower = better. 🟢 <2.0, 🟡 <4.0, 🟠 <6.0, 🔴 >6.0 |
| **PPL** | `ppl 68.8` | **Perplexity** = e^loss. "Model is choosing between ~69 equally likely tokens." Target: 8-15 |
| **LR** | `lr 5.87e-04` | Current learning rate (ramps up then cosine decays) |
| **Throughput** | `142,830 tok/s` | Tokens/second. 🟢 >200K, 🟡 >50K, 🔴 <50K |
| **ETA** | `ETA 2h 15m` | Estimated time remaining |

### Loss & Perplexity Guide

| Loss | Perplexity | What It Means | Stage |
|------|-----------|---------------|-------|
| ~10.4 | ~33,000 | Random guessing (32K vocab) | Step 0 |
| 6.0 | ~403 | Learning basic syntax | First few hundred steps |
| 4.0 | ~55 | Knows common patterns | Early training |
| 3.0 | ~20 | Decent code structure | Mid training |
| 2.5 | ~12 | Good code generation | Target for small models |
| 2.0 | ~7.4 | Very good quality | Target for medium/4080_max |
| <1.8 | <6 | Excellent (watch for overfitting) | Late training |

### What to Watch For

- **Loss going down steadily** — training is working
- **Loss plateauing** — might need more data, lower LR, or you've hit model capacity
- **Loss spiking up** — gradient explosion or bad data. `grad_clip=1.0` prevents most spikes
- **Throughput dropping** — GPU thermal throttling or background process stealing VRAM
- **PPL under 15** — model is generating usable code. Try `scripts/generate.py`

---

## Data Pipeline

The data pipeline handles raw GitHub code → high-quality training tokens.

### Sources

- **HuggingFace** — BigCode StarCoderData, streaming or bulk parquet download (cached locally)
- **GitHub scraping** — additional domain-specific repositories for specialist fine-tuning
- **Local files** — drop your own code into a directory and include it in training

### Quality Filtering

Two filter modes, each running 15+ checks:

| Mode | Rejection Rate | Use Case |
|------|---------------|----------|
| **Conservative** (default) | ~48% of raw GitHub | Base model training |
| **Strict** | ~65% of raw GitHub | Fine-tuning, high-quality specialists |

Checks: min/max length, line length distribution, character diversity, auto-generated headers, binary detection, comment ratio, test dump detection, AST parsing, brace balance. Filters are modular plugins.

### Fill-in-the-Middle (FIM)

FIM teaches the model to complete code at arbitrary cursor positions — not just at the end of a file.

- **PSM format** (Prefix-Suffix-Middle): rearranges code so the model sees before AND after the cursor
- **SPM format** (Suffix-Prefix-Middle): alternative ordering for insert-at-cursor
- Mixing 50/50 PSM + SPM yields +5 points on FIM benchmarks vs PSM alone
- Line-boundary aware splits preserve code integrity

See [`deep-dives/fill-in-the-middle.md`](deep-dives/fill-in-the-middle.md) for details.

### Tokenization Pipeline

Producer-consumer architecture keeps your GPU saturated:

1. Worker processes read and filter source files in parallel
2. Filtered text streams to tokenizer workers (Rust-backed BPE)
3. Tokenized chunks write directly to memory-mapped numpy arrays
4. Training reads from the mmap file with zero RAM overhead

**Per-dataset tokenizers** — each dataset gets its own tokenizer stored alongside the data. The dataset name is derived from `data_sources.yaml` + model config (e.g., `typescript-text-math`), and the tokenizer lives at `<data_dir>/<dataset-name>/tokenizer.json`. The `data.languages` field in model configs overrides the default code languages.

---

## Data Quality Scoring Pipeline

Five independent scorers produce a composite quality score for every file:

| Scorer | What It Measures | Default Weight |
|--------|-----------------|----------------|
| **tsc** | TypeScript compiler errors (strict mode) | 0.30 |
| **eslint** | Lint rule violations | 0.20 |
| **heuristic** | 13 static signals (complexity, docs, naming, types, structure, imports, ...) | 0.20 |
| **stars** | GitHub star count of the source repo | 0.15 |
| **classifier** | Distilled LLM-as-Judge (TF-IDF + logistic regression) | 0.15 |

All tool-based scorers (tsc, eslint) execute inside the **SandboxedRunner** — see [Security Architecture](#security-architecture) below.

### Composite Scoring

Enabled scorers combine by weighted average into a single 0.0-1.0 score, then map to training weights:

| Tier | Score Range | Training Weight |
|------|------------|-----------------|
| Excellent | >= 0.8 | 2.0x |
| Good | >= 0.6 | 1.5x |
| Average | >= 0.4 | 1.0x |
| Poor | >= 0.2 | 0.3x |
| Reject | < 0.2 | 0.0x (excluded) |

**Outputs**: `.weights.npy` sidecar file (same length as `train_data.npy`) for weighted training loss, `.scores.jsonl` per-file details (resume-capable).

### Curriculum Ordering

Reorders training data by difficulty: `easy_to_hard`, `hard_to_easy`, `staged` (phased ramps), or `random`.

### Distilled Classifier Workflow

Use an LLM to annotate ~10k samples, then train a fast classifier for bulk scoring:

```bash
# Step 1: LLM annotates samples (Claude API or Ollama)
python scripts/train_judge_classifier.py annotate --provider ollama --model codellama --data data.jsonl

# Step 2: Train fast classifier from annotations
python scripts/train_judge_classifier.py train --annotations data/annotations.jsonl

# Step 3: Evaluate accuracy
python scripts/train_judge_classifier.py evaluate --model-dir models/quality_classifier --annotations data/annotations.jsonl
```

### Scoring CLI

```bash
python scripts/score_data.py --data code_data.npy --tokenizer tokenizer.json
python scripts/score_data.py --data code_data.npy --scorers tsc,eslint --tokenizer tokenizer.json
python scripts/score_data.py --jsonl github_scraped.jsonl
python scripts/score_data.py --data code_data.npy --tokenizer tokenizer.json --curriculum easy_to_hard
```

Configuration: `configs/scoring.yaml`. Deep dive: [`deep-dives/data-quality-scoring-pipeline.md`](deep-dives/data-quality-scoring-pipeline.md).

### Quality Scoring Flow

```
Raw source file
      │
      ▼
Binary filter (15+ checks)  ──── FAIL ──→  discard
      │ PASS
      ▼
Continuous scorer (13 signals)
      │
      ▼
Quality score: 0.0 ──────────────── 1.0
                │                    │
           low weight           high weight
                └──────────┬─────────┘
                           ▼
                    Weighted dataset
                    (loss ∝ score)
```

---

## Malware Scanning

Untrusted code is scanned for malware before it enters the training pipeline, at three points:

1. **In-stream during collection** (`collect_data.py`): every streamed HuggingFace record's *content* is scanned with `YaraScanner.scan_text()` before tokenization; matching records are dropped and counted. This is the layer that actually sees the text — the tokenized `.npy` output is opaque to pattern scanners. Toggle: `malware_scan.in_stream` (default on).
2. **On every GitHub clone** (`GitHubSource.stream()` and scrape single/import modes): the clone directory is scanned before any file extraction; on threat the repo is skipped and the clone deleted immediately. Enabled by default for *all* clone paths, including the recommended search mode.
3. **Backstop directory scan** after collection (covers any stray real files in the output dir).

| Scanner | Coverage | Availability |
|---------|----------|-------------|
| **YARA** | 6 embedded rules: crypto miners, reverse shells, obfuscation, data exfiltration, postinstall exploits, dangerous imports | Always (falls back to regex) |
| **Windows Defender** | Full AV engine via `MpCmdRun.exe` | Auto-detected on Windows |
| **ClamAV** | `clamd` daemon client | Opt-in (`clamav: true`) |

Threat response: `warn` (log and continue), `quarantine` (isolate), or `abort` (stop pipeline). Configure in `configs/scoring.yaml` under `security.malware_scan`.

**Fail-closed reporting:** a scanner that crashes or times out is recorded in `MalwareScanResult.scan_errors` and logged — a result with scan errors is *not* a verified-clean result, and callers can distinguish the two.

---

## Security Architecture

All code scoring and tool execution runs inside a security sandbox. Three modes:

| Mode | Isolation | Requirements |
|------|-----------|-------------|
| **off** | No sandbox, direct execution | None |
| **native** | Temp dir isolation, process timeout, `CREATE_NO_WINDOW` on Windows | None |
| **docker** | `--network none`, `--read-only`, `--cap-drop ALL`, `--pids-limit 64`, `--memory 512m`, `--user nobody` | Docker |

The **SandboxedRunner** is the single entry point for all tsc/eslint execution on untrusted code. The **TscRunner** handles all TypeScript compiler invocations with a hardened `tsconfig.json` that blocks compiler plugin execution (`plugins: []`, `types: []`, `typeRoots: []`).

### Running Untrusted Code Safely — The Execution Surface Map

What is and is not executed, and where:

| Surface | What runs | Isolation | Default |
|---------|-----------|-----------|---------|
| **Ingestion** (collect, scrape, HF download) | Nothing. Static analysis + content scanning only. Clones are shallow with git hooks disabled; install scripts are never run at ingest time. | n/a | Always static |
| **Quality scoring** (tsc, eslint) | Trusted *analyzer tools* over untrusted code — the untrusted code itself is never executed. | SandboxedRunner: native (temp dir + timeout) or docker (`security.mode`) | native |
| **Curation test execution** (`TestRunner`) | Scraped repos' own install + test scripts — full arbitrary code execution. | `dry_run` detects frameworks without executing; `docker` runs in the hardened DockerSandbox (cap-drop ALL, no-new-privileges, memory/cpu/pids limits); `subprocess` runs on the host and **requires `allow_host_execution=True`** | **dry_run** |
| **GRPO rewards / HumanEval** (`execute_code`) | Model-generated Python. | SandboxedRunner: native (temp dir, empty PATH, timeout — limits accidents, not attackers) or docker (`python:3.12-alpine`, network none, read-only, nobody) via `scoring.security.mode: docker` | native |
| **Inference server / extension** | Nothing — generated code is returned as text, never executed. | n/a | Always static |

To get maximum isolation everywhere, set in `configs/scoring.yaml`:

```yaml
scoring:
  security:
    mode: "docker"        # scoring + model-generated code execution in containers
    require_docker: true  # refuse to run if Docker is unavailable
```

### Additional Security Layers

- **Credential scanner** — 20+ regex patterns detect leaked secrets. Modes: `off`, `warn`, `strip` (redact), `reject` (skip file)
- **Audit logging** — every scoring operation writes to `logs/scoring_audit.jsonl`
- **Malware scanning** — YARA + optional AV at ingestion (see above)

### Key Security Files

| Module | Purpose |
|--------|---------|
| `data/scorers/sandbox.py` | SandboxedRunner (native + Docker) |
| `data/scorers/security.py` | Security manager, mode selection |
| `data/scorers/credential_scanner.py` | Secret detection and handling |
| `data/scorers/audit.py` | JSONL audit trail |
| `reasoning/rewards/tsc_runner.py` | TscRunner (hardened tsc execution) |
| `data/scorers/tsconfig_factory.py` | Hardened tsconfig generation |
| `security/yara_scanner.py` | YARA malware scanner |
| `security/defender_scanner.py` | Windows Defender integration |
| `security/clamav_scanner.py` | ClamAV daemon client |

---

## Reasoning Module

Multi-stage reasoning pipeline inspired by DeepSeek-R1:

1. **Thinking tokens**: `<think>` / `</think>` brackets for chain-of-thought reasoning
2. **SFT warmup** (optional): supervised fine-tuning on curated reasoning examples before RL
3. **GRPO**: Group Relative Policy Optimization — generate multiple solutions per problem, execute tests, reinforce the correct ones
4. **Pluggable rewards**: `python_exec` (test execution), `typescript` (compiler-based), `combined` (multi-signal)
5. **Parallel generation**: batched forward pass with KV-cache expansion for efficiency
6. **Curriculum learning**: easy → medium → hard progression with per-difficulty temperature scaling
7. **62 built-in problems** across easy/medium/hard, plus custom JSONL problem sets

---

## Mixture of Experts

Optional sparse MoE layer replaces the standard FFN in each transformer block:

- **Expert router**: learned gating network assigns tokens to top-k experts
- **Sparse activation**: only k of N experts compute per token (e.g., top-2 of 8)
- **Load balancing**: auxiliary loss prevents expert collapse
- More total parameters without proportional compute increase

Deep dive: [`deep-dives/mixture-of-experts.md`](deep-dives/mixture-of-experts.md).

---

## Performance Stack

Training performance comes from stacking multiple GPU optimizations:

| Optimization | What It Does | Speedup |
|-------------|-------------|---------|
| **torch.compile** | JIT-compiles Python to fused GPU kernels | ~20-40% |
| **Flash Attention** | Tiles attention to stay in GPU SRAM, O(n) memory | ~2-3x attention |
| **TF32 matmul** | Tensor Core acceleration on Ampere+ GPUs | ~10-15% |
| **Fused AdamW** | Single CUDA kernel for optimizer step | ~5-10% |
| **bf16 mixed precision** | Half-precision compute, fp32 optimizer state | ~2x throughput |
| **Non-blocking transfers** | Overlap CPU→GPU data movement with compute | ~5% |

Deep dive: [`deep-dives/torch-compile-and-cuda.md`](deep-dives/torch-compile-and-cuda.md).

---

## Feature Modules

Cola-Coder has **166 optional feature modules** across 10 categories. Every feature follows the same pattern: a `FEATURE_ENABLED` flag and `is_enabled()` function. Enable only what you need — the core training loop runs without any of them.

| Category | Examples | Count |
|----------|----------|-------|
| **Code Analysis** | complexity scorer, code entropy, import analyzer, repetition detector | ~20 |
| **Training Tools** | gradient accumulation calc, activation monitor, plateau detector | ~20 |
| **Model Analysis** | architecture visualizer, attention analyzer, pruning analyzer | ~15 |
| **Data Quality** | data quality report, leakage detector, dedup checker, tokenizer coverage | ~12 |
| **Evaluation** | completion benchmark, benchmark store, safety checker, diversity scorer | ~10 |
| **Inference** | inference profiler, generation cache, latency optimizer | ~8 |
| **Reasoning** | reasoning curriculum, MoE layer, SFT warmup, reward registry | ~6 |
| **Utilities** | prompt templates, code normalizer, checkpoint converter | ~15 |
| **Advanced ML** | distillation helper, checkpoint merger (linear/SLERP), LR range test | ~12 |
| **Experiment Tracking** | hyperparameter logger, experiment comparator, training summary | ~10 |

---

## Scripts

### Training & Data (13)

| Script | Purpose |
|--------|---------|
| `menu.py` | Master arrow-key menu for all scripts |
| `train_tokenizer.py` | Train BPE tokenizer |
| `prepare_data.py` | Download, filter, tokenize training data |
| `prepare_data_interactive.py` | Guided interactive data preparation |
| `prepare_fim_data.py` | Prepare FIM-formatted training data |
| `train.py` | Main training loop |
| `train_reasoning.py` | SFT warmup + GRPO reasoning fine-tune |
| `score_data.py` | Score data quality, generate `.weights.npy` |
| `train_judge_classifier.py` | LLM-as-Judge annotation + distilled classifier |
| `train_quality_classifier.py` | Train ML-based quality scorer |
| `train_router.py` | Train domain router model |
| `find_lr.py` | Learning rate range finder |
| `combine_datasets.py` | Merge multiple datasets |

### Inference & Generation (5)

| Script | Purpose |
|--------|---------|
| `run.py` | Interactive inference REPL |
| `generate.py` | One-shot generation |
| `generate_instructions.py` | Create instruction pairs from code |
| `generate_router_data.py` | Generate router training data |
| `serve.py` | FastAPI inference server |

### Evaluation & Benchmarking (10)

| Script | Purpose |
|--------|---------|
| `evaluate.py` | HumanEval pass@k benchmark |
| `benchmark.py` | Quick tok/s benchmark |
| `nano_benchmark.py` | Fast generation speed test |
| `inference_benchmark.py` | Detailed inference profiling |
| `smoke_test.py` | 8-check quick validation |
| `ts_benchmark.py` | TypeScript-specific benchmark |
| `regression_test.py` | Track quality across versions |
| `quality_report.py` | Auto-generate quality report |
| `compare_models.py` | Side-by-side model comparison |
| `run_eval_suite.py` | Run all evaluations in sequence |

### Analysis & Tools (11)

| Script | Purpose |
|--------|---------|
| `training_status.py` | CPU-only training progress check |
| `training_dashboard.py` | Live training metrics dashboard |
| `training_eval_history.py` | Auto-eval history over training |
| `checkpoint_diff.py` | Detailed checkpoint diff |
| `checkpoint_info.py` | Display checkpoint metadata |
| `average_checkpoints.py` | Checkpoint averaging (model soups) |
| `model_card.py` | Generate HuggingFace model card |
| `vram_estimate.py` | Estimate VRAM before training |
| `export_model.py` | Export to GGUF/Ollama/quantized formats |
| `env_check.py` | Environment validation |
| `project_health.py` | Overall project health score |

### Data Prep Flags

```bash
# Parallel workers, larger batch for faster processing
python scripts/prepare_data.py --config configs/4080_max.yaml --tokenizer tokenizer.json --workers 4 --batch-size 64

# Quality-weighted training data (recommended)
python scripts/prepare_data.py --config configs/4080_max.yaml --tokenizer tokenizer.json --score

# Strict quality filtering
python scripts/prepare_data.py --config configs/4080_max.yaml --tokenizer tokenizer.json --filter-strict

# Cap total tokens (experiments)
python scripts/prepare_data.py --config configs/tiny.yaml --tokenizer tokenizer.json --max-tokens 500000000
```

---

## Training Data

Source: [BigCode StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) — curated, deduplicated code from GitHub across 80+ languages. Configurable per-language filtering.

The dataset is gated. Set `HF_TOKEN` and accept terms at huggingface.co before running data prep.

---

## Disclaimer

This project is for **educational and research purposes only**. When collecting training data:

- Always respect `robots.txt` and applicable rate limits
- The GitHub data collector uses the **official GitHub REST API** — not HTML scraping
- Software Heritage access follows their published API rate limits (1,200 req/hr unauthenticated, 12,000 with token)
- HuggingFace datasets are accessed through their official Python SDK
- Check and comply with all applicable licenses before using collected code for training
- Be mindful of Terms of Service for any data source you access
