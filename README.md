# Cola-Coder

**A code generation transformer AI model.**

A collaboration with Claude — part learning project, part real engineering.

The goal: understand how modern LLMs actually work by building one from the ground up, not by reading about them.

No cloned repos. No copy-pasted model code. Every layer, every attention head, every training loop — written and documented.

---

## The Vision: Multi-Agent Specialization

The end goal isn't one model that's mediocre at everything — it's a **system of specialists** that each know their domain deeply, coordinated by a router model that decomposes tasks and assembles the results.

```
User prompt → [Router Model: 125M]
                     |
       ┌─────────────┼──────────────┐
       ↓             ↓              ↓
  [React 50M]   [Prisma 50M]   [Zod 50M]
       ↓             ↓              ↓
       └─────────────┼──────────────┘
                     ↓
          [Router assembles output]
```

**Why this works:** A 350M general model spreads its capacity across every framework and pattern it's ever seen. Six 50M specialists + a 125M router gives you 475M total parameters, but each specialist dedicates 100% of its capacity to one domain. The React specialist knows hooks, patterns, and conventions that no general model under 7B learns well.

| Component | Size | Role |
|-----------|------|------|
| **Router** | 125M | Task decomposition, context extraction, output assembly |
| **React specialist** | 50M | Components, hooks, JSX, state management |
| **Next.js specialist** | 50M | App router, server components, server actions |
| **GraphQL specialist** | 50M | Schemas, resolvers, Apollo patterns |
| **Prisma/ORM specialist** | 50M | Database models, queries, migrations |
| **Zod specialist** | 50M | Validation schemas, type inference |
| **Testing specialist** | 50M | Vitest, React Testing Library, mocks |
| **General TS fallback** | 125M | Anything that doesn't match a specialist |

**Active per request: ~175M** (router + one specialist). **Total system knowledge: 475M+.** Runs inference in ~2 GB VRAM. Each specialist trains independently in 1-2 days on a consumer GPU.

The practical path: train the base model first (this repo), fine-tune specialists from it, use Claude API as the router initially, then train a local router from the collected routing decisions.

---

## Architecture

Cola-Coder uses the same architecture as the models powering LLaMA 3, Mistral, DeepSeek-Coder, and Qwen.

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

The **4080 Max** config is tuned to squeeze every GB from a 16GB GPU: wider model (dim=1280), double the context length (4096), RoPE theta=500K for long-range position encoding, and zero dropout (regularized by data quality instead). Currently untested as I'm training a small model (2 days).

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

### Understanding the Training Screen

During training, you'll see output like this every 100 steps:

```
14:32:07 step   1,200 (12.0%) loss 4.2341 ppl    68.8 lr 5.87e-04  142,830 tok/s | ETA 2h 15m (16:47)
14:33:19 step   1,300 (13.0%) loss 3.8917 ppl    48.9 lr 5.94e-04  145,102 tok/s | ETA 2h 13m (16:46)
14:34:31 step   1,400 (14.0%) loss 3.5204 ppl    33.8 lr 6.00e-04  143,567 tok/s | ETA 2h 11m (16:45)
```

Here's what every column means:

| Column | Example | What It Means |
|--------|---------|--------------|
| **Timestamp** | `14:32:07` | Wall clock time — when this log line printed |
| **Step** | `step 1,200` | Current optimizer step (one weight update). Each step processes `batch_size × grad_accum × seq_len` tokens |
| **Progress** | `(12.0%)` | Percentage of `max_steps` completed |
| **Loss** | `loss 4.2341` | **Cross-entropy loss** — how wrong the model's predictions are. Lower = better. Color-coded: 🟢 <2.0 (great), 🟡 <4.0 (learning), 🟠 <6.0 (early), 🔴 >6.0 (just started) |
| **PPL** | `ppl 68.8` | **Perplexity** = e^loss. Intuitively: "the model is choosing between 69 equally likely next tokens." Target: 8-15 for good code generation. 68.8 means it's still early |
| **LR** | `lr 5.87e-04` | Current learning rate. Ramps up during warmup, then decays via cosine schedule to `min_lr` |
| **Throughput** | `142,830 tok/s` | Tokens processed per second. Higher = faster training. Color-coded: 🟢 >200K, 🟡 >50K, 🔴 <50K |
| **ETA** | `ETA 2h 15m (16:47)` | Estimated time remaining and finish time (wall clock) |

#### Loss & Perplexity — The Key Numbers

**Loss** (cross-entropy) is the primary training metric. Think of it as a score for how surprised the model is by the correct next token:

| Loss | Perplexity | What It Means | Stage |
|------|-----------|---------------|-------|
| ~10.4 | ~33,000 | Random guessing (32K vocab) | Step 0 |
| 6.0 | ~403 | Learning basic syntax | First few hundred steps |
| 4.0 | ~55 | Knows common patterns | Early training |
| 3.0 | ~20 | Decent code structure | Mid training |
| 2.5 | ~12 | Good code generation | Target for small models |
| 2.0 | ~7.4 | Very good quality | Target for medium/4080_max |
| <1.8 | <6 | Excellent (watch for overfitting) | Late training |

**Perplexity = e^loss**. It answers: "how many tokens is the model effectively choosing between?" A perplexity of 12 means the model has narrowed 32,768 possible tokens down to ~12 plausible candidates at each position. For code, that's good — there are usually only a handful of valid next tokens.

#### What to Watch For

- **Loss going down steadily** — training is working
- **Loss plateauing** — might need more data, lower LR, or you've hit the model's capacity ceiling
- **Loss spiking up suddenly** — gradient explosion or bad data batch. The grad_clip=1.0 setting prevents most spikes, but if you see repeated spikes, reduce learning rate
- **Throughput dropping** — GPU thermal throttling, or background process stealing VRAM
- **PPL under 15** — model is generating usable code. Try `scripts/generate.py` to see what it produces

---

## Data Pipeline

The data pipeline is production-grade, built to handle raw GitHub code and produce high-quality training tokens.

### Sources

- **HuggingFace** — BigCode StarCoderData, streaming or bulk parquet download (cached locally)
- **GitHub scraping** — additional domain-specific repositories for specialist fine-tuning
- **Local files** — drop your own code into a directory and include it in training
- All sources can be combined and deduplicated into a single dataset

### Quality Filtering

Two filter modes, each running 15+ checks in parallel:

| Mode | Rejection Rate | Use Case |
|------|---------------|----------|
| **Conservative** (default) | ~48% of raw GitHub | Base model training |
| **Strict** | ~65% of raw GitHub | Fine-tuning, high-quality specialists |

Filter checks include: minimum/maximum length, line length distribution, character diversity, auto-generated file headers, binary/data file detection, comment ratio bounds, test dump detection, syntax parsing (AST), and brace balance. Filters are modular plugins — easy to add or disable.

### Quality Scoring & Weighted Training

Beyond binary pass/fail filtering, every file receives a continuous quality score from **0.0 to 1.0** based on 13 weighted signals:

- Cyclomatic complexity, documentation density, naming conventions
- Type annotation coverage, function/class structure
- Import organization, test coverage indicators
- And more — all combined into a single float score

High-quality code contributes more to the training loss via per-example weights. Files scoring near 1.0 are weighted more heavily in the loss calculation, so the model learns disproportionately from well-structured code.

See [`docs/deep-dives/quality-weighted-training.md`](docs/deep-dives/quality-weighted-training.md) for the full pipeline.

### Fill-in-the-Middle (FIM) Training

FIM teaches the model to complete code at arbitrary cursor positions — not just at the end of a file. Critical for IDE autocomplete.

- **PSM format** (Prefix-Suffix-Middle): rearranges code so the model sees what comes before AND after the cursor
- **SPM format** (Suffix-Prefix-Middle): alternative ordering that improves insert-at-cursor scenarios
- Mixing 50/50 PSM + SPM yields +5 points on FIM benchmarks vs PSM alone
- Line-boundary aware splits preserve code integrity

See [`docs/deep-dives/fill-in-the-middle.md`](docs/deep-dives/fill-in-the-middle.md) for the full explanation.

### Tokenization Pipeline

A producer-consumer architecture keeps your GPU saturated:

1. Worker processes read and filter source files in parallel
2. Filtered text streams to tokenizer workers (Rust-backed BPE)
3. Tokenized chunks write directly to memory-mapped numpy arrays
4. Training reads from the mmap file with zero RAM overhead

---

## Reasoning Module

Multi-stage reasoning pipeline inspired by DeepSeek-R1:

1. **Thinking tokens**: `<think>` / `</think>` brackets for chain-of-thought reasoning
2. **SFT warmup** (optional): supervised fine-tuning on curated reasoning examples before RL
3. **GRPO**: Group Relative Policy Optimization — generate multiple solutions per problem, execute tests, reinforce the correct ones
4. **Pluggable rewards**: `python_exec` (test execution), `typescript` (compiler-based), `combined` (multi-signal)
5. **Parallel generation**: batched forward pass with KV-cache expansion for efficiency
6. **Curriculum learning**: easy → medium → hard problem progression with per-difficulty temperature scaling
7. **62 built-in problems** across easy/medium/hard difficulty, plus custom JSONL problem sets

---

## Mixture of Experts (MoE)

Optional sparse MoE layer replaces the standard FFN in each transformer block:

- **Expert router**: learned gating network assigns tokens to top-k experts
- **Sparse activation**: only k of N experts compute per token (e.g., top-2 of 8)
- **Load balancing**: auxiliary loss prevents expert collapse
- More total parameters without proportional compute increase

See [`docs/deep-dives/mixture-of-experts.md`](docs/deep-dives/mixture-of-experts.md) for the full explanation.

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

See [`docs/deep-dives/torch-compile-and-cuda.md`](docs/deep-dives/torch-compile-and-cuda.md) for the full breakdown.

---

## Features

Cola-Coder has **166 optional feature modules** across 10 categories. Every feature follows the same pattern: a `FEATURE_ENABLED` flag and `is_enabled()` function. Enable only what you need — the core training loop runs without any of them.

| Category | Examples | Count |
|----------|----------|-------|
| **Code Analysis** | complexity scorer, code entropy, import analyzer, repetition detector, code smell detector | ~20 |
| **Training Tools** | gradient accumulation calculator, activation monitor, gradient noise estimator, plateau detector, anomaly detector | ~20 |
| **Model Analysis** | architecture visualizer, attention analyzer, pruning analyzer, model fingerprint, VRAM estimator | ~15 |
| **Data Quality** | data quality report, data leakage detector, dedup checker, tokenizer coverage, data balancer | ~12 |
| **Evaluation** | completion benchmark, benchmark store, safety checker, output diversity scorer, confidence calibrator | ~10 |
| **Inference** | inference profiler, generation cache, latency optimizer, model size estimator | ~8 |
| **Reasoning** | reasoning curriculum, MoE layer, SFT warmup, reward registry | ~6 |
| **Utilities** | prompt templates, code normalizer, formatting standardizer, checkpoint converter, cost estimator | ~15 |
| **Advanced ML** | distillation helper, checkpoint merger (linear/SLERP/task arithmetic), LR range test, loss landscape analyzer | ~12 |
| **Experiment Tracking** | hyperparameter logger, experiment comparator, training summary, progress estimator | ~10 |

---

## Quick Start

```bash
# Set up
python -m venv .venv
.venv/Scripts/pip install -e ".[dev,logging]"

# Interactive CLI menu (recommended)
.venv/Scripts/python scripts/menu.py

# Or run steps manually:
.venv/Scripts/python scripts/train_tokenizer.py
.venv/Scripts/python scripts/prepare_data.py --config configs/4080_max.yaml --tokenizer tokenizer.json --score
.venv/Scripts/python scripts/train.py --config configs/4080_max.yaml
.venv/Scripts/python scripts/run.py
```

### Data Prep Flags

```bash
# Parallel workers (default: CPU count), larger batch size for faster processing
.venv/Scripts/python scripts/prepare_data.py --config configs/4080_max.yaml --tokenizer tokenizer.json --workers 4 --batch-size 64

# Quality-weighted training data (recommended)
.venv/Scripts/python scripts/prepare_data.py --config configs/4080_max.yaml --tokenizer tokenizer.json --score

# Strict quality filtering
.venv/Scripts/python scripts/prepare_data.py --config configs/4080_max.yaml --tokenizer tokenizer.json --filter-strict

# Cap total tokens (useful for experiments)
.venv/Scripts/python scripts/prepare_data.py --config configs/tiny.yaml --tokenizer tokenizer.json --max-tokens 500000000
```

Prepared data is reusable across training runs. Only re-prepare if you change the tokenizer, sequence length, dataset, languages, or filter mode.

---

## Project Structure

```
cola-coder/
├── configs/                     # YAML configs (model, training, features, storage, reasoning)
├── docs/                        # 5 educational guides + 10 deep-dives
│   └── deep-dives/              # FIM, MoE, RoPE, torch.compile, quality, checkpoints, ...
├── src/cola_coder/
│   ├── model/                   # Transformer (GQA, SwiGLU, RMSNorm, RoPE, MoE)
│   ├── tokenizer/               # BPE tokenizer (Rust-backed)
│   ├── data/                    # Full data pipeline (FIM, quality filter, weighted dataset)
│   │   ├── filters/             # Modular filter plugins
│   │   ├── sources/             # Data sources (HF, GitHub, local)
│   │   └── curation/            # Test execution scoring + Docker sandbox
│   ├── training/                # Training loop, checkpoints, metrics, early stopping
│   ├── inference/               # KV-cache generator, sampling, batched generation, API server
│   ├── evaluation/              # HumanEval (62 problems), completion benchmark, pass@k
│   ├── reasoning/               # CoT, GRPO, SFT warmup, reward registry, curriculum
│   ├── features/                # 166 optional feature modules
│   └── cli.py                   # Rich CLI + questionary arrow menus
├── scripts/                     # 45 CLI entry points
└── tests/                       # 122 test files (~2,600 tests)
```

---

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
| `inference_benchmark.py` | Detailed inference profiling |
| `smoke_test.py` | 8-check quick validation |
| `ts_benchmark.py` | TypeScript-specific benchmark |
| `regression_test.py` | Track quality across versions |
| `quality_report.py` | Auto-generate quality report |
| `compare_models.py` | Side-by-side model comparison |
| `run_eval_suite.py` | Run all evaluations in sequence |

### Analysis & Tools
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

---

## Documentation

### Guides (docs/)
| Doc | Topic |
|-----|-------|
| [`01_python_for_ts_devs.md`](docs/01_python_for_ts_devs.md) | Python fundamentals for TypeScript developers |
| [`02_how_transformers_work.md`](docs/02_how_transformers_work.md) | Transformer architecture from scratch |
| [`03_training_pipeline.md`](docs/03_training_pipeline.md) | Training loop, optimizer, scheduling, mixed precision |
| [`04_reasoning_experiments.md`](docs/04_reasoning_experiments.md) | CoT thinking tokens, GRPO, reward functions |
| [`05_hardware_guide.md`](docs/05_hardware_guide.md) | GPU specs, VRAM budgets, cloud scaling |

### Deep Dives (docs/deep-dives/)
| Doc | Topic |
|-----|-------|
| [`fill-in-the-middle.md`](docs/deep-dives/fill-in-the-middle.md) | FIM training: PSM/SPM formats, special tokens, IDE autocomplete |
| [`mixture-of-experts.md`](docs/deep-dives/mixture-of-experts.md) | MoE layer, expert routing, load balancing, capacity factor |
| [`rope-positional-encoding.md`](docs/deep-dives/rope-positional-encoding.md) | RoPE math, theta tuning (10K→500K), long-context |
| [`torch-compile-and-cuda.md`](docs/deep-dives/torch-compile-and-cuda.md) | torch.compile, TF32, Flash Attention, fused ops |
| [`quality-weighted-training.md`](docs/deep-dives/quality-weighted-training.md) | 13-signal scorer, weighted loss pipeline |
| [`checkpoint-safety.md`](docs/deep-dives/checkpoint-safety.md) | Safetensors, weight tying, atomic saves, recovery |
| [`custom-data-competitive-edge.md`](docs/deep-dives/custom-data-competitive-edge.md) | Phi-1 style synthetic data, distillation |
| [`data-refinement.md`](docs/deep-dives/data-refinement.md) | Quality filtering, scoring, curriculum ordering |
| [`multi-agent-specialization.md`](docs/deep-dives/multi-agent-specialization.md) | Router + specialist architecture |
| [`single-language-specialization.md`](docs/deep-dives/single-language-specialization.md) | TypeScript-only training, type-aware data |

---

## Hardware

| Config | Params | VRAM | Throughput | Training Time |
|--------|--------|------|-----------|---------------|
| tiny   | 50M    | ~3.6 GB | ~86 tok/s | ~4 hours |
| small  | 125M   | ~6.5 GB | ~45 tok/s | ~2 days |
| medium | 350M   | ~8.2 GB | ~22 tok/s | ~7 days |
| 4080_max | 455M | ~14.1 GB | ~16 tok/s | ~10 days |
| large  | 1B+    | ~24 GB  | N/A       | cloud only |

Tested on RTX 4080 Super (16GB, bf16) and RTX 3080 (10GB, fp16 + GradScaler). The 4080_max config pushes to ~14.1 GB VRAM with gradient checkpointing, 4096 context, and RoPE theta=500K.

---

## Training Data

Source: [BigCode StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) — curated, deduplicated code from GitHub across 80+ languages. Configurable per-language filtering (default: Python, TypeScript, JavaScript; 4080_max adds Java, Go, Rust).

The dataset is gated. Set `HF_TOKEN` in your environment and accept the terms at huggingface.co/datasets/bigcode/starcoderdata before running data prep.

### Quality Scoring Pipeline

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

## Disclaimer

This project is for **educational and research purposes only**. When collecting training data:

- Always respect `robots.txt` and applicable rate limits
- The GitHub data collector uses the **official GitHub REST API** — not HTML scraping
- Software Heritage access follows their published API rate limits (1,200 req/hr unauthenticated, 12,000 with token)
- HuggingFace datasets are accessed through their official Python SDK
- Check and comply with all applicable licenses before using collected code for training
- Be mindful of Terms of Service for any data source you access

---

## License

MIT
