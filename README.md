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
| **Positional Encoding** | Rotary Position Embeddings (RoPE) | Generalizes to unseen sequence lengths, zero learned parameters |
| **Attention** | Grouped Query Attention (GQA) | 3-4x smaller KV-cache than standard MHA — critical for consumer GPU inference |
| **Activation** | SwiGLU (Sigmoid Linear Unit + Gated Linear Unit) | Outperforms GELU/ReLU in every published ablation study |
| **Normalization** | RMSNorm (pre-norm) | Simpler and faster than LayerNorm, no centering bias, equally effective |
| **Optimizer** | AdamW with cosine LR schedule + linear warmup | The battle-tested recipe from GPT-2 through LLaMA 3 |
| **Precision** | bf16 / fp16 mixed precision | Half the VRAM, 2x throughput, zero quality loss |
| **Tokenizer** | Byte-Pair Encoding (BPE) via HuggingFace Tokenizers | Rust-backed, handles any encoding, code-aware pre-tokenization |
| **Checkpoints** | Safetensors format | No arbitrary code execution on load (unlike pickle) |
| **Reasoning** | Chain-of-thought + GRPO reinforcement learning | Same approach as DeepSeek-R1 — verifiable rewards from code execution |

### Model Configurations

| Config | Parameters | Layers | Dim | Heads (Q/KV) | FFN Hidden | Max Seq | VRAM (train) |
|--------|-----------|--------|-----|-------------|-----------|---------|-------------|
| **Tiny** | ~50M | 8 | 512 | 8 / 4 | 896 | 1024 | ~3.6 GB |
| **Small** | ~125M | 12 | 768 | 12 / 4 | 1344 | 2048 | ~6.5 GB |
| **Medium** | ~350M | 24 | 1024 | 16 / 4 | 1792 | 2048 | ~8.2 GB |
| **Large** | ~1B+ | 32 | 2048 | 32 / 8 | 3584 | 4096 | ~24 GB |

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

### Quality Scoring

Beyond binary pass/fail filtering, every file receives a continuous quality score from **0.0 to 1.0** based on 13 weighted signals:

- Cyclomatic complexity, documentation density, naming conventions
- Type annotation coverage, function/class structure
- Import organization, test coverage indicators
- And more — all combined into a single float score

### Quality-Weighted Training

High-quality code contributes more to the training loss. Files scoring near 1.0 are sampled more frequently and weighted more heavily in the loss calculation. This means the model spends proportionally more training time on well-structured, well-documented code — not random GitHub noise.

### Tokenization Pipeline

A producer-consumer architecture keeps your GPU saturated:

1. Worker processes read and filter source files in parallel
2. Filtered text streams to tokenizer workers (Rust-backed BPE)
3. Tokenized chunks write directly to memory-mapped numpy arrays
4. Training reads from the mmap file with zero RAM overhead

### Optional: AI-Assisted Code Improvement

Ollama integration (optional, free/local) can rewrite low-scoring files before tokenization — fixing formatting, adding docstrings, improving naming. No API costs, runs on your GPU between training runs.

### Code Quality Classifier

A three-tier classifier for finer-grained quality decisions:
- **Heuristic** — fast rule-based scoring (always on)
- **Neural** — lightweight trained classifier for ambiguous cases
- **LLM-based** — optional Ollama pass for borderline files

---

## Features

Cola-Coder has **83 optional feature modules** across 7 categories. Every feature follows the same pattern: a `FEATURE_ENABLED` flag in the config that defaults to `False`. Enable only what you need — the core training loop runs without any of them.

### Training
Crash recovery and auto-resume, gradient norm monitoring, perplexity tracking, loss curve smoothing, learning rate finder, batch size auto-tuner, wandb/tensorboard integration.

### Generation
Token streaming, beam search, speculative decoding, nucleus/top-k/top-p sampling, repetition penalty tuning, length penalty, early stopping heuristics.

### Evaluation
HumanEval benchmark runner, pass@k metrics, checkpoint-to-checkpoint comparison, per-language breakdown, regression detection.

### Code Analysis
AST-based chunking (split files at function/class boundaries), quality scoring pipeline, import graph analysis, complexity metrics, dead code detection.

### Infrastructure
Config validation with helpful error messages, VRAM estimation before training starts, INT8/INT4 quantization for inference, model export to GGUF format, Docker sandbox for code execution.

### Routing
Multi-agent specialist routing, domain detection from prompt, load balancing across specialists, routing decision logging for router training data collection.

### UI
Master CLI menu (`scripts/menu.py`) with arrow-key navigation, real-time training dashboard, data pipeline progress display, hardware utilization monitor.

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
.venv/Scripts/python scripts/prepare_data.py --config configs/tiny.yaml --tokenizer tokenizer.json
.venv/Scripts/python scripts/train.py --config configs/tiny.yaml
.venv/Scripts/python scripts/run.py
```

### Data Prep Flags

```bash
# Parallel workers (default: CPU count), larger batch size for faster processing
.venv/Scripts/python scripts/prepare_data.py --config configs/tiny.yaml --tokenizer tokenizer.json --workers 4 --batch-size 64

# Strict quality filtering
.venv/Scripts/python scripts/prepare_data.py --config configs/tiny.yaml --tokenizer tokenizer.json --filter-strict

# Cap total tokens (useful for experiments)
.venv/Scripts/python scripts/prepare_data.py --config configs/tiny.yaml --tokenizer tokenizer.json --max-tokens 500000000
```

Prepared data is reusable across training runs. Only re-prepare if you change the tokenizer, sequence length, dataset, languages, or filter mode.

---

## Project Structure

```
cola-coder/
├── configs/                     # YAML configs (model, training, features, storage)
├── src/cola_coder/
│   ├── model/                   # Transformer (GQA, SwiGLU, RMSNorm, RoPE)
│   ├── tokenizer/               # BPE tokenizer (Rust-backed)
│   ├── data/                    # Full data pipeline
│   │   ├── filters/             # Modular filter plugins
│   │   ├── sources/             # Data sources (HF, GitHub, local)
│   │   ├── curation/            # Test execution scoring + Docker sandbox
│   │   ├── quality_filter.py    # Binary + scored filtering
│   │   ├── dataset.py           # Datasets (standard + weighted)
│   │   └── preprocess.py        # Producer-consumer tokenization
│   ├── training/                # Training loop, checkpoints, metrics
│   ├── inference/               # KV-cache generator, sampling, API server
│   ├── evaluation/              # HumanEval, pass@k metrics
│   ├── reasoning/               # CoT, GRPO, reward functions
│   ├── features/                # 83 optional feature modules
│   └── cli.py                   # Rich CLI + questionary arrow menus
├── scripts/                     # 22 CLI entry points
└── tests/                       # 437+ tests
```

---

## Training Data

Source: [BigCode StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) — curated, deduplicated code from GitHub across 80+ languages. Configurable per-language filtering (default: Python, TypeScript, JavaScript).

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

### Filter Mode Comparison

| Check | Conservative | Strict |
|-------|-------------|--------|
| Min content length | 100 chars | 200 chars |
| Max line length | 1000 chars | 500 chars |
| Auto-generated headers | Yes | Yes |
| Binary/data files | Yes | Yes |
| Comment ratio | >90% | >80% |
| Syntax validity (AST) | Yes | Yes |
| Brace balance | Yes | Yes |
| Naming conventions | No | Yes |
| Min documentation | No | Yes |
| **Rejection rate (raw GitHub)** | **~48%** | **~65%** |

---

## Hardware

| Config | Params | VRAM | Throughput | Training Time |
|--------|--------|------|-----------|---------------|
| tiny   | 50M    | ~3.6 GB | ~45k tok/s | ~4 hours |
| small  | 125M   | ~6.5 GB | ~35k tok/s | ~2 days |
| medium | 350M   | ~8.2 GB | ~22k tok/s | ~7 days |
| large  | 1B+    | ~24 GB  | N/A       | cloud only |

Tested on RTX 4080 (16GB, bf16) and RTX 3080 (10GB, fp16 + GradScaler). Medium (350M) requires gradient checkpointing on 16GB — set `gradient_checkpointing: true` in the config to trade ~30% speed for ~50% VRAM savings.

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
