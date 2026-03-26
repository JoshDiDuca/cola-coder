# Cola-Coder

**A code generation transformer AI model — built from scratch.**

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

**Active per request: ~175M** (router + one specialist). **Total system knowledge: 475M+.** Runs inference in ~2 GB VRAM.

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
| **MoE** (optional) | Mixture of Experts upcycling | Convert any dense checkpoint to sparse MoE — more params, same compute |
| **FIM** | Fill-in-the-Middle training (PSM + SPM) | Enables IDE autocomplete at arbitrary cursor positions |
| **Performance** | torch.compile + Flash Attention + TF32 | 2-4x combined speedup from GPU kernel optimizations |
| **Context Extension** | YaRN RoPE scaling | Extend context 2x-16x via frequency-domain positional scaling |

### Model Configurations

| Config | Parameters | Layers | Dim | Heads (Q/KV) | FFN Hidden | Max Seq | VRAM (train) |
|--------|-----------|--------|-----|-------------|-----------|---------|-------------|
| **Tiny** | ~50M | 8 | 512 | 8 / 4 | 896 | 1024 | ~3.6 GB |
| **Small** | ~125M | 12 | 768 | 12 / 4 | 1344 | 2048 | ~6.5 GB |
| **Medium** | ~299M | 18 | 1024 | 16 / 4 | 1792 | 2048 | ~8.2 GB |
| **4080 Max** | ~455M | 24 | 1280 | 20 / 4 | 3456 | 4096 | ~14.1 GB |
| **Large** | ~1B+ | 32 | 2048 | 32 / 8 | 3584 | 4096 | ~24 GB |

The **4080 Max** config is tuned to squeeze every GB from a 16GB GPU: wider model (dim=1280), double the context length (4096), RoPE theta=500K for long-range position encoding, and zero dropout (regularized by data quality instead).

Full technical deep-dive: [**docs/ARCHITECTURE.md**](docs/ARCHITECTURE.md)

---

## Key Features

- **175 optional feature modules** across 10 categories (code analysis, training tools, model analysis, eval, inference, reasoning, routing, ...)
- **5-scorer data quality pipeline** — tsc, eslint, GitHub stars, 13-signal heuristic, LLM-as-judge + distilled classifier
- **Security sandbox** for untrusted code — Docker isolation, hardened tsconfig, credential scanning, audit logging
- **Malware scanning** at ingestion — YARA rules, Windows Defender, ClamAV
- **Curriculum ordering** — train on easy data first, hard data later
- **Per-dataset tokenizers** — config-driven dataset naming, auto-detection
- **GRPO reasoning** — generate, execute, reinforce (62 built-in problems)
- **Performance stack** — torch.compile + Flash Attention + TF32 + fused ops (~2-4x combined)
- **VS Code extension** — inline completions, chat participant, code actions

---

## Menu System

The interactive CLI (`scripts/menu.py`) provides arrow-key navigation through all training and evaluation tools. The Training section is organized into **6 groups** that map directly to the pipeline stages:

```
Training
  1. Pipeline Manager              Create, resume, and manage named pipeline runs
  2. Foundation (Stage 1-2)        Train tokenizer, prepare data
  3. Pre-Training (Stage 3)        Train model, resume, background training
  4. Post-Training (Stage 4-7)     Context extension, instruction generation, SFT, MoE
  5. Alignment & Reasoning (8-9)   Semantic router, GRPO reasoning, self-play
  6. Monitoring & Tools            VRAM estimation, LR finder, dashboard, eval history
```

Other top-level menus:

- **Data** — Collect (GitHub, HuggingFace, SWH, local), Modify, Score, Inspect, Prepare
- **Evaluation** — HumanEval, benchmarks, comparisons, quality reports
- **Tools** — Tests, linting, GPU info, feature toggles, export

---

## Understanding the Training Screen

During training, you'll see output like this every 100 steps:

```
14:32:07 step   1,200 (12.0%) loss 4.2341 ppl    68.8 lr 5.87e-04  142,830 tok/s | ETA 2h 15m (16:47)
14:33:19 step   1,300 (13.0%) loss 3.8917 ppl    48.9 lr 5.94e-04  145,102 tok/s | ETA 2h 13m (16:46)
```

| Column | Example | What It Means |
|--------|---------|--------------|
| **Timestamp** | `14:32:07` | Wall clock time |
| **Step** | `step 1,200` | Current optimizer step (one weight update) |
| **Progress** | `(12.0%)` | Percentage of max_steps completed |
| **Loss** | `loss 4.2341` | Cross-entropy loss — lower is better. 🟢 <2.0 (great), 🟡 <4.0 (learning), 🔴 >6.0 (early) |
| **PPL** | `ppl 68.8` | Perplexity = e^loss. Target: 8-15 for good code generation |
| **LR** | `lr 5.87e-04` | Current learning rate (warmup → cosine decay) |
| **Throughput** | `142,830 tok/s` | Tokens processed per second. 🟢 >200K, 🟡 >50K |
| **ETA** | `ETA 2h 15m (16:47)` | Estimated remaining time and finish wall clock |

### Loss & Perplexity Targets

| Loss | Perplexity | Stage |
|------|-----------|-------|
| ~10.4 | ~33,000 | Step 0 (random) |
| 6.0 | ~403 | Learning basic syntax |
| 3.0 | ~20 | Decent code structure |
| 2.5 | ~12 | Good code generation (small model target) |
| 2.0 | ~7.4 | Very good quality (medium/4080_max target) |
| <1.8 | <6 | Excellent (watch for overfitting) |

---

## Reasoning Module

Multi-stage reasoning pipeline inspired by DeepSeek-R1:

1. **Thinking tokens** — `<think>` / `</think>` brackets for chain-of-thought reasoning
2. **SFT warmup** (optional) — supervised fine-tuning on curated reasoning examples before RL
3. **GRPO** — Group Relative Policy Optimization: generate G solutions per problem, execute tests, reinforce the correct ones
4. **Pluggable rewards** — `python_exec` (test execution), `typescript` (tsc --strict compiler), `combined` (multi-signal)
5. **Parallel generation** — batched forward pass with KV-cache expansion for efficiency
6. **Curriculum learning** — easy → medium → hard problem progression with per-difficulty temperature scaling
7. **62 built-in problems** — easy/medium/hard, plus custom JSONL problem sets

```bash
.venv/Scripts/python scripts/train_reasoning.py \
    --config configs/4080_max.yaml \
    --sft-warmup \
    --reward combined \
    --problems all
```

GRPO works by generating a group of solutions for each problem, scoring them with the reward function, then updating the model to increase probability of correct solutions relative to the group mean. No critic needed — the group itself provides the baseline.

---

## Mixture of Experts (MoE)

Optional sparse MoE layer replaces the standard FFN in each transformer block. Any trained dense checkpoint can be converted:

```bash
.venv/Scripts/python scripts/upcycle_to_moe.py \
    --config configs/4080_max.yaml \
    --checkpoint checkpoints/4080_max/latest \
    --num-experts 8 \
    --num-shared 2
```

- **Expert router** — learned gating network assigns tokens to top-k experts
- **Sparse activation** — only k of N experts compute per token (e.g., top-2 of 8)
- **Shared experts** — N shared experts always active (DeepSeek-V3 approach)
- **Load balancing** — auxiliary loss prevents expert collapse
- More total parameters without proportional compute increase

See [`docs/deep-dives/mixture-of-experts.md`](docs/deep-dives/mixture-of-experts.md).

---

## Performance Stack

| Optimization | What It Does | Speedup |
|-------------|-------------|---------|
| **torch.compile** | JIT-compiles Python to fused GPU kernels | ~20-40% |
| **Flash Attention** | Tiles attention to stay in GPU SRAM, O(n) memory | ~2-3x attention |
| **TF32 matmul** | Tensor Core acceleration on Ampere+ GPUs | ~10-15% |
| **Fused AdamW** | Single CUDA kernel for optimizer step | ~5-10% |
| **bf16 mixed precision** | Half-precision compute, fp32 optimizer state | ~2x throughput |
| **Non-blocking transfers** | Overlap CPU→GPU data movement with compute | ~5% |
| **Gradient checkpointing** | Recompute activations to save ~50% VRAM | (memory, not speed) |

See [`docs/deep-dives/torch-compile-and-cuda.md`](docs/deep-dives/torch-compile-and-cuda.md).

---

## VS Code Extension

A TypeScript extension provides AI-assisted coding directly in VS Code, powered by the local FastAPI server.

### Capabilities

- **Inline completions** — ghost text suggestions at cursor position via FIM format
- **Chat participant** — `@cola-coder` in VS Code chat for code-aware conversation
- **Code actions** — quick-fix suggestions from the model
- **Thinking blocks** — collapsible `<think>...</think>` blocks when using a reasoning model

### Setup

```bash
cd vscode-extension
npm run build
npx tsc --noEmit
npx vsce package --no-dependencies
code --install-extension cola-coder-0.1.0.vsix --force
```

Start the server with `--cors` flag before using the extension:

```bash
.venv/Scripts/python scripts/serve.py --checkpoint checkpoints/4080_max/latest --config configs/4080_max.yaml --cors
```

### Extension ↔ Server Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/v1/chat/completions` | Chat participant, streaming SSE |
| `/v1/fim` | Inline completions (fill-in-the-middle) |
| `/v1/context` | Multi-file context assembly |
| `/v1/models` | Available model list |
| `/health` | Server health check |

### Key Settings

- `cola-coder.chat.baseModelMode` — `true` for raw code completion (base model), `false` for instruction-tuned structured prompts
- `cola-coder.inline.enabled` — enable/disable ghost text completions
- `cola-coder.inline.languages` — which file types receive completions

---

## Checkpoint Safety

**Breaking any of these rules corrupts checkpoints:**

1. **Weight tying** — `tok_emb.weight` and `output.weight` share the same tensor. `output.weight` is excluded from the saved state dict and re-tied on load.
2. **torch.compile** — wraps parameter keys with `_orig_mod.` prefix. Stripped on save, added on load.
3. **Atomic saves** — writes to temp file, then renames. Prevents corruption if training crashes mid-save.
4. **Safetensors only** — never pickle. No arbitrary code execution on checkpoint load.

```bash
# Run this after ANY change to checkpoint.py, transformer.py, or model configs
.venv/Scripts/pytest tests/test_checkpoint.py -v
```

Never start training if checkpoint tests fail.

---

## Features

Cola-Coder has **175 optional feature modules** across 10 categories. Every feature follows the same pattern: a `FEATURE_ENABLED` flag and `is_enabled()` function in `configs/features.yaml`. Enable only what you need — the core training loop runs without any of them.

| Category | Examples | Count |
|----------|----------|-------|
| **Code Analysis** | complexity scorer, code entropy, import analyzer, repetition detector, code smell detector | ~20 |
| **Training Tools** | gradient norm monitor, activation monitor, plateau detector, anomaly detector, dead neuron detection | ~20 |
| **Model Analysis** | architecture visualizer, attention analyzer, pruning analyzer, model fingerprint, VRAM estimator | ~15 |
| **Data Quality** | data quality report, data leakage detector, dedup checker, tokenizer coverage, data balancer | ~12 |
| **Evaluation** | completion benchmark, benchmark store, safety checker, output diversity scorer, confidence calibrator | ~10 |
| **Inference** | inference profiler, generation cache, latency optimizer, speculative decoding | ~8 |
| **Reasoning** | reasoning curriculum, MoE layer, SFT warmup, reward registry, domain rewards | ~6 |
| **Routing & Specialists** | cascade routing, confidence routing, domain detector, hot-swap specialists | ~8 |
| **Utilities** | prompt templates, code normalizer, checkpoint converter, cost estimator, export tools | ~15 |
| **Advanced ML** | distillation helper, checkpoint merger (linear/SLERP/task arithmetic), LR range test | ~12 |

---

## Quick Start

```bash
# Set up
python -m venv .venv
.venv/Scripts/pip install -e ".[dev,logging]"

# Interactive CLI menu (recommended entry point)
.venv/Scripts/python scripts/menu.py

# Or run steps manually:
.venv/Scripts/python scripts/train_tokenizer.py
.venv/Scripts/python scripts/collect_data.py --config configs/small.yaml
.venv/Scripts/python scripts/prepare_data.py --config configs/small.yaml --score
.venv/Scripts/python scripts/train.py --config configs/small.yaml
.venv/Scripts/python scripts/evaluate.py --checkpoint checkpoints/small/latest --config configs/small.yaml
```

### Data Prep Flags

```bash
# Multi-source collection (code 70% + text 20% + math 10%)
.venv/Scripts/python scripts/collect_data.py --config configs/4080_max.yaml

# Code-only (faster, simpler)
.venv/Scripts/python scripts/collect_data.py --config configs/4080_max.yaml --sources code

# Quality-weighted training data (recommended)
.venv/Scripts/python scripts/prepare_data.py --config configs/4080_max.yaml --score

# Strict quality filtering
.venv/Scripts/python scripts/prepare_data.py --config configs/4080_max.yaml --filter-strict

# Parallel workers + larger batch
.venv/Scripts/python scripts/prepare_data.py --config configs/4080_max.yaml --workers 4 --batch-size 64
```

Prepared data is reusable across runs. Only re-prepare if you change the tokenizer, sequence length, dataset, languages, or filter mode.

---

## Project Structure

```
cola-coder/
├── configs/                      # YAML configs: model, training, data sources, features, reasoning
│   ├── tiny.yaml                 # 50M — quick experiments
│   ├── small.yaml                # 125M
│   ├── medium.yaml               # 299M
│   ├── 4080_max.yaml             # 455M — recommended for RTX 4080
│   ├── large.yaml                # 1B+ — cloud only
│   ├── data_sources.yaml         # Code 70% / text 20% / math 10% mixing ratios
│   ├── features.yaml             # 175 feature module toggles
│   ├── reasoning.yaml            # GRPO + thinking token config
│   └── storage.yaml              # Alternate data/checkpoint paths
├── docs/                         # 6 guides + 10 deep-dives
│   └── deep-dives/               # FIM, MoE, RoPE, torch.compile, quality, checkpoints, ...
├── pipeline_runs/                # Named pipeline run state files (JSON)
├── src/cola_coder/
│   ├── model/                    # Transformer: GQA, SwiGLU, RMSNorm, RoPE, MoE
│   ├── tokenizer/                # BPE tokenizer training & utilities
│   ├── data/                     # Full data pipeline (FIM, quality filter, weighted dataset)
│   │   ├── filters/              # Modular filter plugins (15+ checks)
│   │   ├── sources/              # Data sources (HuggingFace, GitHub, SWH, local, docs)
│   │   └── curation/             # Test execution scoring + Docker sandbox
│   ├── training/                 # Trainer loop, checkpoints, optimizer, metrics, early stopping
│   ├── inference/                # KV-cache generator, sampling, batched generation, FastAPI server
│   ├── evaluation/               # HumanEval (62 problems), completion benchmark, pass@k, smoke tests
│   ├── reasoning/                # CoT thinking tokens, GRPO, SFT warmup, reward registry, curriculum
│   ├── pipeline/                 # Pipeline run manager: named runs, state persistence, artifact chains
│   ├── features/                 # 175 optional feature modules
│   │   └── menus/                # Training, data, eval, tools, pipeline sub-menus
│   ├── export/                   # GGUF, Ollama, quantization export
│   ├── tools/                    # Tool registry, agent, executor
│   ├── memory/                   # Long-context memory management
│   └── cli.py                    # Rich CLI + questionary arrow menus
├── scripts/                      # 55 CLI entry points
├── tests/                        # 127 test files (~2,800 tests)
└── vscode-extension/             # TypeScript VS Code extension
    └── src/
        ├── client/               # HTTP + SSE client to FastAPI server
        ├── providers/            # InlineCompletion, ChatParticipant, CodeAction
        ├── server/               # ServerManager, HealthMonitor
        └── ui/                   # StatusBar, ThinkingRenderer
```

---

## Documentation

6 guides, 16 deep-dives, and a full architecture reference — all organized by topic in the [**Documentation Index**](docs/INDEX.md).

---

## Hardware

| Config | Params | VRAM | Throughput | Training Time |
|--------|--------|------|-----------|---------------|
| tiny | 50M | ~3.6 GB | ~86 tok/s | ~4 hours |
| small | 125M | ~6.5 GB | ~45 tok/s | ~2 days |
| medium | 299M | ~8.2 GB | ~22 tok/s | ~14 days |
| 4080_max | 455M | ~14.1 GB | ~16 tok/s | ~10 days |
| large | 1B+ | ~24 GB | N/A | cloud only |

Tested on RTX 4080 Super (16GB, bf16) and RTX 3080 (10GB, fp16 + GradScaler). The 4080_max config pushes to ~14.1 GB VRAM with gradient checkpointing, 4096 context length, and RoPE theta=500K.

---

## Key Commands

```bash
# Interactive menu (main entry point)
.venv/Scripts/python scripts/menu.py

# Full pipeline end-to-end
.venv/Scripts/python scripts/full_pipeline.py --config configs/small.yaml

# Multi-source data collection
.venv/Scripts/python scripts/collect_data.py --config configs/small.yaml

# Training
.venv/Scripts/python scripts/train.py --config configs/4080_max.yaml
.venv/Scripts/python scripts/train.py --config configs/4080_max.yaml --auto-resume

# Instruction tuning (SFT)
.venv/Scripts/python scripts/train_sft.py --data data/sft/instructions.jsonl --config configs/small.yaml --checkpoint checkpoints/small/latest --epochs 2 --lr 2e-5

# Reasoning (GRPO)
.venv/Scripts/python scripts/train_reasoning.py --config configs/4080_max.yaml --sft-warmup --reward combined --problems all

# Generation & evaluation
.venv/Scripts/python scripts/generate.py --checkpoint checkpoints/4080_max/latest --config configs/4080_max.yaml
.venv/Scripts/python scripts/evaluate.py --checkpoint checkpoints/4080_max/latest --config configs/4080_max.yaml
.venv/Scripts/python scripts/quality_report.py --checkpoint checkpoints/4080_max/latest --config configs/4080_max.yaml --eval

# Tests & lint (run before training)
.venv/Scripts/pytest tests/ -v
.venv/Scripts/pytest tests/test_checkpoint.py -v  # CRITICAL
.venv/Scripts/ruff check src/ scripts/ tests/
.venv/Scripts/ruff check --fix src/ scripts/ tests/
```

---

## Training Data

**Primary source:** [BigCode The Stack v2 Dedup](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup) — deduplicated code from GitHub across 600+ languages.

**Mixed pretraining** (Qwen2.5-Coder approach):
- 70% code: Python, TypeScript, JavaScript, Java, Go, Rust
- 20% text: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (educational web text)
- 10% math: [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) (mathematical reasoning)

The dataset is gated. Set `HF_TOKEN` in your environment and accept the terms before running data prep.

---

## Disclaimer

This project is for **educational and research purposes only**. Always respect `robots.txt`, rate limits, and applicable licenses when collecting training data. See [full disclaimer](docs/ARCHITECTURE.md#disclaimer) for details.

## License

MIT
