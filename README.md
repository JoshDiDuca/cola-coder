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

**Active per request: ~175M** (router + one specialist). **Total system knowledge: 475M+.** Runs inference in ~2 GB VRAM.

---

## Architecture

Same architecture as LLaMA 3, Mistral, DeepSeek-Coder, and Qwen:

| Component | Implementation | Why |
|-----------|---------------|-----|
| **Core** | Decoder-only transformer | Standard for all modern code LLMs |
| **Position** | Rotary Position Embeddings (RoPE) | Generalizes to unseen lengths, zero learned params |
| **Attention** | Grouped Query Attention (GQA) | 3-4x smaller KV-cache for consumer GPU inference |
| **Activation** | SwiGLU | Outperforms GELU/ReLU in every ablation study |
| **Norm** | RMSNorm (pre-norm) | Simpler and faster than LayerNorm |
| **Precision** | bf16/fp16 mixed | Half the VRAM, 2x throughput, zero quality loss |
| **Reasoning** | Chain-of-thought + GRPO | Same approach as DeepSeek-R1 |
| **MoE** | Mixture of Experts (optional) | More params, same compute via sparse routing |
| **FIM** | Fill-in-the-Middle (PSM + SPM) | IDE autocomplete at arbitrary cursor positions |

### Model Configurations

| Config | Params | Layers | Dim | VRAM (train) |
|--------|--------|--------|-----|-------------|
| **Tiny** | ~50M | 8 | 512 | ~3.6 GB |
| **Small** | ~125M | 12 | 768 | ~6.5 GB |
| **Medium** | ~350M | 24 | 1024 | ~8.2 GB |
| **4080 Max** | ~455M | 24 | 1280 | ~14.1 GB |
| **Large** | ~1B+ | 32 | 2048 | ~24 GB |

→ Full architecture details in [**docs/ARCHITECTURE.md**](docs/ARCHITECTURE.md)

---

## Key Features

- **166 optional feature modules** across 10 categories (code analysis, training tools, model analysis, eval, inference, reasoning, ...)
- **5-scorer data quality pipeline** — tsc, eslint, GitHub stars, 13-signal heuristic, LLM-as-judge + distilled classifier
- **Security sandbox** for untrusted code — Docker isolation, hardened tsconfig, credential scanning, audit logging
- **Malware scanning** at ingestion — YARA rules, Windows Defender, ClamAV
- **Curriculum ordering** — train on easy data first, hard data later
- **Per-dataset tokenizers** — config-driven dataset naming, auto-detection
- **GRPO reasoning** — generate, execute, reinforce (62 built-in problems)
- **Performance stack** — torch.compile + Flash Attention + TF32 + fused ops (~2-4x combined)

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

---

## Project Structure

```
cola-coder/
├── configs/                     # YAML configs (model, training, features, storage, scoring)
├── docs/                        # 6 guides + 16 deep-dives + ARCHITECTURE.md
├── src/cola_coder/
│   ├── model/                   # Transformer (GQA, SwiGLU, RMSNorm, RoPE, MoE)
│   ├── data/                    # Full data pipeline + quality scorers + security
│   ├── training/                # Training loop, checkpoints, metrics
│   ├── inference/               # KV-cache generator, sampling, API server
│   ├── evaluation/              # HumanEval, benchmarks, pass@k
│   ├── reasoning/               # CoT, GRPO, SFT warmup, rewards
│   ├── features/                # 166 optional feature modules
│   └── security/                # Malware scanning (YARA, Defender, ClamAV)
├── scripts/                     # 47 CLI entry points
└── tests/                       # 122 test files
```

---

## Scripts (47 total)

| Category | Key Scripts |
|----------|------------|
| **Training & Data** | `menu.py` (master menu), `train.py`, `prepare_data.py`, `train_tokenizer.py`, `score_data.py`, `train_judge_classifier.py`, `train_reasoning.py` |
| **Inference** | `run.py` (REPL), `generate.py`, `serve.py` (FastAPI) |
| **Evaluation** | `evaluate.py` (HumanEval), `benchmark.py`, `ts_benchmark.py`, `regression_test.py` |
| **Analysis** | `training_dashboard.py`, `training_status.py`, `checkpoint_diff.py`, `vram_estimate.py`, `export_model.py` |

→ Full script reference in [**docs/ARCHITECTURE.md**](docs/ARCHITECTURE.md#scripts)

---

## Documentation

| Doc | What You'll Learn |
|-----|-------------------|
| [**ARCHITECTURE.md**](docs/ARCHITECTURE.md) | Full technical reference — pipeline stages, data flow, security, performance, all 47 scripts |
| [Python for TS Devs](docs/01_python_for_ts_devs.md) | Python fundamentals if you're coming from TypeScript |
| [How Transformers Work](docs/02_how_transformers_work.md) | Transformer architecture from scratch |
| [Training Pipeline](docs/03_training_pipeline.md) | Training loop, optimizer, scheduling, mixed precision |
| [Reasoning Experiments](docs/04_reasoning_experiments.md) | CoT thinking tokens, GRPO, reward functions |
| [Hardware Guide](docs/05_hardware_guide.md) | GPU specs, VRAM budgets, cloud scaling |
| [Pipeline Guide](docs/06_pipeline_guide.md) | Pipeline manager, named runs, stage override |

### Deep Dives

16 focused technical documents covering: [FIM training](docs/deep-dives/fill-in-the-middle.md) · [MoE routing](docs/deep-dives/mixture-of-experts.md) · [RoPE encoding](docs/deep-dives/rope-positional-encoding.md) · [torch.compile](docs/deep-dives/torch-compile-and-cuda.md) · [Quality scoring](docs/deep-dives/data-quality-scoring-pipeline.md) · [Security architecture](docs/deep-dives/security-architecture.md) · [Malware scanning](docs/deep-dives/malware-scanning-ingestion.md) · [Checkpoint safety](docs/deep-dives/checkpoint-safety.md) · [Per-dataset tokenizer](docs/deep-dives/per-dataset-tokenizer.md) · [TscRunner SOLID](docs/deep-dives/tscrunner-solid-architecture.md) · [Shared utilities](docs/deep-dives/shared-utilities-helpers.md) · [Weighted training](docs/deep-dives/quality-weighted-training.md) · [Custom data](docs/deep-dives/custom-data-competitive-edge.md) · [Data refinement](docs/deep-dives/data-refinement.md) · [Multi-agent](docs/deep-dives/multi-agent-specialization.md) · [Single-language](docs/deep-dives/single-language-specialization.md)

---

## Hardware

| Config | Params | VRAM | Training Time |
|--------|--------|------|---------------|
| tiny   | 50M    | ~3.6 GB | ~4 hours |
| small  | 125M   | ~6.5 GB | ~2 days |
| medium | 350M   | ~8.2 GB | ~7 days |
| 4080_max | 455M | ~14.1 GB | ~10 days |
| large  | 1B+    | ~24 GB  | cloud only |

Tested on RTX 4080 Super (16GB, bf16) and RTX 3080 (10GB, fp16).

---

## Disclaimer

This project is for **educational and research purposes only**. Always respect `robots.txt`, rate limits, and applicable licenses when collecting training data. See [full disclaimer](docs/ARCHITECTURE.md#disclaimer) for details.

## License

MIT
