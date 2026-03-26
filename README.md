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

Full technical deep-dive: [**docs/ARCHITECTURE.md**](docs/ARCHITECTURE.md)

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
├── configs/                     # YAML configs (model, training, features, storage, reasoning, scoring)
├── docs/                        # 6 educational guides + 16 deep-dives + ARCHITECTURE.md
│   └── deep-dives/              # FIM, MoE, RoPE, torch.compile, quality, checkpoints, ...
├── src/cola_coder/
│   ├── model/                   # Transformer (GQA, SwiGLU, RMSNorm, RoPE, MoE)
│   ├── tokenizer/               # BPE tokenizer (Rust-backed)
│   ├── data/                    # Full data pipeline (FIM, quality filter, weighted dataset)
│   │   ├── filters/             # Modular filter plugins
│   │   ├── sources/             # Data sources (HF, GitHub, local)
│   │   ├── scorers/             # Quality scorers (tsc, eslint, heuristic, stars, LLM judge, classifier)
│   │   └── curation/            # Test execution scoring + Docker sandbox
│   ├── security/                # Malware scanning (YARA, Defender, ClamAV)
│   ├── training/                # Training loop, checkpoints, metrics, early stopping
│   ├── inference/               # KV-cache generator, sampling, batched generation, API server
│   ├── evaluation/              # HumanEval (62 problems), completion benchmark, pass@k
│   ├── reasoning/               # CoT, GRPO, SFT warmup, reward registry, curriculum
│   ├── features/                # 166 optional feature modules
│   └── cli.py                   # Rich CLI + questionary arrow menus
├── scripts/                     # 47 CLI entry points
└── tests/                       # 122 test files (~2,600 tests)
```

---

## Scripts (47 total)

| Category | Key Scripts |
|----------|------------|
| **Training & Data** | `menu.py` (master menu), `train.py`, `prepare_data.py`, `train_tokenizer.py`, `score_data.py`, `train_judge_classifier.py`, `train_reasoning.py` |
| **Inference** | `run.py` (REPL), `generate.py`, `serve.py` (FastAPI) |
| **Evaluation** | `evaluate.py` (HumanEval), `benchmark.py`, `ts_benchmark.py`, `regression_test.py` |
| **Analysis** | `training_dashboard.py`, `training_status.py`, `checkpoint_diff.py`, `vram_estimate.py`, `export_model.py` |

Full script reference with descriptions: [**docs/ARCHITECTURE.md** → Scripts](docs/ARCHITECTURE.md#scripts)

---

## Documentation

All documentation is organized in the [**Documentation Index**](docs/INDEX.md) — guides, deep-dives, and architecture reference, categorized by topic.

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
