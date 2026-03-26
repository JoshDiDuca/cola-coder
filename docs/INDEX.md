# Documentation Index

Everything you need to understand, use, and extend Cola-Coder — organized by what you're trying to do.

---

## Getting Started

Start here if you're new to the project or to ML in general.

| Doc | What You'll Learn | Time |
|-----|-------------------|------|
| [**Python for TypeScript Developers**](01_python_for_ts_devs.md) | Python fundamentals mapped to TypeScript concepts — types, async, classes, tooling. Written for someone who already knows TS and wants to read this codebase. | ~30 min |
| [**How Transformers Work**](02_how_transformers_work.md) | The transformer architecture from scratch — attention, embeddings, positional encoding, the forward pass. No prerequisites beyond basic linear algebra intuition. | ~45 min |
| [**Hardware Guide**](05_hardware_guide.md) | GPU specs, VRAM budgets, what fits on which card, cloud GPU options, and how to estimate training time before you start. | ~15 min |

---

## Training & Pipeline

How the training pipeline works end-to-end, and how to operate it.

| Doc | What You'll Learn | Time |
|-----|-------------------|------|
| [**Training Pipeline**](03_training_pipeline.md) | The full training loop — optimizer, learning rate scheduling, mixed precision, gradient accumulation, checkpointing. How each piece fits together. | ~40 min |
| [**Pipeline Guide**](06_pipeline_guide.md) | Using the pipeline manager — named runs, stage selection, resume, state tracking. The practical "how to run things" guide. | ~20 min |
| [**ARCHITECTURE.md**](ARCHITECTURE.md) | Full technical reference — every component, every script, data flow diagrams, security model. The single source of truth for "how does X work?" | Reference |

---

## Deep Dives

Focused technical documents that go deep on one topic. Grouped by category.

### Model Architecture

| Doc | What It Covers |
|-----|---------------|
| [**RoPE Positional Encoding**](deep-dives/rope-positional-encoding.md) | The math behind Rotary Position Embeddings — rotation matrices, theta tuning from 10K to 500K, why it generalizes to unseen sequence lengths, and how we configure it for long-context (4096+ tokens). |
| [**Mixture of Experts**](deep-dives/mixture-of-experts.md) | Sparse MoE layer design — expert routing, top-k gating, load balancing loss, capacity factor. How to get more parameters without proportional compute cost. When to use MoE vs dense. |
| [**torch.compile & CUDA Optimization**](deep-dives/torch-compile-and-cuda.md) | The GPU performance stack — torch.compile JIT fusion, Flash Attention memory/speed, TF32 Tensor Core acceleration, fused AdamW, bf16 mixed precision. How they combine for 2-4x speedup. |

### Data Pipeline

| Doc | What It Covers |
|-----|---------------|
| [**Fill-in-the-Middle Training**](deep-dives/fill-in-the-middle.md) | FIM training for IDE autocomplete — PSM and SPM formats, special tokens, line-boundary splitting, the 50/50 mix that adds +5 benchmark points. How the model learns to complete code at arbitrary cursor positions. |
| [**Data Quality Scoring Pipeline**](deep-dives/data-quality-scoring-pipeline.md) | The five-scorer composite system — tsc compiler checking, ESLint linting, GitHub stars weighting, 13-signal heuristic scorer, LLM-as-judge + distilled classifier. How scores map to training weights. Batch optimization (10k chunks, multiprocessing). |
| [**Quality Weighted Training**](deep-dives/quality-weighted-training.md) | The 13 heuristic signals in detail — length, line quality, structure, naming, comments, documentation, complexity, formatting, duplication, syntax, modernness, error handling, security. How they combine into a single score. The tier-to-weight mapping. |
| [**Data Refinement**](deep-dives/data-refinement.md) | Quality filtering strategies — conservative vs strict mode, the 15+ filter checks, rejection rates, when to use which mode. How filtering interacts with scoring. |
| [**Per-Dataset Tokenizer**](deep-dives/per-dataset-tokenizer.md) | Config-driven dataset naming (e.g., `typescript-text-math`), per-dataset tokenizer storage, auto-detection in the pipeline menu, how `data.languages` in model configs overrides `data_sources.yaml`. |
| [**Custom Data as Competitive Edge**](deep-dives/custom-data-competitive-edge.md) | Phi-1 style synthetic data generation, knowledge distillation from larger models, domain-specific data curation. How small models beat bigger ones with better data. |

### Security

| Doc | What It Covers |
|-----|---------------|
| [**Security Architecture**](deep-dives/security-architecture.md) | The full security model — SandboxedRunner (native + Docker modes), hardened tsconfig factory, credential scanner (20+ patterns), JSONL audit logging. How every tool execution on untrusted code is isolated. |
| [**Malware Scanning at Ingestion**](deep-dives/malware-scanning-ingestion.md) | YARA rules (6 categories: crypto miners, reverse shells, obfuscation, exfiltration, postinstall exploits, dangerous imports), Windows Defender integration, ClamAV daemon client. How scans run at download time. Threat response modes. |
| [**TscRunner SOLID Architecture**](deep-dives/tscrunner-solid-architecture.md) | How SOLID principles shaped the tsc execution path — why TypeCheckReward (RL) and TscScorer (data) share a single TscRunner, the hardened tsconfig invariants, batch optimization, and how to extend the pattern to other compilers. |

### Code Quality & Engineering

| Doc | What It Covers |
|-----|---------------|
| [**Shared Utilities & Helpers**](deep-dives/shared-utilities-helpers.md) | The DRY consolidation — `language_detect.py`, `ScoreMapper`, `code_hash()`, `ScoringAuditLogger`, `tsconfig_factory`. Protocol-based design. When to extract shared code and when not to. |
| [**Checkpoint Safety**](deep-dives/checkpoint-safety.md) | Safetensors format (no pickle), weight tying across embeddings, atomic saves (write-then-rename), recovery from interrupted saves. How checkpoints survive crashes. |

### Training Strategy

| Doc | What It Covers |
|-----|---------------|
| [**Multi-Agent Specialization**](deep-dives/multi-agent-specialization.md) | The router + specialist architecture — how a 125M router coordinates 50M domain experts, why 6 specialists beat one 350M generalist, the training path from base model to deployed system. |
| [**Single-Language Specialization**](deep-dives/single-language-specialization.md) | TypeScript-only training — why focusing on one language yields disproportionate quality gains, type-aware data selection, how `data.languages: ["typescript"]` flows through the pipeline. |
| [**Reasoning Experiments**](04_reasoning_experiments.md) | Chain-of-thought thinking tokens, GRPO reinforcement learning, SFT warmup, reward functions (Python exec, TypeScript compiler, combined), curriculum difficulty scaling. The DeepSeek-R1 approach adapted for code. |

---

## Quick Reference

| I want to... | Read this |
|--------------|-----------|
| Understand the codebase | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Run training for the first time | [Pipeline Guide](06_pipeline_guide.md) |
| Learn Python (coming from TS) | [Python for TS Devs](01_python_for_ts_devs.md) |
| Understand the model architecture | [How Transformers Work](02_how_transformers_work.md) + [RoPE](deep-dives/rope-positional-encoding.md) |
| Set up data quality scoring | [Scoring Pipeline](deep-dives/data-quality-scoring-pipeline.md) |
| Configure security/sandboxing | [Security Architecture](deep-dives/security-architecture.md) |
| Add a new scorer or filter | [Shared Utilities](deep-dives/shared-utilities-helpers.md) + [Scoring Pipeline](deep-dives/data-quality-scoring-pipeline.md) |
| Choose GPU / estimate VRAM | [Hardware Guide](05_hardware_guide.md) |
| Understand training metrics | [ARCHITECTURE.md → Training Screen](ARCHITECTURE.md#understanding-the-training-screen) |
| Add reasoning/GRPO | [Reasoning Experiments](04_reasoning_experiments.md) |
