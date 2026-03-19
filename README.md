# Cola-Coder

**A code generation transformer AI model.**

A collaboration with an Claude — part learning project, part real engineering. 

The goal: understand how modern LLMs actually work by building one from the ground up, not by reading about them.

No cloned repos. No copy-pasted model code. Every layer, every attention head, every training loop — written and documented.

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

### Training Pipeline

- **Gradient accumulation** — simulate batch sizes of 32-64 on consumer GPUs with 2-8 micro-batches
- **Gradient checkpointing** — trade ~30% speed for ~50% VRAM savings on larger configs
- **Gradient clipping** — prevent exploding gradients with norm-based clipping
- **Mixed precision autocast** — bf16 on Ada/Ampere+, fp16 with GradScaler on older cards
- **Cosine LR schedule** — linear warmup then smooth decay to 10% of peak
- **Weight tying** — embedding and output projection share weights, saving `vocab_size * dim` parameters
- **Fill-in-the-Middle (FIM)** — 50% of training examples randomly converted to FIM format for IDE-style completions

### Inference Engine

- **KV-cache** — only compute attention for new tokens, cache all previous keys/values
- **Top-k + top-p + temperature sampling** — full control over generation randomness
- **Repetition penalty** — prevent degenerate loops
- **FastAPI server** — serve your model over HTTP with auto-generated OpenAPI docs

### Reasoning Module

- **Thinking tokens** — `<think>` / `</think>` special tokens for chain-of-thought
- **GRPO (Group Relative Policy Optimization)** — generate multiple solutions, run tests, reinforce correct ones
- **Verifiable rewards** — binary reward from code execution (no reward model needed)
- **HumanEval benchmark** — 20 inline coding problems with sandboxed execution and pass@k metrics

---

## Project Structure

```
cola-coder/
├── configs/                          # YAML model & training configs
│   ├── tiny.yaml                     # 50M params — fast experiments
│   ├── small.yaml                    # 125M params — serious training
│   ├── medium.yaml                   # 350M params — best local results
│   ├── large.yaml                    # 1B+ params — cloud GPUs
│   └── reasoning.yaml                # GRPO/CoT fine-tuning config
│
├── docs/                             # Educational guides (written for TS devs)
│   ├── 01_python_for_ts_devs.md      # Python crash course
│   ├── 02_how_transformers_work.md   # Intuitive transformer guide
│   ├── 03_training_pipeline.md       # Training loop explained
│   ├── 04_reasoning_experiments.md   # CoT + GRPO guide
│   └── 05_hardware_guide.md          # VRAM budgets, cloud scaling
│
├── src/cola_coder/
│   ├── model/                        # Transformer architecture
│   │   ├── attention.py              # Grouped Query Attention + RoPE
│   │   ├── feedforward.py            # SwiGLU FFN
│   │   ├── normalization.py          # RMSNorm
│   │   ├── rope.py                   # Rotary Positional Encoding
│   │   ├── transformer.py            # Full model assembly
│   │   └── config.py                 # Model configuration
│   ├── tokenizer/                    # BPE tokenizer training & usage
│   ├── data/                         # Data pipeline (HuggingFace streaming)
│   ├── training/                     # Training loop, optimizer, checkpoints
│   ├── inference/                    # KV-cache generation, sampling, HTTP server
│   ├── evaluation/                   # HumanEval benchmark + pass@k
│   └── reasoning/                    # CoT data, thinking tokens, GRPO
│
├── scripts/                          # CLI entry points
│   ├── train_tokenizer.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── generate.py
│   ├── evaluate.py
│   ├── train_reasoning.py
│   └── serve.py
│
└── tests/                            # Unit tests for all modules
```

---

## Quick Start

```bash
# Set up the environment
make setup

# Train a BPE tokenizer on code data
make tokenizer

# Download and preprocess training data (streams from HuggingFace)
make prepare

# Train the tiny model (~2 days on RTX 4080, ~3.6 GB VRAM)
make train-tiny

# Generate code interactively
make generate

# Run HumanEval evaluation
make evaluate

# Start the HTTP inference server
make serve
```

---

## Training Data

Streams from [BigCode StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) — curated, deduplicated code from GitHub across 80+ languages. Configurable per-language filtering (default: Python, TypeScript, JavaScript).

Data is tokenized and stored as memory-mapped numpy arrays for fast random access during training. With 64GB system RAM, the entire preprocessing pipeline runs smoothly.

---

## License

MIT
