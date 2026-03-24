# The 10-Stage Training Pipeline

A complete guide to building code generation models end-to-end with Cola-Coder.

## Overview

The Cola-Coder training pipeline transforms raw code, text, and math data into a
fully-functional code generation model through 10 sequential stages:

```
1. Collect Data        Download code, text, math from HuggingFace/GitHub
         ↓
2. Prepare Data        Filter, score, tokenize, mix into .npy training files
         ↓
3. Pretrain            Train the base transformer model from scratch
         ↓
4. Extend Context      (Optional) Apply YaRN RoPE scaling for longer context
         ↓
5. Generate Instructions   Create SFT instruction pairs from code
         ↓
6. Instruction Tune    Fine-tune on ChatML instruction data (SFT)
         ↓
7. Upcycle MoE         (Optional) Convert dense model to Mixture of Experts
         ↓
8. Train Router        Train semantic domain routing classifier
         ↓
9. Train Reasoning     GRPO reinforcement learning with <think> tokens
         ↓
10. Evaluate           Smoke test + HumanEval + quality report
```

**Stages 4 and 7 are optional** and can be skipped. All other stages are recommended
for a production-quality model.

---

## Quick Start

### Via the Menu (recommended)

```bash
.venv/Scripts/python scripts/menu.py
# → Training → Pipeline Manager → New Pipeline Run
```

### Via CLI

```bash
# Run all 10 stages
.venv/Scripts/python scripts/full_pipeline.py --config configs/small.yaml

# Skip optional stages (4, 7)
.venv/Scripts/python scripts/full_pipeline.py --config configs/small.yaml --skip-optional

# Run specific stages
.venv/Scripts/python scripts/full_pipeline.py --config configs/small.yaml --stages 3,6,9,10

# Resume from a failed stage
.venv/Scripts/python scripts/full_pipeline.py --config configs/small.yaml --start-from 6
```

---

## Stage Reference

### Stage 1: Collect Data

| | |
|---|---|
| **Purpose** | Download raw training data from HuggingFace datasets |
| **Script** | `scripts/collect_data.py` (multi-source) or `scripts/prepare_data.py` (code only) |
| **Inputs** | Model config, `configs/data_sources.yaml` |
| **Outputs** | `.npy` files in `data/processed/` |
| **Duration** | 30 min - 4 hours depending on dataset size and internet speed |

**Data Sources** (from `configs/data_sources.yaml`):

| Source | Dataset | Weight | Purpose |
|--------|---------|--------|---------|
| Code | `bigcode/the-stack-v2-dedup` | 70% | Programming languages (Python, TypeScript, JS, Java, Go, Rust) |
| Text | `HuggingFaceFW/fineweb-edu` | 20% | Educational web text for general language understanding |
| Math | `open-web-math/open-web-math` | 10% | Mathematical text for reasoning capability |

These ratios follow the Qwen2.5-Coder validated approach where 70/20/10 code/text/math
produces the best code generation results.

**Usage:**

```bash
# Collect all three sources
.venv/Scripts/python scripts/collect_data.py --config configs/small.yaml

# Code only (faster, simpler)
.venv/Scripts/python scripts/collect_data.py --config configs/small.yaml --sources code

# Quick test run (1000 samples per source)
.venv/Scripts/python scripts/collect_data.py --config configs/small.yaml --max-samples 1000
```

### Stage 2: Prepare Data

| | |
|---|---|
| **Purpose** | Filter, score, tokenize, and chunk raw data into training-ready .npy files |
| **Script** | `scripts/prepare_data.py` |
| **Inputs** | Model config, `tokenizer.json` |
| **Outputs** | `data/processed/train_data.npy` (+ `weights.npy` if scored) |
| **Duration** | 10 min - 2 hours |

**Key flags:**
- `--score` — Compute quality scores, save as `weights.npy` sidecar (recommended)
- `--filter-strict` — Stricter quality filtering
- `--workers N` — Parallel filter workers

**Only re-prepare if** the tokenizer, seq_len, dataset, languages, or filter mode changes.

### Stage 3: Pretrain

| | |
|---|---|
| **Purpose** | Train the base transformer model from scratch |
| **Script** | `scripts/train.py` |
| **Inputs** | Model config, `data/processed/train_data.npy` |
| **Outputs** | Checkpoints in `checkpoints/{model_size}/step_*` |
| **Duration** | 4 hours (tiny) to 10+ days (4080_max) |

**Target loss by model size:**

| Config | Params | Target Loss | Target Perplexity |
|--------|--------|-------------|-------------------|
| tiny | 50M | 2.5 - 3.0 | 12 - 20 |
| small | 125M | 2.0 - 2.5 | 8 - 12 |
| medium | 299M | 1.8 - 2.2 | 6 - 9 |
| 4080_max | 455M | 1.3 - 1.8 | 4 - 6 |

**Resume training:** `--auto-resume` auto-detects the latest checkpoint.

### Stage 4: Extend Context (Optional)

| | |
|---|---|
| **Purpose** | Enable longer context via YaRN RoPE positional scaling |
| **Script** | `scripts/train.py` (with rope_scaling in config) |
| **Inputs** | Pre-trained checkpoint, config with `rope_scaling` section |
| **Outputs** | Updated checkpoint adapted to longer sequences |
| **Duration** | ~1000-2000 steps (~1-2 hours) |

**How it works:**
YaRN (Yet another RoPE extensioN) applies frequency-domain scaling to RoPE
embeddings, enabling the model to handle sequences 2x-8x longer than training.
A short fine-tune adapts the model to the new position encoding.

**Config example:**
```yaml
model:
  max_seq_len: 4096
  rope_scaling:
    type: yarn
    factor: 4.0               # 4K → 16K context
    original_max_seq_len: 4096
```

**Auto-detection:** The pipeline checks `config.model.rope_scaling.type`. If set to
`"none"` (default), this stage is automatically skipped.

### Stage 5: Generate Instructions

| | |
|---|---|
| **Purpose** | Create instruction-response pairs from raw code for SFT |
| **Script** | `scripts/generate_instructions.py` |
| **Inputs** | Code data (HuggingFace or local) |
| **Outputs** | `data/sft/instructions.jsonl` |
| **Duration** | 10-30 minutes |

**Three generation modes:**
1. **Template** (fast) — Regex-based function/class extraction
2. **LLM** (high quality) — Uses an external LLM to generate instructions
3. **Self-instruct** (bootstrap) — The model generates its own instruction pairs

Output format is ChatML:
```json
{"messages": [
  {"role": "user", "content": "Write a function that..."},
  {"role": "assistant", "content": "def solution():..."}
]}
```

### Stage 6: Instruction Tune (SFT)

| | |
|---|---|
| **Purpose** | Fine-tune on instruction data to make the model follow instructions |
| **Script** | `scripts/train_sft.py` |
| **Inputs** | Pre-trained checkpoint, instruction data (`.jsonl`) |
| **Outputs** | SFT checkpoint in `checkpoints/{model}_sft/` |
| **Duration** | 2-8 hours (2-3 epochs) |

**Recommended hyperparameters** (from DeepSeek / Qwen research):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 2e-5 | Standard SFT LR, prevents catastrophic forgetting |
| Epochs | 2-3 | Sufficient for instruction following without overfitting |
| Batch size | 4-8 | Fits in 16GB VRAM with gradient accumulation |
| Warmup | 10% of steps | Gentle start for fine-tuning |

### Stage 7: MoE Upcycling (Optional)

| | |
|---|---|
| **Purpose** | Convert a dense model into a Mixture-of-Experts architecture |
| **Script** | `scripts/upcycle_to_moe.py` |
| **Inputs** | Dense checkpoint, config with MoE settings |
| **Outputs** | MoE checkpoint in `checkpoints/moe/` |
| **Duration** | 5-30 minutes |

**What it does:** Duplicates each FFN layer into N expert copies, adds a gating
network, and initializes shared experts. The model uses top-k experts per token
at inference time, giving more capacity without proportional compute increase.

**Auto-detection:** Skipped if `config.model.moe.enabled` is False.

### Stage 8: Train Router

| | |
|---|---|
| **Purpose** | Train a lightweight domain classifier for semantic routing |
| **Script** | `scripts/train_router.py` |
| **Inputs** | Code data for domain labeling |
| **Outputs** | Router model in `checkpoints/router/` |
| **Duration** | 5-15 minutes |

**Architectures:**
- **MLP Router** (~100us inference) — Bag-of-embeddings → MLP → softmax
- **Transformer Router** (~1ms inference) — 2-layer transformer → classification

**Domains:** React, Next.js, GraphQL, Prisma, Zod, Testing, General TypeScript

### Stage 9: Train Reasoning (GRPO)

| | |
|---|---|
| **Purpose** | Improve reasoning via Group Relative Policy Optimization |
| **Script** | `scripts/train_reasoning.py` |
| **Inputs** | Base checkpoint, problem set, reward function |
| **Outputs** | Reasoning-enhanced checkpoint |
| **Duration** | 2-12 hours |

**How GRPO works** (DeepSeek-R1 approach):
1. For each problem, generate G candidate solutions
2. Execute solutions against test cases
3. Compute rewards (pass/fail + quality signals)
4. Update policy: reinforce correct solutions, penalize incorrect ones
5. Uses `<think>` / `</think>` tokens for chain-of-thought reasoning

**Reward functions:**
- `python_exec` — Execute Python, check test results
- `typescript` — Run `tsc --noEmit --strict` validation
- `combined` — Multi-signal: execution + type checking + quality heuristics

**Optional SFT warmup:** Pre-train on curated chain-of-thought examples before RL
(the `--sft-warmup` flag). Recommended for stable GRPO training.

### Stage 10: Evaluate

| | |
|---|---|
| **Purpose** | Comprehensive model evaluation |
| **Scripts** | `smoke_test.py`, `evaluate.py`, `quality_report.py` |
| **Inputs** | Final checkpoint |
| **Outputs** | Evaluation metrics and quality report |
| **Duration** | 5-30 minutes |

**Three evaluation phases:**
1. **Smoke test** — 8 quick checks (loads model, generates text, checks coherence)
2. **HumanEval** — 62 coding problems, measures pass@k (k=1,5,10)
3. **Quality report** — Comprehensive metrics: syntax validity, type correctness, token efficiency

---

## Pipeline Manager

The Pipeline Manager provides named, resumable pipeline runs with state persistence.

### Creating a Run

```
Training → Pipeline Manager → New Pipeline Run
```

1. Enter a name (e.g., `small-v1`, `4080-experiment-3`)
2. Select model config (tiny / small / medium / 4080_max / large)
3. Choose which stages to include (optional stages 4, 7 unchecked by default)
4. Optionally start running immediately

### Resuming a Run

```
Training → Pipeline Manager → Resume Pipeline Run
```

Shows all saved runs with status summaries:
```
small-v1 — 5/8 done, failed at stage 6
tiny-test — 8/8 done
4080-v2 — 3/8 done, next: stage 5
```

Options when resuming:
- **Continue** from the next pending stage
- **Re-run** a specific failed or completed stage
- **Override** a stage's input (e.g., use a different checkpoint)
- **View details** — full stage-by-stage breakdown

### Run State File

Each run is persisted to `pipeline_runs/{name}.json`:

```json
{
  "name": "small-v1",
  "config_path": "configs/small.yaml",
  "created_at": "2026-03-24T10:00:00+00:00",
  "updated_at": "2026-03-24T15:30:00+00:00",
  "stages": {
    "1": {"status": "completed", "artifact": "data/", "duration_secs": 120.5},
    "2": {"status": "completed", "artifact": "data/processed/train_data.npy"},
    "3": {"status": "completed", "artifact": "checkpoints/small/step_100000"},
    "6": {"status": "failed", "error": "FileNotFoundError: instructions.jsonl"},
    "9": {"status": "pending", "override": "/path/to/custom/checkpoint"}
  }
}
```

### Input Override

When you override a stage's input, it takes priority over the artifact chain:

```
Stage 6 input resolution:
  1. Check: stage 6 override → "/custom/checkpoint" ✓ (use this)
  2. Check: stage 5 artifact → "data/sft/instructions.jsonl"
  3. Check: stage 4 artifact → ...
  4. Fallback: auto-detect from filesystem
```

---

## Training Menu Reference

The Training menu is organized into 6 groups matching the pipeline flow:

### 1. Pipeline Manager
Create, resume, and manage named pipeline runs.

### 2. Foundation (Stage 1-2)
- **Train Tokenizer** — BPE tokenizer from scratch (32K-64K vocab)
- **Prepare Data** — Opens the Data Pipeline menu

### 3. Pre-Training (Stage 3)
- **Train Model** — Select size (tiny / small / medium / 4080_max / large)
- **Resume Training** — Auto-detect latest checkpoint and continue
- **Background Training** — Overnight training with GPU throttling

### 4. Post-Training (Stage 4-7)
- **Extend Context Window** — YaRN RoPE scaling (2x-16x)
- **Generate Instruction Data** — SelfCodeAlign instruction pair creation
- **Instruction Tuning (SFT)** — Fine-tune on ChatML data via train_sft.py
- **MoE Upcycling** — Dense → Mixture of Experts conversion

### 5. Alignment & Reasoning (Stage 8-9)
- **Train Semantic Router** — MLP or Transformer domain classifier
- **Train Reasoning (GRPO)** — RL with thinking tokens and test-based rewards
- **Self-Play Training** — Iterative generate-test-improve loop

### 6. Monitoring & Tools
- **VRAM Estimation** — Estimate GPU memory before training
- **Learning Rate Finder** — Smith's LR range test
- **Training Dashboard** — Real-time TUI with loss/LR curves
- **Auto-Eval History** — View evaluation snapshots from training

---

## FAQ

### How do I resume after a stage fails?

**Via menu:** Training → Pipeline Manager → Resume Pipeline Run → select your run →
"Continue from next pending stage"

**Via CLI:** `full_pipeline.py --config configs/small.yaml --start-from 6`

### How do I use a different dataset for a stage?

Use the Pipeline Manager's "Override a stage's input" option. Enter the path to your
alternative data file or checkpoint. The override takes priority over the artifact chain.

### How do I skip optional stages?

**Via menu:** When creating a new run, uncheck stages 4 and 7 in the stage selection.

**Via CLI:** `full_pipeline.py --config configs/small.yaml --skip-optional`

### How do I add text/math data to training?

Use `collect_data.py` instead of `prepare_data.py`:

```bash
# Download and mix code + text + math per data_sources.yaml ratios
.venv/Scripts/python scripts/collect_data.py --config configs/small.yaml
```

Or configure `configs/data_sources.yaml` to adjust the mixing ratios.

### How do I compare two pipeline runs?

Each run saves artifacts (checkpoint paths) in its state file. Use:
```bash
.venv/Scripts/python scripts/compare_models.py \
    --checkpoint-a checkpoints/small/step_100000 \
    --checkpoint-b checkpoints/small_sft/latest \
    --config configs/small.yaml
```

### What if I want to train on only TypeScript?

Edit your model config's `data.languages` section:
```yaml
data:
  languages:
    - typescript
```

Then re-run stages 1-2 to collect and prepare TypeScript-only data.
