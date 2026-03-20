# Skill: Training Pipeline

## 3-Stage Pipeline
1. Train tokenizer: `python scripts/train_tokenizer.py`
2. Prepare data: `python scripts/prepare_data.py --config configs/tiny.yaml --tokenizer tokenizer.json`
3. Train model: `python scripts/train.py --config configs/tiny.yaml`

## Key Files
- `src/cola_coder/training/trainer.py` — Main training loop with mixed precision
- `src/cola_coder/training/checkpoint.py` — Save/load safetensors checkpoints
- `src/cola_coder/training/optimizer.py` — AdamW + cosine scheduler
- `src/cola_coder/data/dataset.py` — CodeDataset + WeightedCodeDataset
- `src/cola_coder/data/preprocess.py` — Producer-consumer tokenization

## Configs
- `configs/tiny.yaml` — 50M params, ~4 hours on RTX 4080
- `configs/small.yaml` — 125M params, ~2 days
- `configs/medium.yaml` — 350M params, ~7 days

## Resume Training
- `--resume checkpoints/tiny/step_00005000`
- `--auto-resume` — finds latest checkpoint automatically

## Mixed Precision
- RTX 4080: bf16 (no GradScaler needed)
- RTX 3080: fp16 (needs GradScaler)

## Quality-Weighted Training
When a `.weights.npy` file exists alongside training data (created via `--score` flag
during data prep), samples are weighted by quality score automatically.
High-quality code (weight > 1.0) contributes more to the loss.

## Checkpoint Format
```
checkpoints/<size>/step_XXXXXXXX/
  model.safetensors   # model weights
  optimizer.pt        # optimizer state
  metadata.json       # step, loss, config
```
