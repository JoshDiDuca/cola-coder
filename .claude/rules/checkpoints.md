---
match: "**/checkpoint*.py,**/transformer.py,**/config.py,configs/*.yaml"
---

# Checkpoint Safety Rules

- Run `pytest tests/test_checkpoint.py` after ANY change to checkpoint.py, transformer.py, or model configs
- Never break weight tying: `tok_emb.weight` and `output.weight` share the same tensor
- `output.weight` is EXCLUDED from saved state dict — re-tied on load by constructor
- torch.compile wraps keys with `_orig_mod.` — strip on save, add on load
- Checkpoints use safetensors format, never pickle
- Saves are atomic: write to temp file, then rename
- Never interrupt an active training run — checkpoint corruption loses days of GPU time
- Loads go through `_load_state_dict_tied(model, state_dict)` — strict validation
  where ONLY the tied `output.weight` may be missing. Never call
  `load_state_dict(strict=False)` directly: it silently ignores every mismatch

## Vocab Expansion After Reasoning Training
Reasoning training calls `_resize_embeddings` to add thinking tokens (`<think>`/`</think>`),
expanding vocab from e.g. 32768 → 32770. The expanded `tok_emb.weight` is saved to disk.
Any subsequent load (Stage 10 smoke test, inference, SFT continuation) rebuilds the model
from config — which still says `vocab_size=32768` — causing a hard size mismatch error.

**Fixed in `checkpoint.py`:** `_maybe_resize_vocab(model, state_dict)` is called in both
`load_checkpoint` and `load_model_only` before `load_state_dict`. It detects the discrepancy,
resizes the model's embedding and output layers to match the checkpoint, then re-ties weights.

Never call `load_state_dict` with a vocab-expanded checkpoint against a base-config model
without calling `_maybe_resize_vocab` first.

**Device rule:** `nn.Embedding` and `nn.Linear` default to CPU. When the model is already on
CUDA (e.g. `model.to(device)` called before `load_model_only`), always `.to(emb_device)` the
new layers immediately — `_maybe_resize_vocab` does this via `inner.tok_emb.weight.device`.

## `latest` Pointer File Can Be Stale — Never Trust Blindly
`checkpoints/{size}/latest` is a text file pointing to the most recent checkpoint dir.
It becomes stale when training is restarted from scratch in a directory that already has
high-step checkpoints from a previous run: the new `step_00000000` is numerically "oldest"
and gets pruned by `max_checkpoints` cleanup, but `latest` already points to the deleted dir.

**Always resolve checkpoints by scanning `step_*` dirs directly:**
```python
step_dirs = sorted(
    [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
    key=lambda d: int(d.name.split("_")[1]),
) if ckpt_dir.exists() else []
checkpoint = str(step_dirs[-1]) if step_dirs else latest.read_text().strip()
```

**`_cleanup_old_checkpoints` is protected:** it accepts `protected=str(ckpt_dir)` and never
deletes the just-saved checkpoint, even if its step number is lower than existing ones.
