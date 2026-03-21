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
