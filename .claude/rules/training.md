---
match: "**/training/**,**/trainer.py,configs/*.yaml,scripts/train*.py"
---

# Training Rules

- Loss starts ~10.4 (random), targets: 2.0-2.5 small, 1.5-2.0 medium, 1.3-1.8 4080_max
- Perplexity = exp(loss), target 8-15 for good code generation
- effective_batch = batch_size * gradient_accumulation
- bf16 on RTX 4080 (no GradScaler), fp16 on RTX 3080 (needs GradScaler)
- torch.compile: ~20-40% speedup, adds ~20% memory overhead
- Flash Attention: `F.scaled_dot_product_attention` with `is_causal=True`
- Gradient checkpointing: required for 4080_max (455M), optional for medium (299M)
- VRAM activation memory dominated by FFN hidden_dim, not model dim
- Data prep: `--workers N` (parallel filters), `--score` (quality weights), `--no-filter`, `--filter-strict`, `--dedup {none,exact}` (exact SHA-256 chunk dedup, ON by default — raw code is 25-40% dups; `--no-dedup` to keep all). Cross-dataset near-dup (MinHash) lives in `combine_datasets.py`.
- Prepared data reusable — only re-prepare if tokenizer/seq_len/dataset/languages/filter/dedup changes
- HuggingFace dataset gated: needs `HF_TOKEN` env var

## torch.compile — call the model, not its methods
`torch.compile(model)` returns an OptimizedModule that only compiles `__call__`/
`forward`. Calling `model.compute_loss(x)` on a compiled model silently runs the
ORIGINAL uncompiled module (verified: zero dynamo compilations). The trainer must
call `logits = model(input_ids)` and apply `language_modeling_loss(logits, ids,
weights)` from `cola_coder.model.transformer`. Never route training through
method calls on a compiled model.

## Wiring invariants (each was once a silent no-op — keep them wired)
- `config.training.gradient_checkpointing` → `Transformer.forward` wraps blocks
  in `torch.utils.checkpoint` (training only, bypassed for KV-cache inference)
- `config.model.rope_theta` + `rope_scaling` → `get_rope_freqs()` in
  `Transformer.__init__`. Checkpoints trained before 2026-06-10 all used the
  default theta=10000 (no config on disk had a non-default value at the time)
- Quality weights are per-sample: `language_modeling_loss(..., sample_weights)`
  computes a weighted mean over per-sequence losses — never `loss * w.mean()`
- fp16: LR scheduler must NOT step when GradScaler skips the optimizer step
  (compare `scaler.get_scale()` before/after `update()`)
- Warmup lambda uses `(step + 1) / warmup_steps` so step 0 has nonzero LR
