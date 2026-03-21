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
- Data prep: `--workers N` (parallel filters), `--score` (quality weights), `--no-filter`, `--filter-strict`
- Prepared data reusable — only re-prepare if tokenizer/seq_len/dataset/languages/filter changes
- HuggingFace dataset gated: needs `HF_TOKEN` env var
