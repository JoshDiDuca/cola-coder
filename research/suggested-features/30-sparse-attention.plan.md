# Feature 30: Sparse Attention (Sliding Window + Global)

**Status:** Optional | **CLI Flag:** `--sparse-attention` | **Complexity:** High

---

## Overview

Replace the standard O(n²) full self-attention with a sparse attention pattern combining: (1) sliding window local attention (each token attends to ±W/2 neighbors), and (2) global attention tokens (every Gth token attends to all other tokens). This reduces attention complexity from O(n²) to O(n·W) while preserving long-range context through global tokens. Enables 4K+ token contexts on a single consumer GPU. Implementation: modify the attention mask in `attention.py`; no attention kernel changes required for correctness (though custom CUDA kernels like Flash-Attention + sparse patterns improve performance).

Reference: Longformer (Beltagy et al., 2020); BigBird (Zaheer et al., 2020).

---

## Motivation

Cola-Coder's current full attention has quadratic VRAM scaling: doubling context length quadruples the attention map size. For typical code generation (2048 tokens), this is manageable. But:

- Complete function files often exceed 2048 tokens
- Adding file-level context (imports + type definitions + current function) requires 4K+ tokens
- GRPO reasoning with thinking tokens requires long contexts
- Fine-tuning on full TypeScript files is impossible at full attention with 350M model on 3080

Sparse attention allows extending to 4096–8192 tokens with sub-linear VRAM growth:
- Local window W=256: each token attends to 256 neighbors → O(n·256) vs O(n²)
- At n=4096: full attention = 16M pairs; sparse = 1M pairs → 16x fewer attention ops
- Global tokens every 64: adds back ~64 fully-connected "highway" tokens for long-range signal

---

## Architecture / Design

### Attention Pattern Visualization

```
Standard attention (n=8):
  ■■■■■■■■
  ■■■■■■■■
  ■■■■■■■■
  ■■■■■■■■
  ■■■■■■■■
  ■■■■■■■■
  ■■■■■■■■
  ■■■■■■■■

Sliding window (W=4, causal):
  ■·······
  ■■······
  ■■■·····
  ·■■■····
  ··■■■···
  ···■■■··
  ····■■■·
  ·····■■■

Sliding window + global (stride G=4, global token at 0, 4):
  ■···■···  ← token 0 is global: attends to all
  ■■··■···
  ■■■·■···
  ·■■■■···
  ■■■■■···  ← token 4 is global: attends to all
  ···■■■■·
  ···■■■■·  (token 4 also attends here via global)
  ···■■■■■
```

### Complexity Analysis

| Config | Tokens | Attention Pairs |
|--------|--------|----------------|
| Full   | 2048   | 4.2M           |
| Full   | 4096   | 16.8M          |
| Window W=256 | 4096 | 1.0M    |
| Window W=256 + Global G=64 | 4096 | 1.3M |
| Window W=256 + Global G=64 | 8192 | 2.6M |

---

## Implementation Steps

### Step 1: Sparse Attention Config

```python
# cola_coder/model/sparse_attention.py
from dataclasses import dataclass
from typing import Literal

@dataclass
class SparseAttentionConfig:
    window_size: int = 256          # Local attention window (tokens on each side)
    global_stride: int = 64         # Every Nth token is a global token
    attention_pattern: Literal["window", "window_global", "full"] = "window_global"
    # Per-layer config: None = use default for all layers
    layer_patterns: dict[int, str] = None  # {layer_idx: "window" | "full"}
    # Flash attention integration (requires flash-attn package)
    use_flash_sparse: bool = False
```

### Step 2: Sparse Mask Generation

```python
# cola_coder/model/sparse_attention.py  (continued)
import torch
import torch.nn.functional as F

def build_sliding_window_mask(
    seq_len: int,
    window_size: int,
    causal: bool = True,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create boolean attention mask for sliding window attention.
    True = attend, False = mask out.
    Shape: [seq_len, seq_len]

    For causal window: each position attends to [max(0, i-window_size) : i+1]
    For bidirectional window: each position attends to ±window_size neighbors
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    if causal:
        for i in range(seq_len):
            start = max(0, i - window_size + 1)
            end = i + 1
            mask[i, start:end] = True
    else:
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = True

    return mask


def build_global_tokens_mask(
    seq_len: int,
    global_stride: int,
    base_mask: torch.Tensor,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Extend a base mask with global token positions.
    Global tokens: attend to ALL positions and are attended by ALL positions.
    """
    mask = base_mask.clone()
    global_positions = list(range(0, seq_len, global_stride))

    for g in global_positions:
        # Global token attends to everything before it (causal)
        mask[g, :g+1] = True
        # All tokens can attend to global token
        mask[:, g] = True
        # Also allow global tokens to attend to each other
        for g2 in global_positions:
            if g2 <= g:
                mask[g, g2] = True

    return mask


def build_sparse_attention_mask(
    seq_len: int,
    cfg: SparseAttentionConfig,
    causal: bool = True,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Build the full sparse attention mask based on config.
    Returns a float mask (0 = attend, -inf = mask out) for use in attention.
    """
    if cfg.attention_pattern == "full":
        # Standard causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    elif cfg.attention_pattern == "window":
        mask = build_sliding_window_mask(seq_len, cfg.window_size, causal, device)
    elif cfg.attention_pattern == "window_global":
        window_mask = build_sliding_window_mask(seq_len, cfg.window_size, causal, device)
        mask = build_global_tokens_mask(seq_len, cfg.global_stride, window_mask, device)
    else:
        raise ValueError(f"Unknown pattern: {cfg.attention_pattern}")

    # Convert bool mask to float mask for softmax
    float_mask = torch.zeros(seq_len, seq_len, device=device)
    float_mask.masked_fill_(~mask, float("-inf"))
    return float_mask


def get_attention_mask_for_layer(
    layer_idx: int,
    seq_len: int,
    cfg: SparseAttentionConfig,
    device: torch.device,
) -> torch.Tensor:
    """Return appropriate mask for a specific layer (supports per-layer patterns)."""
    if cfg.layer_patterns and layer_idx in cfg.layer_patterns:
        pattern = cfg.layer_patterns[layer_idx]
        layer_cfg = SparseAttentionConfig(
            window_size=cfg.window_size,
            global_stride=cfg.global_stride,
            attention_pattern=pattern,
        )
        return build_sparse_attention_mask(seq_len, layer_cfg, device=device)
    return build_sparse_attention_mask(seq_len, cfg, device=device)
```

### Step 3: Integration into Attention Module

```python
# cola_coder/model/attention.py  (modifications)
from .sparse_attention import (
    SparseAttentionConfig,
    build_sparse_attention_mask,
    get_attention_mask_for_layer,
)

class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg, sparse_cfg: SparseAttentionConfig = None):
        super().__init__()
        # ... existing init ...
        self.sparse_cfg = sparse_cfg
        self._mask_cache: dict[int, torch.Tensor] = {}

    def _get_mask(self, seq_len: int, layer_idx: int, device: torch.device) -> torch.Tensor:
        """Cached mask generation — recomputed only if seq_len changes."""
        cache_key = (seq_len, layer_idx)
        if cache_key not in self._mask_cache:
            if self.sparse_cfg is not None:
                mask = get_attention_mask_for_layer(
                    layer_idx, seq_len, self.sparse_cfg, device
                )
            else:
                # Standard causal mask
                mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=device),
                    diagonal=1,
                )
            self._mask_cache[cache_key] = mask
            # Limit cache size
            if len(self._mask_cache) > 16:
                oldest_key = next(iter(self._mask_cache))
                del self._mask_cache[oldest_key]
        return self._mask_cache[cache_key]

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        layer_idx: int = 0,
        external_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Compute Q, K, V (existing GQA logic)
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim)

        # Apply RoPE (existing)
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Expand KV for GQA
        groups = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(groups, dim=2)
        v = v.repeat_interleave(groups, dim=2)

        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention with sparse mask
        if external_mask is not None:
            attn_mask = external_mask
        elif self.sparse_cfg is not None:
            attn_mask = self._get_mask(T, layer_idx, x.device)
        else:
            attn_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device), diagonal=1
            )

        # Scaled dot-product attention (uses attn_mask additively)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).reshape(B, T, C)
        return self.o_proj(attn_output)
```

### Step 4: Mixed Layer Strategy

```python
# Recommended config for 4K context on RTX 3080 with 125M model:
# configs/model/small_sparse.yaml
"""
d_model: 512
n_heads: 8
n_kv_heads: 2
n_layers: 12
max_seq_len: 4096
use_sparse_attention: true
sparse_window_size: 256
sparse_global_stride: 64
# First 2 layers: full attention (short context is cheap, useful for early layers)
# Middle 8 layers: window+global
# Last 2 layers: full attention (final representation benefits from full context)
sparse_layer_patterns:
  0: "full"
  1: "full"
  10: "full"
  11: "full"
"""
```

### Step 5: Flash-Attention Sparse Integration (Optional)

```python
# cola_coder/model/flash_sparse.py
"""
Optional: Use flash-attention's built-in sparse attention support
for further memory and speed optimization.
Requires: pip install flash-attn
"""

def flash_sparse_attention_forward(q, k, v, window_size: int):
    """
    Use FlashAttention-2's sliding window attention.
    Significantly faster than our mask-based implementation for long sequences.
    """
    try:
        from flash_attn import flash_attn_varlen_func
        # flash_attn supports window_size parameter natively
        return flash_attn_varlen_func(
            q, k, v,
            window_size=(window_size, 0),  # (left, right) for causal
        )
    except ImportError:
        raise ImportError("Install flash-attn for optimized sparse attention: "
                          "pip install flash-attn --no-build-isolation")
```

### Step 6: Context Length Extension CLI

```python
@app.command()
def generate_long(
    prompt: str = typer.Argument(...),
    max_tokens: int = typer.Option(1024, "--max-tokens"),
    window_size: int = typer.Option(256, "--window"),
    global_stride: int = typer.Option(64, "--global-stride"),
    context_file: str = typer.Option(None, "--context-file",
                                      help="Prepend content of this file as context"),
):
    """Generate with sparse attention for long-context code completion."""
    context = ""
    if context_file:
        import pathlib
        context = pathlib.Path(context_file).read_text()
    full_prompt = context + "\n\n" + prompt
    tokens = tokenizer.encode(full_prompt)
    console.print(f"[dim]Context: {len(tokens)} tokens[/dim]")
    # ... generate with sparse attention enabled


@app.command()
def benchmark_attention(
    seq_lengths: str = typer.Option("512,1024,2048,4096,8192"),
    window_size: int = typer.Option(256),
):
    """Benchmark sparse vs full attention at various context lengths."""
    import time
    from .sparse_attention import build_sparse_attention_mask, SparseAttentionConfig

    sparse_cfg = SparseAttentionConfig(window_size=window_size, global_stride=64)
    for T in [int(x) for x in seq_lengths.split(",")]:
        # Full attention time
        t0 = time.perf_counter()
        full_mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        _ = full_mask.to("cuda")
        full_time = (time.perf_counter() - t0) * 1000

        # Sparse mask time
        t0 = time.perf_counter()
        sparse_mask = build_sparse_attention_mask(T, sparse_cfg, device=torch.device("cuda"))
        sparse_time = (time.perf_counter() - t0) * 1000

        density = (sparse_mask != float("-inf")).float().mean().item()
        console.print(
            f"T={T:5d} | full={full_time:6.1f}ms | sparse={sparse_time:6.1f}ms | "
            f"density={density:.1%}"
        )
```

---

## Key Files to Modify

- `cola_coder/model/sparse_attention.py` — new file (masks, config)
- `cola_coder/model/attention.py` — integrate sparse mask + caching
- `cola_coder/model/transformer.py` — pass `layer_idx` to attention
- `cola_coder/model/config.py` — add sparse attention config fields
- `cola_coder/model/flash_sparse.py` — optional flash-attn integration
- `cola_coder/cli.py` — `generate-long`, `benchmark-attention` commands
- `configs/model/small_sparse.yaml` — model config with sparse attention
- `configs/model/medium_sparse.yaml` — medium model sparse config

---

## Testing Strategy

```python
def test_window_mask_shape():
    mask = build_sliding_window_mask(seq_len=16, window_size=4, causal=True)
    assert mask.shape == (16, 16)

def test_window_mask_causal():
    mask = build_sliding_window_mask(seq_len=8, window_size=3, causal=True)
    # Token 5 should attend to 3, 4, 5 but NOT 6, 7
    assert mask[5, 3].item() and mask[5, 4].item() and mask[5, 5].item()
    assert not mask[5, 6].item() and not mask[5, 7].item()

def test_global_tokens_attend_all():
    base = build_sliding_window_mask(8, 3, causal=True)
    mask = build_global_tokens_mask(8, global_stride=4, base_mask=base)
    # Token 4 is global: should attend to 0-4
    assert mask[4, 0].item()
    assert mask[4, 1].item()
    assert mask[4, 2].item()

def test_sparse_mask_density():
    cfg = SparseAttentionConfig(window_size=64, global_stride=64)
    mask = build_sparse_attention_mask(512, cfg)
    float_mask = (mask != float("-inf")).float()
    density = float_mask.mean().item()
    full_causal_density = 0.5  # Lower triangle
    assert density < full_causal_density * 0.5  # Sparse = less than 50% of causal

def test_attention_output_shape():
    # Verify attention with sparse mask produces correct output shape
    from cola_coder.model.attention import GroupedQueryAttention
    sparse_cfg = SparseAttentionConfig(window_size=32, global_stride=16)
    attn = GroupedQueryAttention(model_cfg, sparse_cfg)
    x = torch.randn(1, 128, model_cfg.d_model)
    freqs = model.freqs_cis[:128]
    out = attn(x, freqs, layer_idx=0)
    assert out.shape == x.shape

def test_sparse_context_extension():
    # Verify model can process 4096 tokens without OOM
    # (run on CPU if GPU not available in CI)
    model_with_sparse = load_sparse_model()
    x = torch.randint(0, 32000, (1, 4096))
    with torch.no_grad():
        out = model_with_sparse(x)
    assert out.shape == (1, 4096, 32000)
```

---

## Performance Considerations

- **Mask caching is critical:** Recomputing the sparse mask for every attention call adds overhead. Cache by `(seq_len, layer_idx)`.
- **`F.scaled_dot_product_attention` limitation:** PyTorch's SDPA with an additive float mask still materializes the full O(n²) attention score matrix internally. For true O(n·w) memory usage, use the Flash-Attention sparse kernel or implement a custom CUDA kernel.
- **Flash-Attention 2 native sparse:** FA2 supports `window_size` parameter which implements Longformer-style sliding window natively with O(n·w) memory. This is the recommended path for production use.
- **Profile at target context length:** Measure actual VRAM at n=2048, 4096, 8192 with and without sparse attention. The benefit should be measurable on RTX 3080 (10GB) at n=4096+.
- **Mixed precision:** Use `bfloat16` for the attention computation. The sparse mask float values (-inf, 0) are represented exactly in both fp16 and bf16.
- **Global token selection:** Instead of uniform stride (every 64th token), consider using separator tokens (newlines, function boundaries) as global tokens. This is more semantically meaningful and can improve quality.

---

## Dependencies

- PyTorch 2.x (`F.scaled_dot_product_attention` with additive mask)
- Optional: `flash-attn` (for O(n·w) memory; major performance improvement)
- Existing Cola-Coder attention module (GQA + RoPE)

---

## Estimated Complexity

| Task                               | Effort   |
|------------------------------------|----------|
| Sparse mask generation functions   | 3h       |
| Attention module integration       | 3h       |
| Mask caching                       | 1h       |
| Per-layer pattern support          | 2h       |
| Flash-attn integration (optional)  | 3h       |
| CLI commands + benchmark           | 2h       |
| Config + YAML                      | 1h       |
| Tests                              | 2h       |
| **Total**                          | **~17h** |

Overall complexity: **High** (mask generation is straightforward; true O(n·w) memory requires external kernel; debugging sparse pattern bugs is subtle)

---

## 2026 Best Practices

- **Longformer pattern is the right choice:** The combined window + global attention from Longformer is well-validated and provides both efficiency and quality. Avoid exotic patterns (e.g., random attention from BigBird) for code tasks.
- **Flash-Attention 2 is production-ready:** For 2026, FA2 with native `window_size` support should be the default implementation. The mask-based fallback is for correctness testing and CPU-only environments.
- **Context window vs quality tradeoff:** Longer contexts are not always better. Measure code generation quality at 2048 vs 4096 tokens. If quality doesn't improve with longer context, the extra complexity is not worth it.
- **Rope scaling for length extrapolation:** Sparse attention alone doesn't help if RoPE positional encodings weren't trained at the target length. Pair with YaRN or dynamic RoPE scaling when extending context beyond the training length.
- **Gradient checkpointing at long contexts:** At n=4096+, activation memory dominates. Enable `torch.utils.checkpoint` on transformer blocks when training with long contexts, even if it wasn't needed at 2048 tokens.
- **Evaluate on long-context benchmarks:** Use a held-out set of long TypeScript files (full component trees, full API route files) to measure whether sparse attention meaningfully improves completion quality vs truncated context.
