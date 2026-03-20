# Feature 34: Token Merging (ToMe)

## Overview

Token Merging (ToMe) reduces the number of tokens processed by the attention mechanism
by identifying and merging redundant token pairs. Similar tokens are averaged together
before attention, then "unmerged" (restored) after. This trades a small quality loss
for a significant speedup — roughly proportional to the fraction of tokens merged.

For code generation, tokens are less redundant than image patches, so merge rates
should be conservative (10–25%). Applied selectively in later transformer layers, the
approach can yield 1.3–1.8× speedup with negligible quality degradation.

Status: OPTIONAL — enable via `--feature token-merging` or CLI menu toggle.

---

## Motivation

- Long code sequences (256–1 024 tokens) spend most compute in attention O(T^2).
- Consecutive tokens in Python code often carry similar representations (e.g., multiple
  whitespace tokens, repeated variable name references, comment tokens).
- ToMe reduces effective sequence length without changing the model weights at all —
  zero training required.
- Practical: combine with KV-cache to get fast interactive generation.

Reference paper: "Token Merging: Your ViT But Faster" (Bolya et al. 2022).
Adapted for language models: "ToMe for LLMs" (various 2023 follow-ups).

---

## Architecture / Design

### Algorithm

```
For each transformer block (or subset of later blocks):
  1. Compute token similarity matrix using keys K (already computed for attention)
  2. Find R pairs of most-similar tokens (bipartite soft matching)
  3. Merge: replace each pair with weighted average
  4. Run attention on reduced sequence
  5. Unmerge: scatter averaged representation back to original positions
```

### Bipartite Soft Matching

Split tokens into two sets A and B (alternating: even/odd positions). Find the top-R
most similar A-B pairs using cosine similarity on the keys:

```python
# cola_coder/tome/matching.py

import torch


def bipartite_soft_matching(
    keys: torch.Tensor,   # (B, T, d_head)
    r: int,               # number of pairs to merge
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns merge and unmerge functions as index tensors.

    keys: query/key representation used to determine similarity
    r: number of tokens to reduce by (must be <= T//2)

    Returns:
        merge_idx:   (B, T-r, r_sources) — which tokens to average
        unmerge_idx: (B, T, 1) — where to scatter back
    """
    B, T, D = keys.shape
    half = T // 2
    r = min(r, half)

    a = keys[:, ::2, :]    # even positions  (B, half, D)
    b = keys[:, 1::2, :]   # odd positions   (B, half, D)

    # Cosine similarity between all A-B pairs
    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    sim = torch.bmm(a_norm, b_norm.transpose(1, 2))  # (B, half, half)

    # Greedy matching: for each A token, find best B match (no repeat)
    node_max, node_idx = sim.max(dim=-1)   # (B, half)
    edge_idx = node_max.argsort(dim=-1, descending=True)[..., :r]  # top-r A tokens

    return edge_idx, node_idx


def merge_tokens(
    x: torch.Tensor,    # (B, T, D)
    edge_idx: torch.Tensor,   # (B, r)
    node_idx: torch.Tensor,   # (B, T//2)
) -> tuple[torch.Tensor, dict]:
    """Merge top-r most similar pairs. Returns reduced tensor + merge info."""
    B, T, D = x.shape
    r = edge_idx.shape[1]

    # Build merge mask
    merged = x.clone()
    merge_info = {"edge_idx": edge_idx, "node_idx": node_idx, "orig_T": T}

    for b in range(B):
        for i in range(r):
            a_idx = edge_idx[b, i].item() * 2       # even index
            b_idx = node_idx[b, edge_idx[b, i].item()].item() * 2 + 1  # odd index
            # Average the pair into the A position
            merged[b, a_idx] = (merged[b, a_idx] + merged[b, b_idx]) / 2.0
            merged[b, b_idx] = float("nan")  # mark as merged-away

    # Remove merged-away tokens
    keep_mask = ~torch.isnan(merged[..., 0])  # (B, T)
    # Pad to same length in batch (some sequences may merge different amounts)
    out_list = [merged[b][keep_mask[b]] for b in range(B)]
    T_new = max(t.shape[0] for t in out_list)
    out = torch.zeros(B, T_new, D, device=x.device, dtype=x.dtype)
    for b, t in enumerate(out_list):
        out[b, :t.shape[0]] = t

    return out, merge_info


def unmerge_tokens(
    x: torch.Tensor,    # (B, T_reduced, D)
    merge_info: dict,
) -> torch.Tensor:
    """Restore original sequence length by repeating merged tokens."""
    orig_T = merge_info["orig_T"]
    edge_idx = merge_info["edge_idx"]
    node_idx = merge_info["node_idx"]
    B, T_r, D = x.shape

    out = torch.zeros(B, orig_T, D, device=x.device, dtype=x.dtype)
    # Reverse: scatter reduced tokens back
    src_pos = 0
    for pos in range(orig_T):
        out[:, pos] = x[:, min(src_pos, T_r - 1)]
        # Advance source only for non-merged positions
        src_pos = min(src_pos + 1, T_r - 1)
    return out
```

Note: the above is a simplified version for clarity. A production implementation uses
vectorized scatter/gather operations for efficiency.

### Vectorized Implementation

```python
# cola_coder/tome/merge_fast.py

import torch


def tome_merge(
    x: torch.Tensor,         # (B, T, D)
    keys: torch.Tensor,      # (B, T, D) — used for similarity
    r: int,                  # tokens to remove
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast vectorized merge. Returns (x_merged, src, dst) for unmerge.
    src: indices that were merged INTO dst
    dst: indices that received the merge
    """
    B, T, D = x.shape
    r = min(r, T // 2)

    # Split into two sets
    a = keys[:, ::2]     # (B, T//2, D)
    b = keys[:, 1::2]    # (B, T//2, D)

    a_n = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b_n = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    scores = a_n @ b_n.transpose(-1, -2)   # (B, T//2, T//2)

    # For each a, find best b
    best_b = scores.argmax(dim=-1)  # (B, T//2)

    # Sort a positions by score descending, take top-r
    best_score = scores.gather(2, best_b.unsqueeze(-1)).squeeze(-1)  # (B, T//2)
    top_r = best_score.argsort(dim=-1, descending=True)[:, :r]  # (B, r)

    # Merge: a absorbs b
    # (This is a simplified scatter-add; production code uses index_add_)
    x_new = x.clone()
    for b_idx in range(B):
        for i in range(r):
            a_i = top_r[b_idx, i].item() * 2
            b_i = best_b[b_idx, top_r[b_idx, i]].item() * 2 + 1
            if a_i < T and b_i < T:
                x_new[b_idx, a_i] = (x_new[b_idx, a_i] + x[b_idx, b_i]) / 2.0

    # Remove merged-away (b) positions — build keep mask
    keep = torch.ones(B, T, dtype=torch.bool, device=x.device)
    for b_idx in range(B):
        for i in range(r):
            b_i = best_b[b_idx, top_r[b_idx, i]].item() * 2 + 1
            if b_i < T:
                keep[b_idx, b_i] = False

    T_new = keep[0].sum().item()  # assume uniform for simplicity
    x_merged = x_new[keep].view(B, T_new, D)
    return x_merged, keep, top_r
```

### Integration into Attention Block

```python
# cola_coder/attention.py  (modified block)

class ToMeAttention(nn.Module):
    def __init__(self, config, layer_idx: int, tome_r: int = 0):
        super().__init__()
        self.tome_r = tome_r          # tokens to merge; 0 = disabled
        self.layer_idx = layer_idx
        # ... standard attention init ...

    def forward(self, x, mask=None, **kwargs):
        B, T, D = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        if self.tome_r > 0 and T > 2 * self.tome_r:
            # Use keys for similarity (they already capture content)
            x_merged, keep_mask, top_r_idx = tome_merge(x, k, r=self.tome_r)
            q_m = q[keep_mask].view(B, -1, D)
            k_m = k[keep_mask].view(B, -1, D)
            v_m = v[keep_mask].view(B, -1, D)
            attn_out = self._attention(q_m, k_m, v_m)
            # Unmerge: repeat merged positions
            attn_out = self._unmerge(attn_out, keep_mask, B, T, D)
        else:
            attn_out = self._attention(q, k, v)

        return self.o_proj(attn_out)

    def _unmerge(self, x_merged, keep_mask, B, T, D):
        out = torch.zeros(B, T, D, device=x_merged.device, dtype=x_merged.dtype)
        # Scatter merged output back; merged tokens copy from their source
        src_idx = 0
        for t in range(T):
            if keep_mask[0, t]:
                out[:, t] = x_merged[:, src_idx]
                src_idx += 1
            # else: merged-away token will be filled by scatter
        return out
```

### Adaptive Merge Rate by Layer

Later layers benefit more from merging (early layers need full spatial resolution):

```python
def get_tome_r_per_layer(
    n_layers: int,
    total_r: int,
    start_layer_fraction: float = 0.5,
) -> list[int]:
    """
    Distribute total_r merges across the last (1 - start_layer_fraction) layers.
    Returns list of r values per layer.
    """
    start = int(n_layers * start_layer_fraction)
    active_layers = n_layers - start
    r_per_layer = [0] * n_layers
    for i in range(start, n_layers):
        r_per_layer[i] = total_r // active_layers
    return r_per_layer
```

---

## Implementation Steps

1. **Create `cola_coder/tome/` package**: `__init__.py`, `matching.py`, `merge_fast.py`.

2. **Add `ToMeConfig` to `config.py`**:
   ```python
   @dataclass
   class ToMeConfig:
       enabled: bool = False
       merge_rate: float = 0.15    # fraction of tokens to merge per active layer
       start_layer: float = 0.5    # only merge in last half of layers
       use_keys_for_similarity: bool = True
   ```

3. **Modify attention blocks** in `attention.py` to accept `tome_r` parameter.

4. **Wire in model builder** to distribute r values across layers based on config.

5. **Benchmark**: measure tokens/sec before and after with various merge rates
   (5%, 10%, 15%, 20%, 25%). Plot speedup vs HumanEval pass@1 degradation.

6. **Add CLI option**: "Enable Token Merging (ToMe)" with merge rate slider.

7. **Disable during training** — ToMe is inference-only; gradients through the
   merge/unmerge operations are not needed.

---

## Key Files to Modify

| File | Change |
|---|---|
| `cola_coder/attention.py` | Add `tome_r` param, merge/unmerge in forward |
| `config.py` | Add `ToMeConfig` |
| `cola_coder/model.py` | Pass per-layer r values to attention blocks |
| `generator.py` | Enable/disable ToMe based on config |
| `benchmarks/speed_bench.py` | Add ToMe speedup benchmark |
| `cli/menu.py` | Add ToMe toggle option |
| `cola_coder/tome/` | New package |

---

## Testing Strategy

```python
# tests/test_tome.py

def test_merge_reduces_sequence_length():
    x = torch.randn(2, 64, 256)
    keys = torch.randn(2, 64, 256)
    x_merged, keep_mask, _ = tome_merge(x, keys, r=8)
    assert x_merged.shape[1] == 64 - 8   # 56 tokens after merge

def test_unmerge_restores_length():
    B, T, D = 2, 64, 256
    x = torch.randn(B, T, D)
    keys = torch.randn(B, T, D)
    x_merged, keep_mask, _ = tome_merge(x, keys, r=8)
    x_restored = unmerge_tokens_simple(x_merged, keep_mask, B, T, D)
    assert x_restored.shape == (B, T, D)

def test_tome_output_close_to_no_tome():
    """With small r, output should be close to unmerged."""
    model = build_small_model_with_tome(tome_r=2)
    model_base = build_small_model_with_tome(tome_r=0)
    # Copy weights
    model.load_state_dict(model_base.state_dict(), strict=False)
    x = torch.randint(0, 100, (1, 32))
    with torch.no_grad():
        out_tome = model(x)
        out_base = model_base(x)
    # Should be similar but not identical
    cos_sim = torch.nn.functional.cosine_similarity(
        out_tome.flatten(), out_base.flatten(), dim=0
    )
    assert cos_sim > 0.9   # high similarity at low merge rate

def test_tome_speedup():
    """ToMe should be measurably faster for long sequences."""
    import time
    model_base = build_small_model_with_tome(tome_r=0)
    model_tome = build_small_model_with_tome(tome_r=32)
    x = torch.randint(0, 100, (1, 256))

    t0 = time.perf_counter()
    for _ in range(50):
        model_base(x)
    base_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(50):
        model_tome(x)
    tome_time = time.perf_counter() - t0

    speedup = base_time / tome_time
    print(f"ToMe speedup: {speedup:.2f}x")
    assert speedup > 1.1   # at least 10% speedup
```

---

## Performance Considerations

- **Code vs vision**: image patches are highly redundant (sky pixels are similar). Code
  tokens are more diverse. Keep merge rate <= 20% for code to avoid degradation.
- **Critical tokens**: do not merge the first and last token of a function signature.
  Consider applying ToMe only to tokens in the middle of long sequences.
- **KV-cache interaction**: when using KV-cache for autoregressive generation, ToMe
  applies to the prefill pass (processing the prompt). During token-by-token generation,
  ToMe is less applicable (sequence grows by 1 each step).
- **Layer selection matters**: merging in layer 0 destroys low-level features needed
  by all subsequent layers. Start merging from layer `n_layers // 2`.
- **Profiling**: use `torch.profiler` to confirm attention is the bottleneck before
  adding ToMe complexity. For models < 100 M parameters, linear layers may dominate.

---

## Dependencies

```
torch>=2.2.0    # base requirement, no new deps
```

No new dependencies. ToMe is pure PyTorch.

---

## Estimated Complexity

| Aspect | Estimate |
|---|---|
| Bipartite matching (vectorized) | 3 hours |
| Merge / unmerge operations | 2 hours |
| Attention integration | 2 hours |
| Adaptive per-layer scheduling | 1 hour |
| Speedup benchmarking | 2 hours |
| Quality regression tests | 2 hours |
| CLI integration | 1 hour |
| **Total** | **~13 hours** |

Complexity rating: **Medium-Hard** — the algorithm is straightforward but getting
correct vectorized scatter/gather without off-by-one errors takes effort. Particularly
tricky with batches of different effective lengths after merging.

---

## 2026 Best Practices

- **Weighted merge**: instead of simple average, use attention-score-weighted average
  when merging (tokens with higher attention scores contribute more to the merged
  representation). Shown to reduce quality loss.
- **Merge in prefill, skip in decode**: during KV-cache-based autoregressive generation,
  only apply ToMe to the initial prompt encoding step (prefill), not to each new token.
  This gives most of the benefit with no interference with the decode loop.
- **Learned merge thresholds**: instead of a fixed r, train a lightweight classifier
  to predict which tokens are safe to merge (soft ToMe). More complex but adaptive.
- **StreamingLLM + ToMe**: combine with StreamingLLM (keep only first K + last K tokens
  in KV-cache) for very long context at constant memory — ToMe reduces compute,
  StreamingLLM reduces memory.
- **ToMe for speculative decoding draft model**: the draft model in speculative decoding
  needs to be fast. Applying ToMe to the draft model (not the verifier) is a practical
  combination.
