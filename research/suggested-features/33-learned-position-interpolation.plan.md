# Feature 33: Learned Position Interpolation (RoPE Scaling)

## Overview

Transformers trained with a maximum sequence length of 1 024 tokens can not generalize
to longer sequences out of the box — the rotary position embeddings (RoPE) encounter
angles they have never seen during training. Position interpolation solves this by
scaling down the RoPE frequencies so that positions 0–4095 map to the same angular
range that 0–1023 did during training, then fine-tuning briefly to adapt.

This feature extends Cola-Coder's effective context from 1 K to 4 K (or beyond) tokens
with only ~1 000 steps of additional training on long examples.

Status: OPTIONAL — enable via `--feature rope-scaling` or CLI menu toggle.

---

## Motivation

- Code generation benefits enormously from longer context: full file headers, imports,
  class definitions, and related functions can all fit in one context.
- Training a model from scratch on 4 K context is 4x more expensive than 1 K.
- Position interpolation is a near-free upgrade: fine-tune for a few hundred steps and
  the model handles 4x longer sequences with minimal perplexity degradation at the
  original length.
- Practically: reading a whole Python module (~200 lines) requires ~800 tokens minimum.
  At 4 K context, you can fit entire files with docstrings.

---

## Architecture / Design

### RoPE Recap

RoPE applies a rotation to query and key vectors based on position `m`:

```
q_rotated[m] = q * cos(m * theta) + rotate_half(q) * sin(m * theta)
theta_i = base^(-2i / d_head)   where base=10000, i=0..d_head//2
```

For position `m=1024`, the angle `m * theta_0 = 1024 * 10000^0 = 1024` radians — never
seen if training only went to 1023.

### Linear Interpolation (Meta 2023)

Scale every position by `factor = original_len / new_len`:

```
m_interpolated = m / factor   # maps [0, 4095] -> [0, 1023]
```

In code, this means multiplying the inverse frequency by `factor`:

```python
inv_freq_scaled = inv_freq / factor  # or equivalently:
# new_theta_i = base^(-2i/d) / factor
```

After interpolation, fine-tune for ~1 000 steps to let the model re-learn the
slightly different position encodings.

### YaRN (NTK-aware Scaled RoPE)

YaRN is a more principled scaling that applies different scale factors to different
frequency bands:

```
- Low frequencies (large wavelengths): no scaling needed, already generalize
- High frequencies (small wavelengths): scale by full factor
- Middle: smooth ramp between the two
```

```python
def yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, original_max_pos: int
) -> tuple[float, float]:
    low = max(math.floor(dim * math.log(original_max_pos / (low_rot * 2 * math.pi))
                         / (2 * math.log(base))), 0)
    high = min(math.ceil(dim * math.log(original_max_pos / (high_rot * 2 * math.pi))
                         / (2 * math.log(base))), dim - 1)
    return low, high
```

### Implementation in rope.py

```python
# cola_coder/rope.py  (modified)

import math
import torch
import torch.nn as nn
from enum import Enum


class RopeScalingType(str, Enum):
    NONE = "none"
    LINEAR = "linear"
    DYNAMIC = "dynamic"      # re-compute as needed, no fine-tune required
    YARN = "yarn"


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        base: float = 10_000.0,
        max_position: int = 1024,
        scaling_type: RopeScalingType = RopeScalingType.NONE,
        scaling_factor: float = 1.0,
        # YaRN params
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        yarn_original_max_pos: int = 1024,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position = max_position
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        if scaling_type == RopeScalingType.LINEAR:
            inv_freq = inv_freq / scaling_factor

        elif scaling_type == RopeScalingType.YARN:
            inv_freq = self._yarn_inv_freq(
                inv_freq, dim, base, scaling_factor,
                yarn_beta_fast, yarn_beta_slow, yarn_original_max_pos
            )

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_position)

    def _yarn_inv_freq(
        self, inv_freq, dim, base, scale, beta_fast, beta_slow, orig_max
    ) -> torch.Tensor:
        low, high = self._yarn_correction_range(
            beta_fast, beta_slow, dim, base, orig_max
        )
        smooth = torch.zeros_like(inv_freq)
        for i in range(len(inv_freq)):
            if i < low:
                smooth[i] = 1.0 / scale   # full scaling
            elif i > high:
                smooth[i] = 1.0           # no scaling
            else:
                ramp = (i - low) / max(1, high - low)
                smooth[i] = (1.0 - ramp) / scale + ramp
        return inv_freq * smooth

    @staticmethod
    def _yarn_correction_range(
        beta_fast, beta_slow, dim, base, orig_max
    ) -> tuple[int, int]:
        def correction_dim(num_rot):
            return (dim * math.log(orig_max / (num_rot * 2 * math.pi))
                    / (2 * math.log(base)))
        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), dim // 2 - 1)
        return low, high

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)           # (T, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)         # (T, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None):
        seq_len = x.shape[-2]
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)   # dynamic extension for inference
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        return apply_rotary_emb(x, cos, sin, position_ids)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if position_ids is not None:
        cos = cos[position_ids]   # (B, T, dim)
        sin = sin[position_ids]
    else:
        cos = cos.unsqueeze(0)    # (1, T, dim)
        sin = sin.unsqueeze(0)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin
```

### Config Extension

```python
# in config.py

@dataclass
class RopeScalingConfig:
    enabled: bool = False
    scaling_type: str = "linear"   # "linear" | "yarn" | "dynamic"
    scaling_factor: float = 4.0    # new_max_len / original_max_len
    finetune_steps: int = 1000
    finetune_lr: float = 2e-5
    long_seq_data_path: str = "data/long_sequences/"
```

---

## Implementation Steps

1. **Modify `rope.py`** to add `RopeScalingType` enum and extend `RotaryEmbedding`
   with linear and YaRN scaling options (see code above).

2. **Add `RopeScalingConfig`** to `config.py` and wire into `ModelConfig`.

3. **Update model instantiation** in `model.py` / `attention.py` to pass scaling
   params to `RotaryEmbedding`:
   ```python
   self.rope = RotaryEmbedding(
       dim=head_dim,
       base=cfg.rope_base,
       max_position=cfg.rope_scaling.scaling_factor * cfg.max_seq_len
           if cfg.rope_scaling.enabled else cfg.max_seq_len,
       scaling_type=RopeScalingType(cfg.rope_scaling.scaling_type),
       scaling_factor=cfg.rope_scaling.scaling_factor,
   )
   ```

4. **Generate long-sequence fine-tuning data**: slice training files into 4 K token
   chunks. Even 500–1 000 such examples are sufficient.
   ```python
   def create_long_seq_dataset(source_dir: Path, target_len: int = 4096) -> list:
       chunks = []
       for path in source_dir.glob("**/*.py"):
           tokens = tokenizer.encode(path.read_text())
           for start in range(0, len(tokens) - target_len, target_len // 2):
               chunks.append(tokens[start:start + target_len])
       return chunks
   ```

5. **Fine-tune for 1 000 steps** with AdamW lr=2e-5 on long sequences. Keep batch
   size small (2–4) due to memory.

6. **Add CLI option**: "Extend context with RoPE scaling" → shows factor options
   (2x, 4x, 8x), runs data prep, runs fine-tuning.

7. **Update `generator.py`**: set `max_new_tokens` upper bound to new context length.
   Update KV-cache pre-allocation accordingly.

8. **Evaluate**: compare perplexity at original (1 K) and new (4 K) length before and
   after fine-tuning.

---

## Key Files to Modify

| File | Change |
|---|---|
| `cola_coder/rope.py` | Add scaling modes, YaRN, dynamic extension |
| `config.py` | Add `RopeScalingConfig` |
| `cola_coder/attention.py` | Pass scaling config to `RotaryEmbedding` |
| `train.py` | Add `--extend-context` mode |
| `generator.py` | Update max seq len, KV-cache size |
| `cli/menu.py` | Add "Extend context" option |

---

## Testing Strategy

```python
# tests/test_rope_scaling.py

def test_linear_scaling_covers_new_length():
    rope_orig = RotaryEmbedding(dim=64, max_position=1024)
    rope_scaled = RotaryEmbedding(
        dim=64, max_position=4096,
        scaling_type=RopeScalingType.LINEAR, scaling_factor=4.0
    )
    x = torch.randn(1, 2048, 1, 64)  # exceeds original max
    # Should not raise, should produce valid output
    out = rope_scaled(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()

def test_rope_output_unchanged_at_original_length():
    """Linear scaling should produce nearly identical outputs for short sequences."""
    rope_orig = RotaryEmbedding(dim=64, max_position=1024)
    rope_scaled = RotaryEmbedding(
        dim=64, max_position=4096,
        scaling_type=RopeScalingType.LINEAR, scaling_factor=4.0
    )
    x = torch.randn(1, 100, 1, 64)
    # Not identical (different freqs) but should have similar magnitude
    out_orig = rope_orig(x)
    out_scaled = rope_scaled(x)
    # Rotated vectors should have same norm
    assert torch.allclose(out_orig.norm(dim=-1), out_scaled.norm(dim=-1), atol=1e-5)

def test_yarn_scaling_no_nan():
    rope = RotaryEmbedding(
        dim=64, max_position=4096,
        scaling_type=RopeScalingType.YARN, scaling_factor=4.0,
        yarn_original_max_pos=1024
    )
    x = torch.randn(1, 4000, 1, 64)
    out = rope(x)
    assert not torch.isnan(out).any()

def test_perplexity_doesnt_degrade_short():
    """After fine-tuning on long seqs, short-seq perplexity should stay within 5%."""
    # Integration test — run with a small checkpoint
    pass  # placeholder for benchmark runner
```

---

## Performance Considerations

- **Fine-tuning cost**: 1 000 steps × batch 4 × 4 096 tokens = ~16 M tokens of compute.
  On an RTX 3080 this takes roughly 30–60 minutes depending on model size.
- **KV-cache memory**: scales linearly with context length. At 4 K, KV cache is 4×
  larger than at 1 K. Pre-allocate accordingly or switch to paged/streaming KV cache.
- **Dynamic RoPE**: skip fine-tuning entirely by using `scaling_type="dynamic"` which
  re-computes frequencies on the fly for any seen length. Quality is lower but zero
  fine-tuning cost.
- **Attention complexity**: 4 K context means attention is 16× more expensive than
  1 K (O(T^2)). Combine with Flash Attention 2 for tractable training.
- **Chunk-based long-context training**: instead of full 4 K sequences, train with
  gradient checkpointing and use Ring Attention or sliding-window attention to reduce
  memory.

---

## Dependencies

```
torch>=2.2.0     # base requirement
flash-attn>=2.5  # strongly recommended for long-context training (optional)
```

No new dependencies required for basic linear interpolation.

---

## Estimated Complexity

| Aspect | Estimate |
|---|---|
| Modify rope.py (linear) | 2 hours |
| Add YaRN scaling | 3 hours |
| Config + wiring | 1 hour |
| Long-seq data pipeline | 2 hours |
| Fine-tuning script | 2 hours |
| Perplexity eval | 1 hour |
| CLI integration | 1 hour |
| Tests | 2 hours |
| **Total** | **~14 hours** |

Complexity rating: **Medium** — math is well-documented; tricky parts are getting
fine-tuning data and validating no regression on short sequences.

---

## 2026 Best Practices

- **LongRoPE**: as of 2025-2026, LongRoPE identifies non-uniform optimal scaling per
  dimension using evolutionary search, outperforming both linear and YaRN. Worth
  investigating if context extension quality matters.
- **Sliding window attention (SWA)**: combine RoPE scaling with SWA (attend only to
  the last N tokens per head) for O(T) attention complexity at very long contexts.
- **RoPE base scaling**: increasing the base from 10K to 500K (as done in Llama 3) can
  enable long-context without position interpolation at all — if training from scratch.
- **Context window progressively during training**: train at 1 K, then extend to 2 K,
  then 4 K in successive phases (each phase ~10% of original training compute). Better
  than jumping directly to 4 K.
- **Position IDs for packed sequences**: when packing multiple samples into one long
  sequence for efficiency, use per-sample position IDs (reset to 0 at each sample
  boundary) rather than global position IDs.
