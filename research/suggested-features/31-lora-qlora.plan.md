# Feature 31: LoRA / QLoRA — Low-Rank Adaptation

## Overview

LoRA (Low-Rank Adaptation) adds small trainable low-rank matrices to frozen pretrained
weight matrices. Instead of updating all parameters during fine-tuning, only the LoRA
adapters are trained, cutting VRAM usage by 10-100x. QLoRA extends this by quantizing
the frozen base model to 4-bit before attaching bf16 LoRA adapters, enabling fine-tuning
of large models on consumer GPUs.

This feature makes Cola-Coder cheap to specialize: domain data (Python libs, SQL, shell)
can be injected without re-training from scratch. The final weights can be merged back
into the base for zero-overhead inference.

Status: OPTIONAL — enable via `--feature lora` or CLI menu toggle.

---

## Motivation

- Full fine-tuning of a 300M parameter Cola-Coder model needs ~4 GB VRAM in bf16
  just for weights, plus ~16 GB for optimizer states (Adam). This exceeds many setups.
- LoRA with rank 8 and alpha 16 adds ~1% extra parameters, needs ~200 MB VRAM for
  adapters, and Adam states only update those small matrices.
- QLoRA (4-bit base) halves the base weight footprint again, fitting fine-tuning inside
  8 GB VRAM.
- Common use cases: make the model better at FastAPI boilerplate, NumPy patterns,
  SQL generation, or any narrow domain where you have 500–5 000 examples.

---

## Architecture / Design

### Core Concept

For a weight matrix `W ∈ R^{d×k}`, LoRA decomposes the update as:

```
W' = W + BA
where B ∈ R^{d×r}, A ∈ R^{r×k}, rank r << min(d, k)
```

During training, `W` is frozen. Only `A` and `B` are updated.
At inference time, `W + BA` is pre-computed (merged) so there is no runtime cost.

### Injection Points

LoRA is injected into the attention projections of each transformer block:
- `q_proj` — query projection
- `k_proj` — key projection
- `v_proj` — value projection
- `o_proj` — output projection

Optionally also into `gate_proj` / `up_proj` / `down_proj` in the MLP (more parameters,
more specialization but higher cost).

### Class Design

```python
# cola_coder/lora/lora_linear.py

import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with LoRA adapter."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.scaling = lora_alpha / rank
        self.merged = False

        # Frozen base weight
        self.weight = nn.Parameter(linear.weight.data, requires_grad=False)
        self.bias = (
            nn.Parameter(linear.bias.data, requires_grad=False)
            if linear.bias is not None
            else None
        )

        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        # Initialize A with kaiming, B with zeros (so adapter starts as identity delta)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = nn.functional.linear(x, self.weight, self.bias)
        if self.merged or self.rank == 0:
            return base_out
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling

    def merge(self) -> None:
        """Merge LoRA delta into base weight for zero-overhead inference."""
        if self.merged:
            return
        self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        self.merged = True

    def unmerge(self) -> None:
        """Undo merge (needed if you want to fine-tune again after merging)."""
        if not self.merged:
            return
        self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
        self.merged = False
```

### Injection Helper

```python
# cola_coder/lora/inject.py

import torch.nn as nn
from .lora_linear import LoRALinear


TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    target_modules: set[str] | None = None,
) -> nn.Module:
    """Replace target Linear layers with LoRALinear. Freeze everything else."""
    targets = target_modules or TARGET_MODULES
    replaced = 0

    for name, module in list(model.named_modules()):
        parent_name, _, child_name = name.rpartition(".")
        if child_name not in targets:
            continue
        if not isinstance(module, nn.Linear):
            continue
        parent = model.get_submodule(parent_name) if parent_name else model
        lora_layer = LoRALinear(module, rank=rank, lora_alpha=lora_alpha,
                                lora_dropout=lora_dropout)
        setattr(parent, child_name, lora_layer)
        replaced += 1

    # Freeze everything that is not a LoRA parameter
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad_(False)

    print(f"Injected LoRA into {replaced} linear layers. "
          f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge all LoRA adapters back into base weights for fast inference."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
    return model
```

### QLoRA Extension

```python
# cola_coder/lora/qlora.py

from bitsandbytes.nn import Linear4bit
import torch.nn as nn
from .lora_linear import LoRALinear


def quantize_to_4bit(model: nn.Module) -> nn.Module:
    """Replace nn.Linear with 4-bit quantized versions (NF4 by default)."""
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        q_layer = Linear4bit(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            quant_type="nf4",           # NormalFloat4 — best quality
            compute_dtype="bfloat16",   # compute in bf16 for speed
        )
        q_layer.weight = module.weight
        setattr(parent, child_name, q_layer)
    return model
```

---

## Implementation Steps

1. **Create `cola_coder/lora/` package** with `__init__.py`, `lora_linear.py`,
   `inject.py`, `qlora.py`, `trainer.py`, `save_load.py`.

2. **Add `--lora` flag** to the CLI menu and `train.py` entry point.
   When enabled, call `inject_lora(model, rank=cfg.lora_rank)` before training.

3. **Add `LoRAConfig` dataclass** to `config.py`:
   ```python
   @dataclass
   class LoRAConfig:
       enabled: bool = False
       rank: int = 8
       alpha: float = 16.0
       dropout: float = 0.05
       target_modules: list[str] = field(default_factory=lambda: ["q_proj","v_proj"])
       use_qlora: bool = False
   ```

4. **Implement `LoRATrainer`** (thin wrapper around standard training loop that only
   passes LoRA parameters to the optimizer):
   ```python
   lora_params = [p for p in model.parameters() if p.requires_grad]
   optimizer = torch.optim.AdamW(lora_params, lr=2e-4, weight_decay=0.01)
   ```

5. **Checkpoint saving** — save only LoRA weights, not full model:
   ```python
   lora_state = {k: v for k, v in model.state_dict().items()
                 if "lora_A" in k or "lora_B" in k}
   torch.save(lora_state, "lora_adapter.pt")
   ```

6. **Checkpoint loading**:
   ```python
   model.load_state_dict(lora_state, strict=False)  # strict=False — only updates LoRA keys
   ```

7. **Merge step** before inference or export:
   ```python
   if cfg.lora.merge_for_inference:
       model = merge_lora_weights(model)
   ```

8. **Wire into `generator.py`** — load base model, optionally load LoRA adapter, optionally merge.

9. **Add CLI menu option**: "Fine-tune with LoRA" → prompts for data path, rank, steps.

---

## Key Files to Modify

| File | Change |
|---|---|
| `config.py` | Add `LoRAConfig` dataclass, include in `ModelConfig` |
| `train.py` | Accept `--lora` flag, call `inject_lora` if enabled |
| `generator.py` | Add `load_lora_adapter(path)` and `merge_lora()` methods |
| `server.py` | Accept `lora_adapter` query param or startup config |
| `cli/menu.py` | Add LoRA fine-tune option |
| `cola_coder/lora/` | New package (all new files) |

---

## Testing Strategy

```python
# tests/test_lora.py

def test_lora_injection_reduces_trainable_params():
    model = build_small_model()
    total_before = sum(p.numel() for p in model.parameters())
    inject_lora(model, rank=4)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable / total_before < 0.02  # < 2% of params trainable

def test_lora_merge_is_equivalent():
    model = build_small_model()
    inject_lora(model, rank=4)
    x = torch.randn(2, 16, model.config.d_model)
    out_before = model(x)
    merge_lora_weights(model)
    out_after = model(x)
    assert torch.allclose(out_before, out_after, atol=1e-5)

def test_lora_checkpoint_roundtrip():
    model = build_small_model()
    inject_lora(model, rank=4)
    # Simulate training step
    lora_state = {k: v for k, v in model.state_dict().items() if "lora" in k}
    torch.save(lora_state, "/tmp/test_lora.pt")
    model2 = build_small_model()
    inject_lora(model2, rank=4)
    model2.load_state_dict(torch.load("/tmp/test_lora.pt"), strict=False)
    for k in lora_state:
        assert torch.allclose(model.state_dict()[k], model2.state_dict()[k])
```

---

## Performance Considerations

- **Rank selection**: rank 4 is sufficient for narrow tasks (single language). Rank 16
  for broader adaptation. Rank 32+ rarely outperforms rank 16 but costs more.
- **Alpha/rank ratio**: keeping `alpha = 2 * rank` is a common baseline. Higher alpha
  means a stronger adapter signal (can cause instability if too high).
- **Target modules**: injecting only `q_proj` + `v_proj` (PEFT default) is faster and
  usually matches injecting all 4 attention projections for code tasks.
- **Gradient checkpointing**: combine with `model.gradient_checkpointing_enable()` for
  long sequence fine-tuning.
- **Learning rate**: 1e-4 to 5e-4 works well; much higher than full fine-tune LR because
  adapters are randomly initialized and need faster convergence.
- **Merge before export**: always merge before ONNX export or quantization to avoid
  handling LoRALinear as a custom op.

---

## Dependencies

```
bitsandbytes>=0.43.0   # QLoRA quantization (optional, only for --qlora)
torch>=2.2.0           # base requirement already satisfied
peft>=0.10.0           # optional reference implementation, not required
```

Install extras:
```bash
pip install bitsandbytes  # only needed for QLoRA path
```

---

## Estimated Complexity

| Aspect | Estimate |
|---|---|
| Core LoRALinear class | 2 hours |
| Injection + freezing | 1 hour |
| Training loop integration | 2 hours |
| Save/load adapters | 1 hour |
| QLoRA (bitsandbytes) | 3 hours |
| CLI integration | 1 hour |
| Tests | 2 hours |
| **Total** | **~12 hours** |

Complexity rating: **Medium** — well-understood technique, straightforward to implement
from scratch without using PEFT.

---

## 2026 Best Practices

- **DoRA (Weight-Decomposed LoRA)**: Decompose weight into magnitude + direction, apply
  LoRA only to direction component. Reported +1-2% quality over standard LoRA with same
  rank. Worth trying if LoRA quality is not sufficient.
- **LoRA+**: Use different learning rates for `A` (lower) and `B` (higher) matrices;
  shown to converge faster.
- **rsLoRA**: Scale by `1/sqrt(rank)` instead of `1/rank` for stable training at high
  ranks.
- **Adapter merging (DARE/TIES)**: If you have multiple domain adapters, merge them
  into one model without catastrophic interference using task vector arithmetic.
- **Flash Attention compatibility**: Ensure LoRALinear works with flash-attn 2.x — the
  forward pass signature must match `nn.Linear` exactly after merging.
- **Quantization-aware LoRA**: When using QLoRA, keep adapter matrices in bf16 even
  though base is NF4; bitsandbytes handles this automatically with `compute_dtype`.
