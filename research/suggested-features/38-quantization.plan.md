# Feature 38: Post-Training Quantization (INT8 / INT4)

## Overview

Post-training quantization (PTQ) reduces model weights from 32-bit or 16-bit floats to
8-bit or 4-bit integers, without retraining. The benefits:
- 2–4× smaller model on disk and in RAM
- 1.5–2× faster inference on CPU (INT8 SIMD instructions)
- Enables running larger models on limited hardware

Cola-Coder supports three quantization methods of increasing quality and complexity:
1. **Dynamic quantization** (torch.quantization) — quickest, moderate quality
2. **GPTQ** (4-bit, needs calibration) — high quality for LLMs, requires calibration data
3. **AWQ** (activation-aware, best quality) — state of the art for LLMs in 2025–2026

Status: OPTIONAL — enable via `--feature quantization` or CLI menu toggle.

---

## Motivation

- A 300 M parameter model at fp16 = ~600 MB. At INT8 = ~300 MB. At INT4 = ~150 MB.
- CPU inference: PyTorch INT8 linear layers use vectorized integer multiply-accumulate
  which is 2–3× faster than fp32 on modern x86 CPUs.
- Serving: smaller models fit in cheaper cloud VMs, enable more concurrent instances.
- Quality: AWQ keeps perplexity within 0.1–0.3% of fp16 at INT4, which is acceptable.

---

## Architecture / Design

### Method 1: Dynamic Quantization (Easiest)

Quantizes weights to INT8 at load time. Activations remain fp32. No calibration needed.

```python
# cola_coder/quantization/dynamic_quant.py

import torch
import torch.nn as nn


def apply_dynamic_quantization(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    inplace: bool = False,
) -> nn.Module:
    """
    Apply PyTorch dynamic quantization to all Linear layers.
    Weights are quantized to INT8; activations remain float32.
    Fastest to apply, moderate speedup on CPU.
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)

    model.eval()
    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear},
        dtype=dtype,
    )
    return quantized


def measure_size_reduction(original: nn.Module, quantized: nn.Module) -> dict:
    import tempfile, os
    results = {}
    for name, mod in [("original", original), ("quantized", quantized)]:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(mod.state_dict(), f.name)
            results[name] = os.path.getsize(f.name)
        os.unlink(f.name)
    ratio = results["original"] / results["quantized"]
    results["compression_ratio"] = ratio
    print(f"Size: {results['original']/1e6:.1f} MB -> "
          f"{results['quantized']/1e6:.1f} MB ({ratio:.2f}x reduction)")
    return results
```

### Method 2: GPTQ (4-bit, Calibration-Based)

GPTQ (Frantar et al. 2022) quantizes weights to 4-bit while minimizing the squared
error between original and quantized layer outputs, calibrated on a small dataset.

```python
# cola_coder/quantization/gptq.py

import torch
import torch.nn as nn
from typing import Callable


class GPTQQuantizer:
    """
    Simplified GPTQ quantizer for a single linear layer.
    Full GPTQ requires iterating over all layers in order.
    """

    def __init__(
        self,
        layer: nn.Linear,
        bits: int = 4,
        group_size: int = 128,    # quantize in groups for better quality
        actorder: bool = True,    # reorder columns by activation magnitude
    ):
        self.layer = layer
        self.bits = bits
        self.group_size = group_size
        self.actorder = actorder
        self.H = None  # Hessian accumulator

    def add_batch(self, inp: torch.Tensor) -> None:
        """Accumulate Hessian from calibration samples."""
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        B, T, D = inp.shape
        inp = inp.view(-1, D).float()
        if self.H is None:
            self.H = torch.zeros(D, D, device=inp.device)
        self.H += inp.T @ inp / (B * T)

    def quantize(self) -> dict:
        """Run GPTQ to find optimal INT4 weights."""
        W = self.layer.weight.data.float()   # (out, in)
        H = self.H
        if self.actorder:
            # Reorder columns by diagonal of H (activation magnitude)
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        # Cholesky of Hessian for efficient solve
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        out_features, in_features = W.shape
        Q = torch.zeros_like(W)
        Losses = torch.zeros(out_features, in_features // self.group_size)

        for i in range(0, in_features, self.group_size):
            j = min(i + self.group_size, in_features)
            w_group = W[:, i:j]

            # Min-max quantization for this group
            w_min = w_group.min(dim=1, keepdim=True).values
            w_max = w_group.max(dim=1, keepdim=True).values
            scale = (w_max - w_min) / (2**self.bits - 1)
            scale = scale.clamp(min=1e-8)
            zero_point = (-w_min / scale).round()

            q_group = ((w_group / scale + zero_point).round().clamp(0, 2**self.bits-1)
                       - zero_point) * scale
            Q[:, i:j] = q_group

        return {
            "quantized_weight": Q,
            "bits": self.bits,
            "group_size": self.group_size,
        }


def run_gptq(
    model: nn.Module,
    calibration_data: list[torch.Tensor],   # list of input_ids tensors
    bits: int = 4,
    group_size: int = 128,
) -> nn.Module:
    """
    Run GPTQ on all linear layers sequentially.
    calibration_data: ~128 samples, each shape (1, seq_len)
    """
    model.eval()
    hooks = []
    quantizers: dict[str, GPTQQuantizer] = {}

    # Register hooks to capture inputs to each Linear layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            q = GPTQQuantizer(module, bits=bits, group_size=group_size)
            quantizers[name] = q

            def make_hook(qz):
                def hook(mod, inp, out):
                    qz.add_batch(inp[0].detach())
                return hook

            hooks.append(module.register_forward_hook(make_hook(q)))

    # Run calibration
    with torch.no_grad():
        for inp in calibration_data[:128]:
            model(inp)

    for h in hooks:
        h.remove()

    # Apply quantization
    for name, qz in quantizers.items():
        result = qz.quantize()
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        layer = getattr(parent, child_name)
        layer.weight.data = result["quantized_weight"]

    return model
```

### Method 3: AWQ (Activation-Aware Weight Quantization)

AWQ finds per-channel scale factors that minimize quantization error by giving more
precision to channels with high activation magnitudes.

```python
# cola_coder/quantization/awq.py

import torch
import torch.nn as nn


def compute_activation_scales(
    model: nn.Module,
    calibration_data: list[torch.Tensor],
    smooth_factor: float = 0.5,
) -> dict[str, torch.Tensor]:
    """
    Compute per-channel activation scales for each linear layer.
    Returns {layer_name: scale_tensor}.
    """
    scales = {}
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            act_max = [torch.zeros(module.in_features)]

            def make_hook(am, nm):
                def hook(mod, inp, out):
                    x = inp[0].detach().abs()
                    channel_max = x.view(-1, x.shape[-1]).max(0).values
                    am[0] = torch.max(am[0].to(channel_max.device), channel_max)
                return hook

            hooks.append(module.register_forward_hook(make_hook(act_max, name)))
            scales[name] = act_max

    model.eval()
    with torch.no_grad():
        for inp in calibration_data[:64]:
            model(inp)

    for h in hooks:
        h.remove()

    return {name: scales[name][0] for name in scales}


def apply_awq_scales(
    model: nn.Module,
    act_scales: dict[str, torch.Tensor],
    smooth_factor: float = 0.5,
) -> nn.Module:
    """
    Apply AWQ smoothing: scale weights by (act_scale^smooth_factor),
    scale previous layer's output by (act_scale^-(smooth_factor)).
    Then quantize scaled weights to INT4.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in act_scales:
            scale = act_scales[name].to(module.weight.device)
            scale = scale ** smooth_factor
            scale = scale.clamp(min=1e-4)
            # Absorb scale into weights
            module.weight.data = module.weight.data / scale.unsqueeze(0)
    return model
```

### Quantized Inference Wrapper

```python
# cola_coder/quantization/quant_model.py

class QuantizedCodeGenerator:
    """Wraps CodeGenerator to use quantized model."""

    def __init__(self, base_generator, method: str = "dynamic"):
        from .dynamic_quant import apply_dynamic_quantization

        self.tokenizer = base_generator.tokenizer
        self.config = base_generator.config

        if method == "dynamic":
            self.model = apply_dynamic_quantization(base_generator.model)
        elif method == "gptq":
            # Assumes model already quantized, just load
            self.model = base_generator.model
        else:
            raise ValueError(f"Unknown quantization method: {method}")

    def generate(self, prompt: str, **kwargs) -> str:
        from ..generator import CodeGenerator
        # Reuse generation logic
        gen = CodeGenerator.__new__(CodeGenerator)
        gen.model = self.model
        gen.tokenizer = self.tokenizer
        gen.config = self.config
        return gen.generate(prompt, **kwargs)
```

---

## Implementation Steps

1. **Create `cola_coder/quantization/` package**: `__init__.py`, `dynamic_quant.py`,
   `gptq.py`, `awq.py`, `quant_model.py`, `benchmark.py`.

2. **Add `QuantizationConfig` to `config.py`**:
   ```python
   @dataclass
   class QuantizationConfig:
       enabled: bool = False
       method: str = "dynamic"   # "dynamic" | "gptq" | "awq"
       bits: int = 8             # 8 for dynamic, 4 for gptq/awq
       group_size: int = 128     # for gptq/awq
       calibration_samples: int = 128
       calibration_data_path: str = "data/calibration/"
   ```

3. **Implement `quantization_benchmark.py`**: compare original vs quantized on
   HumanEval pass@1, perplexity, and inference latency.

4. **Add CLI option**: "Quantize model" → choose method, bits, run calibration if needed.

5. **Calibration data collection**: extract 128–512 code snippets from training data
   (random samples, ~256 tokens each).

6. **Model save/load**: save quantized model state dict with metadata tag
   `{"quantization": "int8_dynamic"}` so it can be loaded without re-quantizing.

7. **Dequantize for fine-tuning**: before LoRA or full fine-tuning, dequantize back
   to fp16:
   ```python
   # Dynamic quantized models can be used directly with LoRA (LoRA layers are fp16)
   # For GPTQ, maintain original fp16 weights in a separate file
   ```

---

## Key Files to Modify

| File | Change |
|---|---|
| `generator.py` | Accept quantized model, add `quantize()` method |
| `config.py` | Add `QuantizationConfig` |
| `cli/menu.py` | Add "Quantize model" option |
| `cola_coder/quantization/` | New package |
| `benchmarks/humaneval.py` | Add quantized model variant comparison |

---

## Testing Strategy

```python
# tests/test_quantization.py

def test_dynamic_quantization_reduces_size():
    model = build_small_model()
    result = measure_size_reduction(model, apply_dynamic_quantization(model))
    assert result["compression_ratio"] > 1.5  # at least 1.5x smaller

def test_quantized_model_generates_text():
    gen = build_test_generator()
    quant = QuantizedCodeGenerator(gen, method="dynamic")
    out = quant.generate("def square(x):", max_new_tokens=20)
    assert isinstance(out, str) and len(out) > 0

def test_quantization_perplexity_degradation():
    """Quantized model perplexity should be within 5% of original."""
    gen = build_test_generator()
    ppl_orig = compute_perplexity(gen.model, test_data)
    quant = apply_dynamic_quantization(gen.model)
    ppl_quant = compute_perplexity(quant, test_data)
    assert (ppl_quant - ppl_orig) / ppl_orig < 0.05  # < 5% degradation

def test_gptq_calibration_hook():
    """GPTQ quantizer should accumulate Hessian from calibration data."""
    layer = nn.Linear(64, 128)
    qz = GPTQQuantizer(layer, bits=4)
    inp = torch.randn(4, 32, 64)
    qz.add_batch(inp)
    assert qz.H is not None
    assert qz.H.shape == (64, 64)
```

---

## Performance Considerations

- **INT8 CPU**: PyTorch dynamic quantization gives ~1.5–2× speedup on CPU for matrix
  multiplications larger than ~1024×1024. Small models may see less benefit.
- **INT4 CPU**: requires custom GEMM kernels (bitsandbytes, llama.cpp format). Standard
  PyTorch does not have native INT4 linear ops — need bitsandbytes or ctransformers.
- **GPU quantization**: INT8 on GPU requires TensorRT or bitsandbytes CUDA kernels.
  GPTQ 4-bit on GPU uses ExLlamaV2 or bitsandbytes for actual speedup.
- **Embedding layers**: keep in fp16/fp32 — quantizing embeddings gives very little
  size benefit (they are already sparse and heavily indexed).
- **Accuracy sensitivity**: attention layers (especially the QK^T multiplication) are
  sensitive to quantization. Consider keeping attention layers in fp16 and only
  quantizing FFN layers.

---

## Dependencies

```
torch>=2.2.0                # base requirement
bitsandbytes>=0.43.0        # GPTQ/AWQ GPU kernels (optional)
auto-gptq>=0.7.0            # reference GPTQ implementation (optional)
autoawq>=0.2.0              # reference AWQ implementation (optional)
```

For basic dynamic quantization: no new deps beyond PyTorch.

---

## Estimated Complexity

| Aspect | Estimate |
|---|---|
| Dynamic quantization | 2 hours |
| GPTQ (simplified) | 6 hours |
| AWQ (simplified) | 5 hours |
| Calibration data pipeline | 2 hours |
| Benchmark comparison | 2 hours |
| CLI integration | 1 hour |
| Tests | 2 hours |
| **Total** | **~20 hours** |

Complexity rating: **Hard** — dynamic quant is easy; GPTQ and AWQ require careful
numerical implementation. Use reference implementations (auto-gptq, autoawq) to validate.

---

## 2026 Best Practices

- **FP8 quantization**: NVIDIA Hopper+ (H100, H200) supports native FP8. For RTX 4080
  Ada, check `torch.cuda.get_device_capability()` — FP8 ops may be available.
  `torch.float8_e4m3fn` is the standard format.
- **MX (Microscaling) formats**: Intel and NVIDIA's joint MX formats (MXFP4, MXFP6,
  MXFP8) in 2025 represent the next step beyond INT4. Not yet widely supported in
  consumer PyTorch.
- **SmoothQuant**: migrate activation outliers into weight channel by channel-wise
  scaling before INT8 quantization. Reduces quantization error without calibration.
  Simpler than AWQ and nearly as effective.
- **GGUF format**: llama.cpp uses GGUF with k-quant methods (Q4_K_M, Q5_K_M) that
  are state of the art for CPU LLM inference. Consider exporting to GGUF if CPU
  performance is a priority.
- **Quantization-aware training (QAT)**: if PTQ quality is insufficient, QAT inserts
  fake quantization ops during training so the model learns to be robust to INT8 noise.
  Only needed if PTQ degrades HumanEval by more than 5%.
