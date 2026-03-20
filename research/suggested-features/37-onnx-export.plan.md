# Feature 37: ONNX Export

## Overview

ONNX (Open Neural Network Exchange) is a standard format for representing neural network
models. Exporting Cola-Coder to ONNX enables:
- CPU inference without PyTorch installed
- Runtime optimization via ONNXRuntime (graph fusion, quantization)
- Deployment to edge devices, mobile, or cloud services that speak ONNX
- Potential integration with ONNX-aware serving stacks

Status: OPTIONAL — enable via `--feature onnx-export` or CLI menu toggle.

---

## Motivation

- A user wanting to use Cola-Coder without a full PyTorch environment can run the ONNX
  model with only `onnxruntime` installed (~50 MB vs ~2 GB for PyTorch).
- ONNXRuntime applies automatic graph optimizations (op fusion, constant folding) that
  often give 10–30% CPU speedup over PyTorch eager mode.
- INT8 quantization via ONNXRuntime further reduces model size and speeds up CPU inference.
- Server deployment: ONNX models can be served with Triton Inference Server, which
  handles batching, scheduling, and multi-GPU routing automatically.

---

## Architecture / Design

### Export Flow

```
PyTorch nn.Module
       |
       | torch.onnx.export()
       v
ONNX graph (model.onnx)
       |
       | onnxruntime GraphOptimizationLevel.ORT_ENABLE_ALL
       v
Optimized ONNX (model_opt.onnx)
       |
       | onnxruntime quantization (optional)
       v
INT8 ONNX (model_int8.onnx)
```

### Key Challenges

1. **Dynamic axes**: batch size and sequence length vary at inference time. Must be
   declared as dynamic axes in `torch.onnx.export`.
2. **Custom ops**: RoPE (using `torch.polar` or complex numbers) and SwiGLU may not
   have native ONNX equivalents. Decompose into basic ops.
3. **KV-cache**: ONNX does not natively support stateful loops. Two options:
   - Export the model without KV-cache (full sequence each call, slower)
   - Export with KV-cache as explicit I/O tensors (complex but fast)

### Export Script

```python
# cola_coder/export/onnx_export.py

import torch
import torch.onnx
from pathlib import Path


def prepare_model_for_onnx(model: torch.nn.Module) -> torch.nn.Module:
    """
    Patch the model to remove constructs that don't export cleanly.
    - Replace complex-number RoPE with real-valued equivalent
    - Ensure forward() takes (input_ids, attention_mask) not **kwargs
    """
    model.eval()
    # Disable dropout — ONNX export does not handle training-mode randomness
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    return model


class OnnxWrapper(torch.nn.Module):
    """Wraps the model with a clean signature for ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,        # (B, T)
        attention_mask: torch.Tensor,   # (B, T)
    ) -> torch.Tensor:                  # (B, T, vocab_size)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits


def export_to_onnx(
    model: torch.nn.Module,
    tokenizer,
    output_path: Path,
    opset_version: int = 17,
    verify: bool = True,
    max_seq_len: int = 256,
) -> Path:
    """
    Export model to ONNX format with dynamic batch and sequence axes.

    Args:
        model: trained Cola-Coder model
        tokenizer: corresponding tokenizer
        output_path: path to write .onnx file
        opset_version: ONNX opset (17 = 2022+, supports most ops)
        verify: if True, compare ONNX output against PyTorch
        max_seq_len: used to create dummy input for shape inference

    Returns:
        path to exported ONNX file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = prepare_model_for_onnx(model)
    wrapper = OnnxWrapper(model)

    # Create dummy inputs for shape tracing
    dummy_text = "def hello_world():\n    "
    dummy_ids = tokenizer.encode(dummy_text, return_tensors="pt")
    B, T = dummy_ids.shape
    dummy_mask = torch.ones_like(dummy_ids)

    # Dynamic axes: batch and sequence length can vary
    dynamic_axes = {
        "input_ids":      {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits":         {0: "batch_size", 1: "sequence_length"},
    }

    print(f"Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        wrapper,
        args=(dummy_ids, dummy_mask),
        f=str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )
    print(f"Exported to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    if verify:
        verify_onnx_output(wrapper, dummy_ids, dummy_mask, str(output_path))

    return output_path


def verify_onnx_output(
    pytorch_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    onnx_path: str,
    atol: float = 1e-4,
) -> bool:
    """Compare PyTorch and ONNX outputs to ensure correctness."""
    import onnxruntime as ort
    import numpy as np

    # PyTorch reference
    with torch.no_grad():
        pt_out = pytorch_model(input_ids, attention_mask).numpy()

    # ONNX output
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids":      input_ids.numpy().astype(np.int64),
        "attention_mask": attention_mask.numpy().astype(np.int64),
    }
    ort_out = session.run(["logits"], ort_inputs)[0]

    max_diff = np.abs(pt_out - ort_out).max()
    passed = max_diff < atol
    print(f"ONNX verification: max diff = {max_diff:.2e} — {'PASS' if passed else 'FAIL'}")
    return passed
```

### ONNX Optimization Pass

```python
# cola_coder/export/onnx_optimize.py

from pathlib import Path


def optimize_onnx(
    input_path: Path,
    output_path: Path,
    optimization_level: str = "all",   # "basic" | "extended" | "all"
) -> Path:
    """Apply ONNXRuntime graph optimizations."""
    import onnxruntime as ort

    level_map = {
        "none":     ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic":    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all":      ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = level_map[optimization_level]
    sess_options.optimized_model_filepath = str(output_path)

    # Running a session with the optimized path set writes the optimized graph
    ort.InferenceSession(
        str(input_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    print(f"Optimized ONNX written to {output_path}")
    return output_path


def quantize_onnx_int8(
    input_path: Path,
    output_path: Path,
) -> Path:
    """Quantize all linear layers to INT8 (dynamic quantization)."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        per_channel=False,     # True gives better quality but needs calibration
    )
    orig_size = input_path.stat().st_size / 1e6
    quant_size = output_path.stat().st_size / 1e6
    print(f"Quantized: {orig_size:.1f} MB -> {quant_size:.1f} MB "
          f"({100*(1-quant_size/orig_size):.0f}% reduction)")
    return output_path
```

### ONNX Runtime Inference Wrapper

```python
# cola_coder/export/onnx_runner.py

import numpy as np
import torch
from pathlib import Path


class OnnxCodeGenerator:
    """
    Drop-in replacement for CodeGenerator using ONNX Runtime for inference.
    Does NOT require PyTorch for forward pass (only for sampling utilities).
    """

    def __init__(
        self,
        onnx_path: Path,
        tokenizer,
        device: str = "cpu",   # ONNX runs well on CPU; GPU requires CUDA EP
    ):
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=providers,
        )
        self.tokenizer = tokenizer
        self.device = device

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        input_ids = self.tokenizer.encode(prompt)
        eos_id = self.tokenizer.eos_token_id
        generated = []

        for _ in range(max_new_tokens):
            ids_np = np.array([input_ids + generated], dtype=np.int64)
            mask_np = np.ones_like(ids_np, dtype=np.int64)

            logits = self.session.run(
                ["logits"],
                {"input_ids": ids_np, "attention_mask": mask_np},
            )[0]  # (1, T, V)

            next_logits = logits[0, -1, :]   # (V,)
            next_token = self._sample(next_logits, temperature, top_k, top_p)
            generated.append(next_token)

            if next_token == eos_id:
                break

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _sample(self, logits: np.ndarray, temp: float, top_k: int, top_p: float) -> int:
        # Apply temperature
        logits = logits / max(temp, 1e-8)
        # Top-k
        if top_k > 0:
            top_k_vals = np.partition(logits, -top_k)[-top_k]
            logits = np.where(logits < top_k_vals, -np.inf, logits)
        # Softmax
        exp_l = np.exp(logits - logits.max())
        probs = exp_l / exp_l.sum()
        # Top-p
        if top_p < 1.0:
            sorted_idx = np.argsort(-probs)
            cum_probs = np.cumsum(probs[sorted_idx])
            cutoff = sorted_idx[cum_probs > top_p]
            if len(cutoff):
                probs[cutoff[1:]] = 0  # keep first token above cutoff
                probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs))
```

---

## Implementation Steps

1. **Create `cola_coder/export/` package**: `__init__.py`, `onnx_export.py`,
   `onnx_optimize.py`, `onnx_runner.py`.

2. **Add CLI command**: "Export to ONNX" → shows options (opset, quantize, verify).

3. **Handle RoPE custom ops**: use only real-valued RoPE (sin/cos approach), not
   `torch.view_as_complex`. The complex number path does not export cleanly to ONNX < 18.

4. **Handle SwiGLU**: ensure `gate * silu(x)` is decomposed into standard ops
   (`Mul`, `Sigmoid`, `Mul`) rather than a fused custom kernel.

5. **Test export with small model**: use `n_layers=2, d_model=64` to validate the
   export pipeline before running on full model.

6. **Add `--backend onnx` flag** to server.py so it can serve via OnnxCodeGenerator.

7. **Benchmark**: compare PyTorch vs ONNX vs ONNX+INT8 on latency and throughput.

8. **Document deployment**: add a section to README-style help about CPU-only deployment.

---

## Key Files to Modify

| File | Change |
|---|---|
| `cola_coder/rope.py` | Ensure real-valued only (no complex ONNX issues) |
| `cli/menu.py` | Add "Export to ONNX" menu option |
| `server.py` | Add `--backend onnx` startup option |
| `config.py` | Add `ExportConfig` |
| `cola_coder/export/` | New package (all new files) |
| `requirements.txt` | Add `onnx`, `onnxruntime` as optional deps |

---

## Testing Strategy

```python
# tests/test_onnx_export.py

def test_export_produces_valid_onnx():
    import onnx
    model = build_tiny_model()
    tokenizer = build_test_tokenizer()
    out_path = Path("/tmp/test_cola.onnx")
    export_to_onnx(model, tokenizer, out_path, verify=False)
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)  # raises if invalid

def test_onnx_output_matches_pytorch():
    model = build_tiny_model()
    tokenizer = build_test_tokenizer()
    out_path = Path("/tmp/test_cola_verify.onnx")
    passed = export_to_onnx(model, tokenizer, out_path, verify=True)
    assert passed

def test_onnx_runner_generates_text():
    model = build_tiny_model()
    tokenizer = build_test_tokenizer()
    out_path = Path("/tmp/test_runner.onnx")
    export_to_onnx(model, tokenizer, out_path, verify=False)
    runner = OnnxCodeGenerator(out_path, tokenizer)
    result = runner.generate("def hello():", max_new_tokens=10)
    assert isinstance(result, str) and len(result) > 0

def test_onnx_quantization_reduces_size():
    out_path = Path("/tmp/test_cola.onnx")
    quant_path = Path("/tmp/test_cola_int8.onnx")
    # Assumes test_cola.onnx already exported
    quantize_onnx_int8(out_path, quant_path)
    assert quant_path.stat().st_size < out_path.stat().st_size * 0.8
```

---

## Performance Considerations

- **No KV-cache in ONNX export**: the simple export runs the full sequence each step
  (O(T^2) attention). For short generations this is fine; for 256+ tokens, latency
  compounds. Consider exporting with explicit KV-cache tensors as I/O.
- **ONNX CPU vs GPU**: ONNX CPU inference is often faster than PyTorch CPU due to
  multi-threading and op fusion. For GPU, the CUDA EP often matches PyTorch performance.
- **Static vs dynamic shapes**: static shapes allow more aggressive optimization but
  require separate ONNX files per input shape. Dynamic axes are more flexible.
- **Model size**: a 300 M parameter model in fp32 = ~1.2 GB ONNX file. Export in fp16
  (requires ONNX opset 15+ and mixed-precision support) to halve this.
- **Thread count**: set `intra_op_num_threads` for optimal CPU performance:
  ```python
  sess_options.intra_op_num_threads = os.cpu_count() // 2
  ```

---

## Dependencies

```
onnx>=1.16.0           # ONNX graph format
onnxruntime>=1.18.0    # CPU/GPU inference runtime
# onnxruntime-gpu      # GPU variant (mutually exclusive with onnxruntime)
```

---

## Estimated Complexity

| Aspect | Estimate |
|---|---|
| Basic export script | 2 hours |
| Dynamic axes + verification | 2 hours |
| RoPE/SwiGLU compat fixes | 3 hours |
| Optimization pass | 1 hour |
| INT8 quantization | 1 hour |
| OnnxCodeGenerator inference wrapper | 3 hours |
| CLI integration | 1 hour |
| Tests | 2 hours |
| **Total** | **~15 hours** |

Complexity rating: **Medium-Hard** — export script is straightforward, but custom ops
(RoPE, SwiGLU, possible flash attention) require careful treatment.

---

## 2026 Best Practices

- **ONNX opset 18+**: supports FP8 quantization and new transformer ops (GroupNorm, etc.).
  Use opset 17 for maximum compatibility, 18+ for edge cases.
- **ExecuTorch**: Meta's mobile deployment framework that extends PyTorch export beyond
  ONNX. If targeting mobile/embedded, ExecuTorch is increasingly preferred in 2026.
- **TensorRT via ONNX**: on NVIDIA GPUs, compile ONNX to TensorRT engine for 2–3× speedup
  over ONNX CUDA EP. Requires `tensorrt` installation and rebuilding on each GPU arch.
- **Olive optimization pipeline**: Microsoft's `olive` tool automates ONNX export +
  optimization + quantization in one step with quality-aware search.
- **KV-cache ONNX pattern**: ONNX Runtime 1.18+ has a built-in GroupQueryAttention op
  with KV-cache I/O support, enabling stateful ONNX transformer inference.
