"""Post-training quantization for cola-coder models.

Provides two quantization approaches:

1. Dynamic INT8 quantization (torch.ao.quantization)
   - Quantizes Linear layer weights to INT8 at runtime
   - Works on CPU, no GPU needed
   - ~2× memory reduction, ~1.5–2× speedup on CPU
   - Zero accuracy loss on most tasks (weights are de-quantized before compute)

2. Weight-only quantization (INT4 or INT8)
   - Weights stored as INT4/INT8, de-quantized for each matmul
   - Better memory reduction for INT4 (~4× vs FP32)
   - Uses bitsandbytes if available, falls back to torch.ao

For a TS dev analogy:
   Dynamic quant ≈ "compress the bundle at deploy time, decompress per request"
   Weight-only    ≈ "store values as smaller numbers but compute in full precision"
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Optional: bitsandbytes for advanced quantization
try:
    import bitsandbytes as bnb  # type: ignore
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    logger.debug("bitsandbytes not found — falling back to torch.ao for weight-only quant")


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QuantResult:
    """Summary of a quantization run."""
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    method: str
    backend: str = "torch.ao"


# ──────────────────────────────────────────────────────────────────────────────
# Size helper
# ──────────────────────────────────────────────────────────────────────────────

def _model_size_mb(model: nn.Module) -> float:
    """Estimate model size in MB by summing parameter and buffer bytes."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    for b in model.buffers():
        total_bytes += b.nelement() * b.element_size()
    return total_bytes / (1024 * 1024)


# ──────────────────────────────────────────────────────────────────────────────
# Weight-only INT4 fallback (pure PyTorch, no external deps)
# ──────────────────────────────────────────────────────────────────────────────

class _Int4Linear(nn.Module):
    """Linear layer with INT4 weight-only quantization.

    Weights are packed as uint8 (two INT4 values per byte) and de-quantized
    to float32 on the fly during the forward pass.  Biases (if any) stay FP32.
    """

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        out_features, in_features = weight.shape

        # Per-row scales (one scale per output neuron)
        abs_max = weight.abs().max(dim=1).values.clamp(min=1e-8)
        scale = abs_max / 7.0  # INT4 symmetric range: -8..7, use -7..7

        # Quantize and clamp to INT4 symmetric range
        w_int = (weight / scale.unsqueeze(1)).round().clamp(-7, 7).to(torch.int8)

        # Pack two INT4 values per byte (lower nibble = even cols, upper = odd cols)
        # Pad in_features to even
        if in_features % 2 != 0:
            w_int = torch.cat(
                [w_int, torch.zeros(out_features, 1, dtype=torch.int8)], dim=1
            )
        # Offset to unsigned [0, 15] for nibble packing
        w_uint = (w_int + 8).to(torch.uint8)
        packed = (w_uint[:, 0::2] | (w_uint[:, 1::2] << 4))

        self.register_buffer("weight_packed", packed)
        self.register_buffer("scale", scale.float())
        if bias is not None:
            self.register_buffer("bias", bias.clone().float())
        else:
            self.bias = None

        self.out_features = out_features
        self.in_features = in_features

    def _dequantize(self) -> torch.Tensor:
        """Unpack and dequantize weights back to float32."""
        lo = (self.weight_packed & 0x0F).to(torch.int8) - 8
        hi = ((self.weight_packed >> 4) & 0x0F).to(torch.int8) - 8

        # Interleave: even cols from lo, odd cols from hi
        out_f, pack_cols = lo.shape
        w = torch.empty(out_f, pack_cols * 2, dtype=torch.int8, device=lo.device)
        w[:, 0::2] = lo
        w[:, 1::2] = hi

        # Trim padding if in_features was odd
        w = w[:, : self.in_features]
        return w.float() * self.scale.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequantize()
        return nn.functional.linear(x, w, self.bias)


# ──────────────────────────────────────────────────────────────────────────────
# Main quantizer
# ──────────────────────────────────────────────────────────────────────────────

class ModelQuantizer:
    """Post-training quantization for faster inference.

    Example usage:
        quantizer = ModelQuantizer(model)

        # Dynamic INT8 (best for CPU, no data needed)
        q_model, result = quantizer.quantize_dynamic()

        # Weight-only INT4 (smaller files, good for memory-constrained inference)
        q4_model, result = quantizer.quantize_weights_only(bits=4)

        # Compare performance
        stats = quantizer.benchmark(model, q_model, ["def hello():"])
    """

    def __init__(self, model: nn.Module):
        self.model = model

    # ── Public quantization methods ───────────────────────────────────────────

    def quantize_dynamic(self) -> tuple[nn.Module, QuantResult]:
        """Dynamic INT8 quantization using torch.ao.quantization.

        Quantizes all Linear layers to INT8.  Works on CPU without any
        calibration data.  Weights are stored as INT8 and de-quantized
        to float at runtime.

        Returns:
            (quantized_model, QuantResult)
        """
        original_size = _model_size_mb(self.model)
        q_model = copy.deepcopy(self.model).cpu()

        # torch.ao.quantization.quantize_dynamic is available in PyTorch 2.x.
        # Note: PyTorch 2.10+ deprecates torch.ao in favour of torchao.
        # The API still works; suppress the deprecation warning here so tests
        # don't emit noise.  If torchao is installed a future version of this
        # function should migrate to torchao.quantization.quantize_.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message="torch.ao.quantization is deprecated",
            )
            q_model = torch.ao.quantization.quantize_dynamic(
                q_model,
                {nn.Linear},
                dtype=torch.qint8,
            )
        q_model.eval()

        quantized_size = _model_size_mb(q_model)
        ratio = original_size / max(quantized_size, 1e-6)

        return q_model, QuantResult(
            original_size_mb=round(original_size, 2),
            quantized_size_mb=round(quantized_size, 2),
            compression_ratio=round(ratio, 2),
            method="dynamic_int8",
            backend="torch.ao",
        )

    def quantize_weights_only(self, bits: int = 4) -> tuple[nn.Module, QuantResult]:
        """Weight-only quantization (INT4 or INT8).

        For INT8: uses torch.ao.quantization.quantize_dynamic (same as
        quantize_dynamic but emphasised as weight-only).
        For INT4: uses bitsandbytes Linear4bit if available, otherwise falls
        back to the built-in _Int4Linear implementation.

        Args:
            bits: 4 or 8.

        Returns:
            (quantized_model, QuantResult)
        """
        if bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {bits}")

        original_size = _model_size_mb(self.model)

        if bits == 8:
            return self.quantize_dynamic()

        # INT4
        if BNB_AVAILABLE:
            q_model, backend = self._quantize_int4_bnb()
        else:
            q_model, backend = self._quantize_int4_builtin()

        q_model.eval()
        quantized_size = _model_size_mb(q_model)
        ratio = original_size / max(quantized_size, 1e-6)

        return q_model, QuantResult(
            original_size_mb=round(original_size, 2),
            quantized_size_mb=round(quantized_size, 2),
            compression_ratio=round(ratio, 2),
            method="weight_only_int4",
            backend=backend,
        )

    def benchmark(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        prompts: list[str],
        device: str = "cpu",
        max_new_tokens: int = 20,
    ) -> dict:
        """Compare original vs quantized model: speed, memory, output similarity.

        Does NOT require a tokenizer — runs forward passes on random token IDs
        to measure latency and checks that output logit distributions are similar.

        Args:
            original_model: The unquantized model.
            quantized_model: The quantized model.
            prompts: List of prompt strings (used only to determine batch count;
                     actual token IDs are random since no tokenizer is available here).
            device: "cpu" or "cuda".
            max_new_tokens: Number of forward passes per prompt.

        Returns:
            Dict with keys: original_ms, quantized_ms, speedup,
                            original_size_mb, quantized_size_mb,
                            compression_ratio, logit_cosine_sim.
        """
        original_model = original_model.to(device).eval()
        quantized_model = quantized_model.to(device).eval()

        # Determine model vocab size and seq len from config if available
        vocab_size = 32768
        seq_len = 16
        if hasattr(original_model, "config"):
            vocab_size = original_model.config.vocab_size
            seq_len = min(16, original_model.config.max_seq_len)

        n_batches = max(1, len(prompts))

        def _run_forward(model: nn.Module) -> tuple[float, torch.Tensor]:
            torch.manual_seed(42)
            token_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_batches):
                    logits = model(token_ids)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return elapsed_ms, logits

        orig_ms, orig_logits = _run_forward(original_model)
        quant_ms, quant_logits = _run_forward(quantized_model)

        # Cosine similarity between last-position logits
        o = orig_logits[0, -1].float()
        q = quant_logits[0, -1].float()
        cos_sim = float(
            torch.nn.functional.cosine_similarity(o.unsqueeze(0), q.unsqueeze(0)).item()
        )

        orig_size = _model_size_mb(original_model)
        quant_size = _model_size_mb(quantized_model)
        speedup = orig_ms / max(quant_ms, 1e-6)

        return {
            "original_ms": round(orig_ms, 2),
            "quantized_ms": round(quant_ms, 2),
            "speedup": round(speedup, 3),
            "original_size_mb": round(orig_size, 2),
            "quantized_size_mb": round(quant_size, 2),
            "compression_ratio": round(orig_size / max(quant_size, 1e-6), 3),
            "logit_cosine_sim": round(cos_sim, 4),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _quantize_int4_bnb(self) -> tuple[nn.Module, str]:
        """Quantize to INT4 using bitsandbytes."""
        q_model = copy.deepcopy(self.model).cpu()
        for name, module in list(q_model.named_modules()):
            if isinstance(module, nn.Linear):
                parent_name, _, child_name = name.rpartition(".")
                parent = q_model if not parent_name else _get_nested_module(q_model, parent_name)
                new_layer = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    quant_type="nf4",
                )
                new_layer.weight = bnb.nn.Params4bit(
                    module.weight.data, requires_grad=False, quant_type="nf4"
                )
                if module.bias is not None:
                    new_layer.bias = nn.Parameter(module.bias.data.clone())
                setattr(parent, child_name, new_layer)
        return q_model, "bitsandbytes"

    def _quantize_int4_builtin(self) -> tuple[nn.Module, str]:
        """Quantize to INT4 using the built-in _Int4Linear implementation."""
        q_model = copy.deepcopy(self.model).cpu()
        for name, module in list(q_model.named_modules()):
            if isinstance(module, nn.Linear):
                parent_name, _, child_name = name.rpartition(".")
                parent = q_model if not parent_name else _get_nested_module(q_model, parent_name)
                new_layer = _Int4Linear(
                    module.weight.data.float(),
                    module.bias.data.float() if module.bias is not None else None,
                )
                setattr(parent, child_name, new_layer)
        return q_model, "torch_builtin_int4"


def _get_nested_module(model: nn.Module, dotted_name: str) -> nn.Module:
    """Resolve a dotted module path, e.g. 'blocks.0.attention'."""
    parts = dotted_name.split(".")
    m = model
    for p in parts:
        m = getattr(m, p)
    return m
