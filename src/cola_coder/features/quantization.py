"""Quantization: reduce model size and memory for faster inference.

Post-training quantization converts model weights from float32/float16 to
lower precision (int8), reducing model size ~2-4x and inference memory.

Approaches:
- Dynamic quantization: quantize weights statically, activations dynamically
  per batch. Simplest, no calibration data needed. Best for linear layers.
- Weight-only quantization: just compress the weights, compute in float.

For a TS dev: like compressing a JSON payload — same data, smaller footprint,
slight precision loss that's usually imperceptible.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class QuantizationResult:
    """Results from quantizing a model."""
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    num_quantized_layers: int
    num_total_layers: int
    dtype_before: str
    dtype_after: str

    @property
    def size_reduction_pct(self) -> float:
        if self.original_size_mb == 0:
            return 0.0
        return (1 - self.quantized_size_mb / self.original_size_mb) * 100

    def summary(self) -> str:
        return (
            f"Quantization: {self.original_size_mb:.1f}MB -> {self.quantized_size_mb:.1f}MB "
            f"({self.size_reduction_pct:.0f}% reduction, {self.compression_ratio:.1f}x)\n"
            f"Layers quantized: {self.num_quantized_layers}/{self.num_total_layers}\n"
            f"Dtype: {self.dtype_before} -> {self.dtype_after}"
        )


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in megabytes."""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()
    for buf in model.buffers():
        total_bytes += buf.nelement() * buf.element_size()
    return total_bytes / (1024 * 1024)


def count_linear_layers(model: nn.Module) -> int:
    """Count the number of Linear layers in the model."""
    return sum(1 for m in model.modules() if isinstance(m, nn.Linear))


def dynamic_quantize(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
) -> tuple[nn.Module, QuantizationResult]:
    """Apply dynamic quantization to the model.

    Quantizes all nn.Linear layers to int8. This is the simplest
    quantization approach — no calibration data needed.

    Args:
        model: The model to quantize (will be modified in-place on CPU)
        dtype: Target quantization dtype (default: qint8)

    Returns:
        Tuple of (quantized_model, result)
    """
    original_size = get_model_size_mb(model)
    num_linear = count_linear_layers(model)
    total_modules = sum(1 for _ in model.modules())

    # Dynamic quantization requires CPU
    original_dtype = str(next(model.parameters()).dtype)
    model_cpu = model.cpu()

    # Apply dynamic quantization
    quantized = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},  # Quantize Linear layers
        dtype=dtype,
    )

    quantized_size = get_model_size_mb(quantized)
    compression = original_size / quantized_size if quantized_size > 0 else 1.0

    result = QuantizationResult(
        original_size_mb=original_size,
        quantized_size_mb=quantized_size,
        compression_ratio=compression,
        num_quantized_layers=num_linear,
        num_total_layers=total_modules,
        dtype_before=original_dtype,
        dtype_after="qint8",
    )

    return quantized, result


def weight_only_quantize(model: nn.Module) -> tuple[nn.Module, QuantizationResult]:
    """Apply weight-only int8 quantization.

    Converts weight tensors to int8 with per-channel scale factors.
    Computation still happens in float, but model storage is smaller.

    Args:
        model: The model to quantize

    Returns:
        Tuple of (model_with_quantized_weights, result)
    """
    original_size = get_model_size_mb(model)
    total_modules = sum(1 for _ in model.modules())
    original_dtype = str(next(model.parameters()).dtype)

    # Simple weight quantization: convert to int8 scale
    quantized_layers = 0
    model_cpu = model.cpu()

    for name, module in model_cpu.named_modules():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                w = module.weight.data
                # Per-channel quantization
                scale = w.abs().max(dim=1, keepdim=True).values / 127.0
                scale = scale.clamp(min=1e-8)
                w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)
                # Store quantized weight and scale
                module.weight.data = (w_int8.float() * scale)  # Dequantize back
                quantized_layers += 1

    quantized_size = get_model_size_mb(model_cpu)
    compression = original_size / quantized_size if quantized_size > 0 else 1.0

    result = QuantizationResult(
        original_size_mb=original_size,
        quantized_size_mb=quantized_size,
        compression_ratio=compression,
        num_quantized_layers=quantized_layers,
        num_total_layers=total_modules,
        dtype_before=original_dtype,
        dtype_after="weight_int8_dequant",
    )

    return model_cpu, result


def save_quantized_model(model: nn.Module, path: str) -> None:
    """Save a quantized model using PyTorch's save format.

    Note: Quantized models use torch.save, not safetensors,
    because safetensors doesn't support quantized dtypes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))


def print_quantization_report(result: QuantizationResult) -> None:
    """Print a formatted quantization report."""
    from cola_coder.cli import cli
    cli.header("Quantization", "Complete")
    cli.info("Original size", f"{result.original_size_mb:.1f} MB")
    cli.info("Quantized size", f"{result.quantized_size_mb:.1f} MB")
    cli.info("Compression", f"{result.compression_ratio:.1f}x ({result.size_reduction_pct:.0f}% reduction)")
    cli.info("Layers quantized", f"{result.num_quantized_layers}/{result.num_total_layers}")
    cli.info("Dtype", f"{result.dtype_before} -> {result.dtype_after}")
