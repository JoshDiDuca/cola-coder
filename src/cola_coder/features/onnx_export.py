"""ONNX Export: convert model to ONNX format for cross-platform deployment.

Exports the trained PyTorch model to ONNX format, which can then be run
with ONNX Runtime, TensorRT, or other inference engines for deployment
without PyTorch.

For a TS dev: like compiling TypeScript to JavaScript — the ONNX format
is a "compiled" neural network that runs on any compatible runtime.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


def check_onnx_available() -> bool:
    """Check if ONNX and ONNX Runtime are available."""
    try:
        import onnx  # noqa: F401
        return True
    except ImportError:
        return False


def check_onnxruntime_available() -> bool:
    """Check if ONNX Runtime is available."""
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class ExportResult:
    """Results from an ONNX export."""
    output_path: str
    file_size_mb: float
    opset_version: int
    input_names: list[str]
    output_names: list[str]
    dynamic_axes: dict
    success: bool
    error: str = ""

    def summary(self) -> str:
        if not self.success:
            return f"Export FAILED: {self.error}"
        return (
            f"ONNX Export: {self.output_path}\n"
            f"  Size: {self.file_size_mb:.1f} MB\n"
            f"  Opset: {self.opset_version}\n"
            f"  Inputs: {', '.join(self.input_names)}\n"
            f"  Outputs: {', '.join(self.output_names)}"
        )


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    max_seq_len: int = 1024,
    vocab_size: int = 32768,
    opset_version: int = 17,
    batch_size: int = 1,
) -> ExportResult:
    """Export a model to ONNX format.

    Args:
        model: The PyTorch model to export
        output_path: Where to save the .onnx file
        max_seq_len: Maximum sequence length for the dummy input
        vocab_size: Vocabulary size (for input range validation)
        opset_version: ONNX opset version (17 is well-supported in 2026)
        batch_size: Batch size for the dummy input

    Returns:
        ExportResult with export details
    """
    output_path = str(Path(output_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    input_names = ["input_ids"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size", 1: "seq_len"},
    }

    try:
        # Create dummy input
        dummy_input = torch.randint(0, vocab_size, (batch_size, 64), dtype=torch.long)

        # Move model to CPU for export
        model_cpu = model.cpu().eval()

        # Export
        torch.onnx.export(
            model_cpu,
            (dummy_input,),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )

        file_size = Path(output_path).stat().st_size / (1024 * 1024)

        return ExportResult(
            output_path=output_path,
            file_size_mb=file_size,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            success=True,
        )

    except Exception as e:
        return ExportResult(
            output_path=output_path,
            file_size_mb=0.0,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            success=False,
            error=str(e),
        )


def validate_onnx_model(onnx_path: str) -> tuple[bool, str]:
    """Validate an exported ONNX model.

    Args:
        onnx_path: Path to the .onnx file

    Returns:
        Tuple of (is_valid, message)
    """
    if not check_onnx_available():
        return False, "onnx package not installed. Install with: pip install onnx"

    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        return True, "Model is valid"
    except Exception as e:
        return False, f"Validation failed: {e}"


def test_onnx_inference(
    onnx_path: str,
    seq_len: int = 32,
    vocab_size: int = 32768,
) -> tuple[bool, str]:
    """Test inference with ONNX Runtime.

    Args:
        onnx_path: Path to the .onnx file
        seq_len: Sequence length for test input
        vocab_size: Vocabulary size

    Returns:
        Tuple of (success, message)
    """
    if not check_onnxruntime_available():
        return False, "onnxruntime not installed. Install with: pip install onnxruntime"

    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(onnx_path)
        dummy_input = np.random.randint(0, vocab_size, (1, seq_len)).astype(np.int64)

        outputs = session.run(None, {"input_ids": dummy_input})
        logits = outputs[0]

        return True, f"Inference OK: output shape {logits.shape}"

    except Exception as e:
        return False, f"Inference failed: {e}"


def print_export_report(result: ExportResult) -> None:
    """Print a formatted export report."""
    from cola_coder.cli import cli

    if result.success:
        cli.header("ONNX Export", "Complete")
        cli.info("Output", result.output_path)
        cli.info("Size", f"{result.file_size_mb:.1f} MB")
        cli.info("Opset", result.opset_version)
        cli.info("Inputs", ", ".join(result.input_names))
        cli.info("Outputs", ", ".join(result.output_names))
        cli.success("Export successful!")
    else:
        cli.error(f"Export failed: {result.error}")
        cli.dim("Make sure the model is compatible with ONNX tracing")
