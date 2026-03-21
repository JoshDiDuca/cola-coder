"""Pruning Analyzer: identify candidates for model compression.

Analyzes a saved model checkpoint's state dict (safetensors or plain dict) to
find:
- Dead neurons (weight rows/columns with near-zero L2 norm)
- Low-magnitude attention heads (entire head weight slice near zero)
- Estimated speedup from removing identified components

Works on checkpoint state dicts only — no live model, no GPU needed.

For a TS dev: like tree-shaking for neural networks — find the unused parts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the pruning analyzer feature is active."""
    return FEATURE_ENABLED


@dataclass
class DeadNeuronInfo:
    """A neuron (row in a weight matrix) with very low magnitude."""

    layer_name: str
    neuron_index: int
    l2_norm: float


@dataclass
class LowMagHeadInfo:
    """An attention head whose combined weight magnitude is very low."""

    layer_name: str
    head_index: int
    avg_magnitude: float


@dataclass
class PruningReport:
    """Results from analyzing a checkpoint for pruning candidates."""

    total_params: int = 0
    total_layers_analyzed: int = 0
    dead_neurons: list[DeadNeuronInfo] = field(default_factory=list)
    low_mag_heads: list[LowMagHeadInfo] = field(default_factory=list)
    prunable_param_fraction: float = 0.0
    estimated_speedup: float = 1.0  # multiplicative speedup estimate
    layer_magnitudes: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"params={self.total_params:,} layers={self.total_layers_analyzed} "
            f"dead_neurons={len(self.dead_neurons)} "
            f"low_mag_heads={len(self.low_mag_heads)} "
            f"prunable_frac={self.prunable_param_fraction:.2%} "
            f"est_speedup={self.estimated_speedup:.2f}x"
        )


class PruningAnalyzer:
    """Analyze a model state dict for prunable components.

    Accepts either a ``dict[str, Any]`` state dict (tensors as numpy arrays,
    lists, or objects with a ``tolist()`` method) or a path to a safetensors
    file.

    Usage::

        analyzer = PruningAnalyzer()
        report = analyzer.analyze(state_dict)
        print(report.summary())
    """

    def __init__(
        self,
        dead_neuron_threshold: float = 1e-3,
        low_head_threshold: float = 1e-2,
        num_heads_hint: int = 8,
    ) -> None:
        self.dead_neuron_threshold = dead_neuron_threshold
        self.low_head_threshold = low_head_threshold
        self.num_heads_hint = num_heads_hint  # fallback if can't infer from weights

    def analyze(self, model_or_path: Any) -> PruningReport:
        """Analyze model weights for pruning candidates.

        Args:
            model_or_path: A ``dict`` state dict, a ``Path``/``str`` to a
                safetensors file, or any object with a ``state_dict()`` method.

        Returns:
            PruningReport with identified prunable components.
        """
        state_dict = self._load_state_dict(model_or_path)
        report = PruningReport()
        total_params = 0
        prunable_params = 0

        for name, tensor in state_dict.items():
            weights = self._to_flat_list(tensor)
            if not weights:
                continue

            dims = self._infer_dims(tensor)
            param_count = len(weights)
            total_params += param_count
            report.total_layers_analyzed += 1

            # Layer-level magnitude
            l2 = math.sqrt(sum(w * w for w in weights))
            report.layer_magnitudes[name] = l2

            if len(dims) < 2:
                continue

            rows, cols = dims[0], dims[1]

            # Check for dead neurons (rows with near-zero norm)
            row_size = len(weights) // rows if rows > 0 else 0
            if row_size > 0:
                for row_idx in range(rows):
                    start = row_idx * row_size
                    row_weights = weights[start : start + row_size]
                    row_norm = math.sqrt(sum(w * w for w in row_weights))
                    if row_norm < self.dead_neuron_threshold:
                        report.dead_neurons.append(
                            DeadNeuronInfo(
                                layer_name=name,
                                neuron_index=row_idx,
                                l2_norm=row_norm,
                            )
                        )
                        prunable_params += row_size

            # Detect attention head layers by name heuristics
            if any(kw in name for kw in ("q_proj", "k_proj", "v_proj", "attn", "attention")):
                num_heads = self._infer_num_heads(name, rows, cols)
                head_size = rows // num_heads if num_heads > 0 else rows
                if head_size > 0:
                    for head_idx in range(num_heads):
                        start = head_idx * head_size * cols
                        end = start + head_size * cols
                        head_weights = weights[start:end]
                        if head_weights:
                            avg_mag = sum(abs(w) for w in head_weights) / len(head_weights)
                            if avg_mag < self.low_head_threshold:
                                report.low_mag_heads.append(
                                    LowMagHeadInfo(
                                        layer_name=name,
                                        head_index=head_idx,
                                        avg_magnitude=avg_mag,
                                    )
                                )
                                prunable_params += len(head_weights)

        report.total_params = total_params
        if total_params > 0:
            report.prunable_param_fraction = prunable_params / total_params
        report.estimated_speedup = self._estimate_speedup(report)
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_num_heads(self, name: str, rows: int, cols: int) -> int:
        """Try to infer number of heads from common naming conventions.

        Prioritises returning a reasonable head count > 1 when possible.
        Uses the heuristic that head_dim is typically 64 or 128.
        """
        # Prefer head_dim=64: n_heads = rows // 64
        for head_dim in (64, 128, 32):
            n = rows // head_dim
            if n > 1 and rows % head_dim == 0:
                return n
        return self.num_heads_hint

    @staticmethod
    def _estimate_speedup(report: PruningReport) -> float:
        """Rough speedup estimate: 1 + prunable_fraction * 0.5."""
        return 1.0 + report.prunable_param_fraction * 0.5

    @staticmethod
    def _load_state_dict(model_or_path: Any) -> dict[str, Any]:
        """Load a state dict from various sources."""
        if isinstance(model_or_path, dict):
            return model_or_path

        path = Path(str(model_or_path))
        if path.exists() and path.suffix == ".safetensors":
            try:
                from safetensors import safe_open  # type: ignore[import-not-found]

                result: dict[str, Any] = {}
                with safe_open(str(path), framework="numpy") as f:  # type: ignore[attr-defined]
                    for key in f.keys():
                        result[key] = f.get_tensor(key)
                return result
            except ImportError:
                pass  # safetensors not installed — fall through

        # Try torch
        if path.exists():
            try:
                import torch  # type: ignore[import-not-found]

                return torch.load(str(path), map_location="cpu")
            except Exception:
                pass

        # Try state_dict() method (live model)
        try:
            return {k: v for k, v in model_or_path.state_dict().items()}
        except AttributeError:
            pass

        return {}

    @staticmethod
    def _to_flat_list(tensor: Any) -> list[float]:
        """Flatten a tensor/array/list to a Python float list."""
        try:
            return list(tensor.reshape(-1).tolist())  # numpy / torch
        except AttributeError:
            pass
        try:
            return list(tensor.flatten())
        except AttributeError:
            pass

        def _flatten(obj: Any) -> list[float]:
            if isinstance(obj, (int, float)):
                return [float(obj)]
            result: list[float] = []
            try:
                for item in obj:
                    result.extend(_flatten(item))
            except TypeError:
                pass
            return result

        return _flatten(tensor)

    @staticmethod
    def _infer_dims(tensor: Any) -> tuple[int, ...]:
        """Return the shape tuple of a tensor/array/list."""
        try:
            return tuple(int(d) for d in tensor.shape)
        except AttributeError:
            pass
        # Nested list: peek at structure
        if isinstance(tensor, list):
            if tensor and isinstance(tensor[0], list):
                return (len(tensor), len(tensor[0]))
            return (len(tensor),)
        return ()
