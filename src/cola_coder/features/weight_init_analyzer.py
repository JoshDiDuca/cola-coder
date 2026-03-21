"""Weight Initialization Analyzer — feature 43.

Checks whether a model's initial weight tensors conform to best practices
(Xavier/Glorot, He/Kaiming, or plain normal/uniform).  Flags layers with
suspiciously high or low initial magnitudes.

Why this matters:
    Bad initialization → vanishing or exploding gradients at step 0.
    Xavier init targets var = 2 / (fan_in + fan_out) (Glorot 2010).
    He init targets var = 2 / fan_in (He 2015, for ReLU networks).

This module works with plain Python dicts of (name → tensor-like objects)
so it can be tested without a GPU.  When PyTorch is available the helper
``analyze_model`` accepts a real ``nn.Module``.

Feature toggle pattern (project convention):
    FEATURE_ENABLED = False → analyzer returns an empty AnalysisReport.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if weight init analysis is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LayerReport:
    """Analysis result for a single weight tensor."""

    name: str
    shape: Tuple[int, ...]
    mean: float
    std: float
    abs_max: float
    fan_in: int
    fan_out: int

    # Expected std under each init scheme
    xavier_std: float
    he_std: float

    # How many standard deviations the actual std is from each expected
    xavier_deviation: float
    he_deviation: float

    # Flags
    is_bias: bool
    is_suspicious: bool
    issue: str  # human-readable description or ""

    def summary(self) -> str:
        flag = " [SUSPICIOUS]" if self.is_suspicious else ""
        return (
            f"{self.name}{flag}: shape={self.shape} "
            f"std={self.std:.4f} xavier_std={self.xavier_std:.4f} "
            f"he_std={self.he_std:.4f}"
        )


@dataclass
class WeightInitReport:
    """Aggregate analysis of all weight tensors."""

    layers: List[LayerReport] = field(default_factory=list)
    num_suspicious: int = 0
    num_zero_init: int = 0
    overall_ok: bool = True
    summary_lines: List[str] = field(default_factory=list)

    def suspicious_layers(self) -> List[LayerReport]:
        return [lr for lr in self.layers if lr.is_suspicious]

    def summary(self) -> str:
        lines = [
            f"WeightInitReport: {len(self.layers)} layers, "
            f"{self.num_suspicious} suspicious, "
            f"{self.num_zero_init} zero-init, "
            f"overall_ok={self.overall_ok}"
        ]
        for sl in self.suspicious_layers():
            lines.append(f"  [WARN] {sl.summary()} — {sl.issue}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class WeightInitAnalyzer:
    """Analyzes weight tensors for initialization quality.

    Accepts either real torch tensors (if torch is available) or any
    object that exposes ``.shape``, ``.mean()``, ``.std()``, and
    ``.__abs__().max()`` — which covers both torch.Tensor and numpy ndarray.
    """

    def __init__(
        self,
        suspicious_threshold: float = 3.0,
        near_zero_threshold: float = 1e-6,
    ) -> None:
        """
        Args:
            suspicious_threshold: A layer's std is suspicious if
                ``|actual_std - expected_std| / expected_std > threshold``.
            near_zero_threshold: Layers with std < this are considered zero-init.
        """
        self.suspicious_threshold = suspicious_threshold
        self.near_zero_threshold = near_zero_threshold

    def analyze(
        self, state_dict: Dict[str, Any]
    ) -> WeightInitReport:
        """Analyze a state dict mapping name → tensor.

        Args:
            state_dict: Dict of {param_name: tensor} (or any array-like with
                .shape, .mean(), .std(), .abs(), .max()).

        Returns:
            WeightInitReport with per-layer results and summary flags.
        """
        if not FEATURE_ENABLED:
            return WeightInitReport()

        report = WeightInitReport()

        for name, tensor in state_dict.items():
            layer_report = self._analyze_tensor(name, tensor)
            if layer_report is None:
                continue
            report.layers.append(layer_report)
            if layer_report.is_suspicious:
                report.num_suspicious += 1
            if layer_report.std < self.near_zero_threshold and not layer_report.is_bias:
                report.num_zero_init += 1

        report.overall_ok = report.num_suspicious == 0
        return report

    def _analyze_tensor(self, name: str, tensor: Any) -> Optional[LayerReport]:
        """Compute statistics and expected init values for a single tensor."""
        shape = tuple(tensor.shape)
        if len(shape) < 1:
            return None  # scalar — skip

        # Compute stats — handle both torch tensors and numpy arrays
        try:
            mean = float(tensor.mean())
            std = float(tensor.std())
            # Support objects with an .abs() method (our _SimpleArray and numpy)
            # or fall back to the builtin abs() for scalar-like things.
            if hasattr(tensor, "abs"):
                abs_max = float(tensor.abs().max())
            else:
                abs_max = float(abs(tensor).max())
        except Exception:
            return None

        # Identify bias (1-D tensors named *bias*)
        is_bias = len(shape) == 1 and "bias" in name.lower()

        # Compute fan_in / fan_out
        fan_in, fan_out = _compute_fan(shape)

        # Expected stds
        xavier_std = _xavier_std(fan_in, fan_out)
        he_std = _he_std(fan_in)

        # Deviation from each scheme (relative)
        xavier_deviation = abs(std - xavier_std) / max(xavier_std, 1e-12)
        he_deviation = abs(std - he_std) / max(he_std, 1e-12)

        # A weight is suspicious if it deviates greatly from BOTH schemes
        # and is not a bias (biases are usually zero-initialized)
        issues: List[str] = []
        is_suspicious = False

        if not is_suspicious and not is_bias:
            if std < self.near_zero_threshold:
                issues.append("near-zero std (possible zero-init or vanishing)")
                is_suspicious = True
            elif std > 10.0:
                issues.append(f"very large std={std:.3f} (possible exploding init)")
                is_suspicious = True
            elif (
                xavier_deviation > self.suspicious_threshold
                and he_deviation > self.suspicious_threshold
            ):
                issues.append(
                    f"std={std:.4f} deviates from Xavier ({xavier_std:.4f}) "
                    f"and He ({he_std:.4f}) by >{self.suspicious_threshold}x"
                )
                is_suspicious = True

        # Check for non-zero mean in weight matrices (usually a sign of trouble)
        if not is_bias and abs(mean) > 0.1 * max(abs_max, 1e-12):
            issues.append(f"non-zero mean={mean:.4f} (weight shift may cause saturation)")
            is_suspicious = True

        return LayerReport(
            name=name,
            shape=shape,
            mean=mean,
            std=std,
            abs_max=abs_max,
            fan_in=fan_in,
            fan_out=fan_out,
            xavier_std=xavier_std,
            he_std=he_std,
            xavier_deviation=xavier_deviation,
            he_deviation=he_deviation,
            is_bias=is_bias,
            is_suspicious=is_suspicious,
            issue="; ".join(issues),
        )


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _compute_fan(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """Compute fan_in and fan_out for a weight tensor shape."""
    if len(shape) == 1:
        return shape[0], shape[0]
    if len(shape) == 2:
        return shape[1], shape[0]
    # Conv-like: shape = (out_channels, in_channels, *kernel)
    receptive_field = 1
    for s in shape[2:]:
        receptive_field *= s
    fan_in = shape[1] * receptive_field
    fan_out = shape[0] * receptive_field
    return fan_in, fan_out


def _xavier_std(fan_in: int, fan_out: int) -> float:
    """Xavier / Glorot uniform std (Glorot 2010)."""
    return math.sqrt(2.0 / max(fan_in + fan_out, 1))


def _he_std(fan_in: int) -> float:
    """He / Kaiming normal std (He 2015, mode='fan_in')."""
    return math.sqrt(2.0 / max(fan_in, 1))


# ---------------------------------------------------------------------------
# Convenience helper for NumPy arrays (no torch dependency in tests)
# ---------------------------------------------------------------------------


class _SimpleArray:
    """Minimal array-like for testing without NumPy or PyTorch."""

    def __init__(self, data: Sequence[float], shape: Tuple[int, ...]) -> None:
        self._data = list(data)
        self.shape = shape

    def mean(self) -> float:
        return sum(self._data) / max(len(self._data), 1)

    def std(self) -> float:
        m = self.mean()
        var = sum((x - m) ** 2 for x in self._data) / max(len(self._data) - 1, 1)
        return math.sqrt(var)

    def abs(self) -> "_SimpleArray":
        return _SimpleArray([abs(x) for x in self._data], self.shape)

    def max(self) -> float:
        return max(self._data) if self._data else 0.0


def make_test_tensor(
    values: Sequence[float],
    shape: Tuple[int, ...],
) -> _SimpleArray:
    """Create a minimal array-like for testing purposes."""
    return _SimpleArray(values, shape)
