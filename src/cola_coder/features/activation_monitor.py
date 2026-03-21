"""Activation Monitor — feature 44.

Tracks activation statistics (mean, std, percentiles) per layer during
training.  Detects:

- Dead ReLUs: many near-zero activations (> dead_relu_threshold fraction).
- Exploding activations: mean absolute value > explode_threshold.
- Vanishing signals: mean absolute value < vanish_threshold.

The monitor is intentionally decoupled from PyTorch hook mechanics so it can
be tested with plain Python lists.  The caller is responsible for extracting
activation values and calling ``record()``.

Feature toggle pattern (project convention):
    FEATURE_ENABLED = False → monitor silently discards all records.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if activation monitoring is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ActivationStats:
    """Statistics for one layer over a collection of activation values."""

    layer_name: str
    count: int
    mean: float
    std: float
    abs_mean: float
    min_val: float
    max_val: float
    p5: float   # 5th percentile
    p25: float  # 25th percentile
    p50: float  # median
    p75: float  # 75th percentile
    p95: float  # 95th percentile
    dead_fraction: float
    """Fraction of values that are effectively zero (< dead_zero_eps)."""

    is_dead: bool
    is_exploding: bool
    is_vanishing: bool

    def summary(self) -> str:
        flags = []
        if self.is_dead:
            flags.append("DEAD")
        if self.is_exploding:
            flags.append("EXPLODING")
        if self.is_vanishing:
            flags.append("VANISHING")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        return (
            f"{self.layer_name}{flag_str}: "
            f"mean={self.mean:.4f} std={self.std:.4f} "
            f"abs_mean={self.abs_mean:.4f} dead={self.dead_fraction:.2%}"
        )


@dataclass
class MonitorReport:
    """Aggregate monitor report for all tracked layers."""

    step: int
    layer_stats: Dict[str, ActivationStats] = field(default_factory=dict)

    def dead_layers(self) -> List[str]:
        return [n for n, s in self.layer_stats.items() if s.is_dead]

    def exploding_layers(self) -> List[str]:
        return [n for n, s in self.layer_stats.items() if s.is_exploding]

    def vanishing_layers(self) -> List[str]:
        return [n for n, s in self.layer_stats.items() if s.is_vanishing]

    def has_issues(self) -> bool:
        return bool(self.dead_layers() or self.exploding_layers() or self.vanishing_layers())

    def summary(self) -> str:
        lines = [f"MonitorReport step={self.step} layers={len(self.layer_stats)}"]
        for name, stats in self.layer_stats.items():
            if stats.is_dead or stats.is_exploding or stats.is_vanishing:
                lines.append(f"  [WARN] {stats.summary()}")
        if not any(
            s.is_dead or s.is_exploding or s.is_vanishing
            for s in self.layer_stats.values()
        ):
            lines.append("  All layers OK")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core monitor
# ---------------------------------------------------------------------------


class ActivationMonitor:
    """Collects and analyses layer activation statistics.

    Usage::

        monitor = ActivationMonitor()
        # In your forward pass / hook:
        monitor.record("relu_1", activations.flatten().tolist())
        # After N steps:
        report = monitor.report(step=100)
        if report.has_issues():
            print(report.summary())
    """

    def __init__(
        self,
        window: int = 50,
        dead_relu_threshold: float = 0.5,
        explode_threshold: float = 100.0,
        vanish_threshold: float = 1e-4,
        dead_zero_eps: float = 1e-6,
    ) -> None:
        """
        Args:
            window: Number of recent activation snapshots kept per layer.
            dead_relu_threshold: Fraction of near-zero values above which a
                layer is classified as "dead".
            explode_threshold: abs_mean above this → "exploding".
            vanish_threshold: abs_mean below this → "vanishing".
            dead_zero_eps: Values with |x| < eps are considered "dead zeros".
        """
        self.window = window
        self.dead_relu_threshold = dead_relu_threshold
        self.explode_threshold = explode_threshold
        self.vanish_threshold = vanish_threshold
        self.dead_zero_eps = dead_zero_eps

        # {layer_name: deque of flat value lists}
        self._buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._step: int = 0

    def record(
        self,
        layer_name: str,
        values: Sequence[float],
        step: Optional[int] = None,
    ) -> None:
        """Record activation values for a named layer.

        Args:
            layer_name: Identifier for the layer (e.g. "transformer.layer3.relu").
            values: Flat sequence of activation scalar values.
            step: Optional training step (auto-increments if not provided).
        """
        if not FEATURE_ENABLED:
            return
        self._buffers[layer_name].append(list(values))
        self._step = step if step is not None else self._step + 1

    def report(self, step: Optional[int] = None) -> MonitorReport:
        """Compute a MonitorReport from current recorded data.

        Args:
            step: Training step label for the report.

        Returns:
            MonitorReport with per-layer statistics.
        """
        current_step = step if step is not None else self._step
        monitor_report = MonitorReport(step=current_step)

        for layer_name, snapshots in self._buffers.items():
            if not snapshots:
                continue
            # Flatten all snapshots in the window
            all_vals = [v for snap in snapshots for v in snap]
            stats = _compute_stats(
                layer_name,
                all_vals,
                dead_relu_threshold=self.dead_relu_threshold,
                explode_threshold=self.explode_threshold,
                vanish_threshold=self.vanish_threshold,
                dead_zero_eps=self.dead_zero_eps,
            )
            monitor_report.layer_stats[layer_name] = stats

        return monitor_report

    def reset(self, layer_name: Optional[str] = None) -> None:
        """Clear buffers.  Pass layer_name to reset a single layer."""
        if layer_name is not None:
            self._buffers.pop(layer_name, None)
        else:
            self._buffers.clear()
            self._step = 0

    @property
    def tracked_layers(self) -> List[str]:
        """Names of all layers with buffered data."""
        return list(self._buffers.keys())


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _percentile(sorted_vals: List[float], p: float) -> float:
    """Compute percentile p (0–100) from a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    idx = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _compute_stats(
    layer_name: str,
    values: List[float],
    dead_relu_threshold: float,
    explode_threshold: float,
    vanish_threshold: float,
    dead_zero_eps: float,
) -> ActivationStats:
    n = len(values)
    if n == 0:
        return ActivationStats(
            layer_name=layer_name,
            count=0,
            mean=0.0,
            std=0.0,
            abs_mean=0.0,
            min_val=0.0,
            max_val=0.0,
            p5=0.0,
            p25=0.0,
            p50=0.0,
            p75=0.0,
            p95=0.0,
            dead_fraction=0.0,
            is_dead=False,
            is_exploding=False,
            is_vanishing=False,
        )

    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / max(n - 1, 1)
    std = math.sqrt(variance)
    abs_vals = [abs(x) for x in values]
    abs_mean = sum(abs_vals) / n
    min_val = min(values)
    max_val = max(values)

    sorted_vals = sorted(values)
    p5 = _percentile(sorted_vals, 5)
    p25 = _percentile(sorted_vals, 25)
    p50 = _percentile(sorted_vals, 50)
    p75 = _percentile(sorted_vals, 75)
    p95 = _percentile(sorted_vals, 95)

    dead_count = sum(1 for x in abs_vals if x < dead_zero_eps)
    dead_fraction = dead_count / n

    is_dead = dead_fraction >= dead_relu_threshold
    is_exploding = abs_mean > explode_threshold
    is_vanishing = abs_mean < vanish_threshold and not is_dead

    return ActivationStats(
        layer_name=layer_name,
        count=n,
        mean=mean,
        std=std,
        abs_mean=abs_mean,
        min_val=min_val,
        max_val=max_val,
        p5=p5,
        p25=p25,
        p50=p50,
        p75=p75,
        p95=p95,
        dead_fraction=dead_fraction,
        is_dead=is_dead,
        is_exploding=is_exploding,
        is_vanishing=is_vanishing,
    )
