"""Training Summary Generator — Feature 100

Generate a comprehensive summary of a training run from logged metrics.

Synthesises:
- Key metrics (best loss, best perplexity, final loss)
- Best checkpoint identification
- Training curve statistics (min, max, mean, trend)
- Hardware utilisation estimates
- Total compute (GPU-hours, tokens processed)
- Duration and speed statistics

All inputs are plain Python dicts/lists — no torch/GPU required.

Feature toggle: set FEATURE_ENABLED = False to disable.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the training summary generator is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CurveStats:
    """Statistics for a single metric series."""

    name: str
    values: list[float]

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def min_value(self) -> Optional[float]:
        return min(self.values) if self.values else None

    @property
    def max_value(self) -> Optional[float]:
        return max(self.values) if self.values else None

    @property
    def mean(self) -> Optional[float]:
        return statistics.mean(self.values) if self.values else None

    @property
    def final(self) -> Optional[float]:
        return self.values[-1] if self.values else None

    @property
    def trend(self) -> Optional[str]:
        """Simple linear trend: 'improving', 'degrading', or 'stable'."""
        if len(self.values) < 2:
            return None
        first_half = statistics.mean(self.values[: len(self.values) // 2])
        second_half = statistics.mean(self.values[len(self.values) // 2 :])
        delta = second_half - first_half
        if abs(delta) < 0.01 * max(abs(first_half), 1e-9):
            return "stable"
        # For loss, lower is better → negative delta = improving
        if self.name in ("loss", "val_loss", "train_loss"):
            return "improving" if delta < 0 else "degrading"
        return "improving" if delta > 0 else "degrading"

    @property
    def best_step(self) -> Optional[int]:
        """0-based index of the best value (min for loss, max for accuracy)."""
        if not self.values:
            return None
        if self.name in ("loss", "val_loss", "train_loss"):
            return self.values.index(min(self.values))
        return self.values.index(max(self.values))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            "min": self.min_value,
            "max": self.max_value,
            "mean": self.mean,
            "final": self.final,
            "trend": self.trend,
            "best_step": self.best_step,
        }


@dataclass
class CheckpointInfo:
    """Metadata for a training checkpoint."""

    path: str
    step: int
    loss: Optional[float] = None
    val_loss: Optional[float] = None
    epoch: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Lower = better.  Uses val_loss if available, else train loss."""
        v = self.val_loss if self.val_loss is not None else self.loss
        return v if v is not None else float("inf")


@dataclass
class HardwareStats:
    """Hardware utilisation estimates for the run."""

    gpu_name: str = "unknown"
    gpu_count: int = 1
    avg_gpu_util_pct: Optional[float] = None  # 0-100
    peak_vram_gb: Optional[float] = None
    avg_tokens_per_sec: Optional[float] = None
    total_gpu_hours: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "gpu_name": self.gpu_name,
            "gpu_count": self.gpu_count,
            "avg_gpu_util_pct": self.avg_gpu_util_pct,
            "peak_vram_gb": self.peak_vram_gb,
            "avg_tokens_per_sec": self.avg_tokens_per_sec,
            "total_gpu_hours": self.total_gpu_hours,
        }


@dataclass
class TrainingSummary:
    """Comprehensive training run summary."""

    run_name: str
    total_steps: int
    total_tokens: int
    duration_seconds: float
    best_checkpoint: Optional[CheckpointInfo]
    curves: dict[str, CurveStats]
    hardware: HardwareStats
    config: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    @property
    def best_loss(self) -> Optional[float]:
        curve = self.curves.get("val_loss") or self.curves.get("train_loss")
        return curve.min_value if curve else None

    @property
    def best_perplexity(self) -> Optional[float]:
        loss = self.best_loss
        if loss is None:
            return None
        try:
            return math.exp(loss)
        except OverflowError:
            return float("inf")

    @property
    def tokens_per_second(self) -> Optional[float]:
        if self.duration_seconds > 0:
            return self.total_tokens / self.duration_seconds
        return None

    @property
    def duration_hours(self) -> float:
        return self.duration_seconds / 3600.0

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens,
            "duration_hours": self.duration_hours,
            "best_loss": self.best_loss,
            "best_perplexity": self.best_perplexity,
            "tokens_per_second": self.tokens_per_second,
            "best_checkpoint": self.best_checkpoint.path if self.best_checkpoint else None,
            "hardware": self.hardware.as_dict(),
            "curves": {k: v.as_dict() for k, v in self.curves.items()},
            "config": self.config,
            "notes": self.notes,
        }

    def text_report(self) -> str:
        """Return a human-readable summary string."""
        lines: list[str] = [
            f"=== Training Summary: {self.run_name} ===",
            f"Steps:            {self.total_steps:,}",
            f"Total tokens:     {self.total_tokens / 1e9:.2f}B",
            f"Duration:         {self.duration_hours:.2f} h",
        ]
        if self.best_loss is not None:
            lines.append(f"Best loss:        {self.best_loss:.4f}")
        if self.best_perplexity is not None:
            lines.append(f"Best perplexity:  {self.best_perplexity:.2f}")
        if self.tokens_per_second is not None:
            lines.append(f"Speed:            {self.tokens_per_second:,.0f} tok/s")
        if self.best_checkpoint:
            lines.append(f"Best checkpoint:  {self.best_checkpoint.path}")
        hw = self.hardware
        lines.append(f"GPU:              {hw.gpu_name} x{hw.gpu_count}")
        if hw.total_gpu_hours is not None:
            lines.append(f"GPU-hours:        {hw.total_gpu_hours:.2f}")
        for note in self.notes:
            lines.append(f"Note: {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class TrainingSummaryGenerator:
    """Build a :class:`TrainingSummary` from raw training logs."""

    def __init__(self, run_name: str = "unnamed") -> None:
        self.run_name = run_name
        self._metric_series: dict[str, list[float]] = {}
        self._checkpoints: list[CheckpointInfo] = []
        self._total_tokens: int = 0
        self._duration_seconds: float = 0.0
        self._hardware: HardwareStats = HardwareStats()
        self._config: dict[str, Any] = {}
        self._notes: list[str] = []

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_hardware(self, hardware: HardwareStats) -> "TrainingSummaryGenerator":
        self._hardware = hardware
        return self

    def set_config(self, config: dict[str, Any]) -> "TrainingSummaryGenerator":
        self._config = dict(config)
        return self

    def set_duration(self, seconds: float) -> "TrainingSummaryGenerator":
        self._duration_seconds = seconds
        return self

    def set_total_tokens(self, n: int) -> "TrainingSummaryGenerator":
        self._total_tokens = n
        return self

    def add_note(self, note: str) -> "TrainingSummaryGenerator":
        self._notes.append(note)
        return self

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def record_metric(self, name: str, value: float) -> None:
        """Append a single value to a named metric series."""
        self._metric_series.setdefault(name, []).append(value)

    def record_metrics(self, metrics: dict[str, float]) -> None:
        """Append multiple metrics at once."""
        for name, value in metrics.items():
            self.record_metric(name, value)

    def add_checkpoint(self, checkpoint: CheckpointInfo) -> None:
        self._checkpoints.append(checkpoint)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> TrainingSummary:
        """Assemble and return the :class:`TrainingSummary`."""
        curves = {
            name: CurveStats(name=name, values=list(vals))
            for name, vals in self._metric_series.items()
        }

        # Determine total steps from longest curve
        total_steps = max(
            (len(v) for v in self._metric_series.values()), default=0
        )

        # Best checkpoint
        best_ckpt: Optional[CheckpointInfo] = None
        if self._checkpoints:
            best_ckpt = min(self._checkpoints, key=lambda c: c.score)

        # Hardware: compute total GPU-hours if not set
        hw = self._hardware
        if hw.total_gpu_hours is None and self._duration_seconds > 0:
            hw = HardwareStats(
                gpu_name=self._hardware.gpu_name,
                gpu_count=self._hardware.gpu_count,
                avg_gpu_util_pct=self._hardware.avg_gpu_util_pct,
                peak_vram_gb=self._hardware.peak_vram_gb,
                avg_tokens_per_sec=self._hardware.avg_tokens_per_sec,
                total_gpu_hours=(self._duration_seconds / 3600.0) * self._hardware.gpu_count,
            )

        return TrainingSummary(
            run_name=self.run_name,
            total_steps=total_steps,
            total_tokens=self._total_tokens,
            duration_seconds=self._duration_seconds,
            best_checkpoint=best_ckpt,
            curves=curves,
            hardware=hw,
            config=dict(self._config),
            notes=list(self._notes),
        )
