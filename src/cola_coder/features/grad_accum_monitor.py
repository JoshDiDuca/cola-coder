"""Gradient Accumulation Monitor (improvement #61).

Tracks and validates gradient accumulation behavior across micro-batches.
Detects accumulation drift — when gradients deviate unexpectedly between steps.

In TypeScript terms, think of this like a Redux middleware that logs every
action's state delta and alerts when the accumulated state diverges from what
you'd expect from sequential micro-batch processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math

# ---------------------------------------------------------------------------
# Feature toggle (project convention)
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if gradient accumulation monitoring is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MicroBatchRecord:
    """Record for a single micro-batch gradient snapshot."""

    step: int
    micro_batch_idx: int
    grad_norms: Dict[str, float]
    running_norm: float


@dataclass
class AccumStepReport:
    """Report for one full accumulation step (N micro-batches)."""

    step: int
    num_micro_batches: int
    micro_batch_records: List[MicroBatchRecord] = field(default_factory=list)
    final_grad_norm: float = 0.0
    expected_norm: float = 0.0
    drift_ratio: float = 0.0
    has_drift: bool = False
    has_nan: bool = False
    has_inf: bool = False

    @property
    def drift_detected(self) -> bool:
        return self.has_drift or self.has_nan or self.has_inf


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class GradAccumMonitor:
    """Monitor gradient accumulation across micro-batches.

    Usage::

        monitor = GradAccumMonitor(accumulation_steps=4, drift_threshold=0.1)
        for step in range(num_steps):
            for mb_idx in range(4):
                loss = compute_loss(micro_batch)
                (loss / 4).backward()
                monitor.record_micro_batch(step, mb_idx, model.named_parameters())
            optimizer.step()
            report = monitor.finalize_step(step, model.named_parameters())
            if report.drift_detected:
                print(f"Drift at step {step}: ratio={report.drift_ratio:.3f}")
    """

    def __init__(
        self,
        accumulation_steps: int = 4,
        drift_threshold: float = 0.15,
        history_size: int = 100,
    ) -> None:
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be >= 1")
        if not (0.0 < drift_threshold <= 1.0):
            raise ValueError("drift_threshold must be in (0, 1]")

        self.accumulation_steps = accumulation_steps
        self.drift_threshold = drift_threshold
        self.history_size = history_size

        self._current_records: List[MicroBatchRecord] = []
        self._step_reports: List[AccumStepReport] = []
        self._micro_batch_norms: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_micro_batch(
        self,
        step: int,
        micro_batch_idx: int,
        named_params: list[tuple[str, "object"]],
    ) -> MicroBatchRecord:
        """Record gradient norms for a single micro-batch."""
        grad_norms: Dict[str, float] = {}
        total_sq = 0.0

        for name, param in named_params:
            # Accept either real tensors or plain floats/None (for testing)
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            # Support both tensors (with .item()) and plain numbers
            if hasattr(grad, "norm"):
                norm_val = float(grad.norm().item())
            elif hasattr(grad, "__float__"):
                norm_val = abs(float(grad))
            else:
                norm_val = 0.0
            grad_norms[name] = norm_val
            total_sq += norm_val**2

        running_norm = math.sqrt(total_sq)
        record = MicroBatchRecord(
            step=step,
            micro_batch_idx=micro_batch_idx,
            grad_norms=grad_norms,
            running_norm=running_norm,
        )
        self._current_records.append(record)
        self._micro_batch_norms.append(running_norm)
        return record

    def finalize_step(
        self,
        step: int,
        named_params: list[tuple[str, "object"]],
    ) -> AccumStepReport:
        """Finalize an accumulation step and produce a report."""
        # Compute final gradient norm after accumulation
        total_sq = 0.0
        has_nan = False
        has_inf = False
        for _, param in named_params:
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            if hasattr(grad, "norm"):
                norm_val = float(grad.norm().item())
            elif hasattr(grad, "__float__"):
                norm_val = abs(float(grad))
            else:
                norm_val = 0.0
            if math.isnan(norm_val):
                has_nan = True
            elif math.isinf(norm_val):
                has_inf = True
            else:
                total_sq += norm_val**2

        final_norm = math.sqrt(total_sq)

        # Expected norm: if gradients accumulate without drift, the final norm
        # should scale roughly with sqrt(N) * avg_micro_norm
        avg_micro = (
            sum(self._micro_batch_norms) / len(self._micro_batch_norms)
            if self._micro_batch_norms
            else 0.0
        )
        expected = avg_micro * math.sqrt(self.accumulation_steps)

        drift_ratio = (
            abs(final_norm - expected) / max(expected, 1e-8)
            if expected > 0
            else 0.0
        )
        has_drift = drift_ratio > self.drift_threshold

        report = AccumStepReport(
            step=step,
            num_micro_batches=len(self._current_records),
            micro_batch_records=list(self._current_records),
            final_grad_norm=final_norm,
            expected_norm=expected,
            drift_ratio=drift_ratio,
            has_drift=has_drift,
            has_nan=has_nan,
            has_inf=has_inf,
        )

        # Maintain rolling history
        self._step_reports.append(report)
        if len(self._step_reports) > self.history_size:
            self._step_reports.pop(0)

        # Reset for next accumulation window
        self._current_records.clear()
        self._micro_batch_norms.clear()

        return report

    def record_grad_norms_dict(
        self,
        step: int,
        micro_batch_idx: int,
        grad_norms: Dict[str, float],
    ) -> MicroBatchRecord:
        """Convenience: record from a pre-computed {name: norm} dict."""
        total_sq = sum(v**2 for v in grad_norms.values())
        running_norm = math.sqrt(total_sq)
        record = MicroBatchRecord(
            step=step,
            micro_batch_idx=micro_batch_idx,
            grad_norms=grad_norms,
            running_norm=running_norm,
        )
        self._current_records.append(record)
        self._micro_batch_norms.append(running_norm)
        return record

    def finalize_step_from_norm(self, step: int, final_norm: float) -> AccumStepReport:
        """Finalize a step when final norm is already known (e.g. from grad clipper)."""
        has_nan = math.isnan(final_norm)
        has_inf = math.isinf(final_norm)
        safe_norm = 0.0 if (has_nan or has_inf) else final_norm

        avg_micro = (
            sum(self._micro_batch_norms) / len(self._micro_batch_norms)
            if self._micro_batch_norms
            else 0.0
        )
        expected = avg_micro * math.sqrt(self.accumulation_steps)
        drift_ratio = (
            abs(safe_norm - expected) / max(expected, 1e-8) if expected > 0 else 0.0
        )
        has_drift = drift_ratio > self.drift_threshold

        report = AccumStepReport(
            step=step,
            num_micro_batches=len(self._current_records),
            micro_batch_records=list(self._current_records),
            final_grad_norm=safe_norm,
            expected_norm=expected,
            drift_ratio=drift_ratio,
            has_drift=has_drift,
            has_nan=has_nan,
            has_inf=has_inf,
        )
        self._step_reports.append(report)
        if len(self._step_reports) > self.history_size:
            self._step_reports.pop(0)
        self._current_records.clear()
        self._micro_batch_norms.clear()
        return report

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def get_history(self) -> List[AccumStepReport]:
        """Return all stored step reports."""
        return list(self._step_reports)

    def drift_rate(self) -> float:
        """Fraction of recent steps that had drift."""
        if not self._step_reports:
            return 0.0
        return sum(1 for r in self._step_reports if r.has_drift) / len(
            self._step_reports
        )

    def summary(self) -> dict:
        """Return a summary dict suitable for logging."""
        reports = self._step_reports
        if not reports:
            return {"steps_tracked": 0}
        norms = [r.final_grad_norm for r in reports]
        return {
            "steps_tracked": len(reports),
            "drift_rate": self.drift_rate(),
            "nan_rate": sum(1 for r in reports if r.has_nan) / len(reports),
            "inf_rate": sum(1 for r in reports if r.has_inf) / len(reports),
            "avg_grad_norm": sum(norms) / len(norms),
            "max_grad_norm": max(norms),
            "min_grad_norm": min(norms),
        }

    def reset(self) -> None:
        """Clear all state."""
        self._current_records.clear()
        self._micro_batch_norms.clear()
        self._step_reports.clear()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def check_accumulation(
    micro_norms: List[float],
    final_norm: float,
    accumulation_steps: Optional[int] = None,
    drift_threshold: float = 0.15,
) -> tuple[bool, float]:
    """Quick check: does final_norm match what micro_norms predict?

    Returns (has_drift, drift_ratio).
    """
    n = accumulation_steps or len(micro_norms)
    avg = sum(micro_norms) / max(len(micro_norms), 1)
    expected = avg * math.sqrt(n)
    ratio = abs(final_norm - expected) / max(expected, 1e-8)
    return ratio > drift_threshold, ratio
