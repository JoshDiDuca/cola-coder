"""Training Stability Monitor — Feature 94

Monitor training stability signals and alert on instability indicators.

Tracked signals
---------------
- **Gradient norm variance**: high variance in per-step grad-norms → unstable
- **Loss oscillation**: measures oscillation amplitude in recent loss history
- **Learning rate sensitivity**: change in loss per unit LR change
- **NaN / Inf detection**: flag steps where loss or grad-norm is non-finite

All methods accept plain Python lists of floats — no PyTorch / CUDA required.

Feature toggle: set FEATURE_ENABLED = False to disable.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the training stability monitor is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------


@dataclass
class StabilityAlert:
    """A single stability alert raised by the monitor."""

    severity: AlertSeverity
    signal: str
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None

    def __str__(self) -> str:
        parts = [f"[{self.severity.value.upper()}] {self.signal}: {self.message}"]
        if self.value is not None:
            parts.append(f"(value={self.value:.4g}, threshold={self.threshold:.4g})")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclass
class StabilitySnapshot:
    """Point-in-time stability assessment."""

    step: int
    grad_norm_variance: Optional[float]
    loss_oscillation: Optional[float]
    lr_sensitivity: Optional[float]
    has_nan_inf: bool
    alerts: list[StabilityAlert] = field(default_factory=list)

    @property
    def is_stable(self) -> bool:
        critical = [a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]
        return not critical and not self.has_nan_inf

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "grad_norm_variance": self.grad_norm_variance,
            "loss_oscillation": self.loss_oscillation,
            "lr_sensitivity": self.lr_sensitivity,
            "has_nan_inf": self.has_nan_inf,
            "n_alerts": len(self.alerts),
            "is_stable": self.is_stable,
        }


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


@dataclass
class StabilityThresholds:
    """Configurable alert thresholds."""

    grad_norm_var_warning: float = 5.0
    grad_norm_var_critical: float = 50.0
    loss_oscillation_warning: float = 0.5
    loss_oscillation_critical: float = 2.0
    lr_sensitivity_warning: float = 10.0
    lr_sensitivity_critical: float = 100.0
    min_window: int = 5  # min steps needed before analysis


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class TrainingStabilityMonitor:
    """Track training stability signals and emit alerts.

    Usage
    -----
    Call :meth:`record` after each training step.  Call :meth:`assess` at
    any time to get a :class:`StabilitySnapshot`.
    """

    def __init__(self, thresholds: Optional[StabilityThresholds] = None) -> None:
        self.thresholds = thresholds or StabilityThresholds()
        self._grad_norms: list[float] = []
        self._losses: list[float] = []
        self._lrs: list[float] = []
        self._steps: list[int] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        lr: float = 1e-4,
    ) -> None:
        """Record a single training step's metrics.

        Parameters
        ----------
        step: Global training step number.
        loss: Training loss at this step.
        grad_norm: Overall gradient L2 norm.
        lr: Learning rate used at this step.
        """
        self._steps.append(step)
        self._losses.append(loss)
        self._grad_norms.append(grad_norm)
        self._lrs.append(lr)

    def clear(self) -> None:
        """Reset all recorded history."""
        self._grad_norms.clear()
        self._losses.clear()
        self._lrs.clear()
        self._steps.clear()

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def grad_norm_variance(self, window: Optional[int] = None) -> Optional[float]:
        """Variance of grad norms over the last *window* steps."""
        data = self._recent(self._grad_norms, window)
        if len(data) < 2:
            return None
        return statistics.variance(data)

    def loss_oscillation(self, window: Optional[int] = None) -> Optional[float]:
        """Measure loss oscillation as mean absolute difference between consecutive steps."""
        data = self._recent(self._losses, window)
        if len(data) < 2:
            return None
        diffs = [abs(data[i + 1] - data[i]) for i in range(len(data) - 1)]
        return statistics.mean(diffs)

    def lr_sensitivity(self, window: Optional[int] = None) -> Optional[float]:
        """Approximate LR sensitivity: |Δloss| / |ΔLR| per consecutive step pair."""
        losses = self._recent(self._losses, window)
        lrs = self._recent(self._lrs, window)
        if len(losses) < 2:
            return None
        sensitivities: list[float] = []
        for i in range(len(losses) - 1):
            dlr = abs(lrs[i + 1] - lrs[i])
            if dlr > 1e-15:
                sensitivities.append(abs(losses[i + 1] - losses[i]) / dlr)
        if not sensitivities:
            return None
        return statistics.mean(sensitivities)

    def has_nan_inf(self) -> bool:
        """Return True if any recorded loss or grad-norm is NaN or Inf."""
        for v in self._losses + self._grad_norms:
            if math.isnan(v) or math.isinf(v):
                return True
        return False

    # ------------------------------------------------------------------
    # Assessment
    # ------------------------------------------------------------------

    def assess(self, window: Optional[int] = None) -> StabilitySnapshot:
        """Return a stability snapshot for the current history.

        Parameters
        ----------
        window:
            Number of recent steps to analyse.  ``None`` uses all history.
        """
        step = self._steps[-1] if self._steps else 0
        t = self.thresholds

        gnv = self.grad_norm_variance(window)
        lo = self.loss_oscillation(window)
        lrs = self.lr_sensitivity(window)
        nan_inf = self.has_nan_inf()
        alerts: list[StabilityAlert] = []

        # NaN / Inf
        if nan_inf:
            alerts.append(
                StabilityAlert(
                    severity=AlertSeverity.CRITICAL,
                    signal="nan_inf",
                    message="NaN or Inf detected in loss or gradient norm",
                )
            )

        # Grad norm variance
        if gnv is not None:
            if gnv >= t.grad_norm_var_critical:
                alerts.append(
                    StabilityAlert(
                        severity=AlertSeverity.CRITICAL,
                        signal="grad_norm_variance",
                        message="Gradient norm variance is critically high",
                        value=gnv,
                        threshold=t.grad_norm_var_critical,
                    )
                )
            elif gnv >= t.grad_norm_var_warning:
                alerts.append(
                    StabilityAlert(
                        severity=AlertSeverity.WARNING,
                        signal="grad_norm_variance",
                        message="Gradient norm variance is elevated",
                        value=gnv,
                        threshold=t.grad_norm_var_warning,
                    )
                )

        # Loss oscillation
        if lo is not None:
            if lo >= t.loss_oscillation_critical:
                alerts.append(
                    StabilityAlert(
                        severity=AlertSeverity.CRITICAL,
                        signal="loss_oscillation",
                        message="Loss is oscillating critically",
                        value=lo,
                        threshold=t.loss_oscillation_critical,
                    )
                )
            elif lo >= t.loss_oscillation_warning:
                alerts.append(
                    StabilityAlert(
                        severity=AlertSeverity.WARNING,
                        signal="loss_oscillation",
                        message="Loss oscillation is elevated",
                        value=lo,
                        threshold=t.loss_oscillation_warning,
                    )
                )

        # LR sensitivity
        if lrs is not None:
            if lrs >= t.lr_sensitivity_critical:
                alerts.append(
                    StabilityAlert(
                        severity=AlertSeverity.CRITICAL,
                        signal="lr_sensitivity",
                        message="Learning rate sensitivity is critically high",
                        value=lrs,
                        threshold=t.lr_sensitivity_critical,
                    )
                )
            elif lrs >= t.lr_sensitivity_warning:
                alerts.append(
                    StabilityAlert(
                        severity=AlertSeverity.WARNING,
                        signal="lr_sensitivity",
                        message="Learning rate sensitivity is elevated",
                        value=lrs,
                        threshold=t.lr_sensitivity_warning,
                    )
                )

        return StabilitySnapshot(
            step=step,
            grad_norm_variance=gnv,
            loss_oscillation=lo,
            lr_sensitivity=lrs,
            has_nan_inf=nan_inf,
            alerts=alerts,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_steps(self) -> int:
        return len(self._steps)

    @property
    def latest_loss(self) -> Optional[float]:
        return self._losses[-1] if self._losses else None

    @property
    def latest_grad_norm(self) -> Optional[float]:
        return self._grad_norms[-1] if self._grad_norms else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _recent(data: list[float], window: Optional[int]) -> list[float]:
        if window is None or window >= len(data):
            return list(data)
        return data[-window:]
