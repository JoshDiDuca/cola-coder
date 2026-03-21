"""Training Anomaly Detector: detect anomalies in training metrics.

Monitors a stream of training metrics and flags:
- Sudden loss spikes (z-score above threshold)
- Gradient explosions (gradient norm exceeds threshold)
- Learning rate anomalies (unexpected jumps or zeros)
- NaN / Inf values in any metric
- Stagnation (metric not improving for N steps)

Uses a rolling window of recent values to compute mean and standard
deviation, then z-scores new observations against that baseline.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the training anomaly detector feature is active."""
    return FEATURE_ENABLED


class AnomalyType(str, Enum):
    NAN_INF = "nan_inf"
    LOSS_SPIKE = "loss_spike"
    GRADIENT_EXPLOSION = "gradient_explosion"
    LR_ANOMALY = "lr_anomaly"
    STAGNATION = "stagnation"
    METRIC_REGRESSION = "metric_regression"


@dataclass
class Anomaly:
    """A single detected anomaly event."""

    step: int
    metric: str
    anomaly_type: AnomalyType
    value: float
    z_score: Optional[float]  # None for NaN/Inf events
    message: str
    severity: str  # "warning" or "critical"

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "metric": self.metric,
            "type": self.anomaly_type.value,
            "value": self.value,
            "z_score": self.z_score,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class DetectorConfig:
    """Configuration for anomaly detection thresholds."""

    window_size: int = 20  # Steps in the rolling baseline window
    z_score_threshold: float = 3.0  # Z-score above which a value is anomalous
    grad_norm_threshold: float = 100.0  # Absolute gradient norm explosion threshold
    lr_jump_factor: float = 10.0  # LR change ratio above which it's flagged
    stagnation_steps: int = 50  # Steps without improvement before stagnation alert
    stagnation_min_delta: float = 1e-5  # Minimum improvement to count as progress


@dataclass
class DetectorState:
    """Internal state for one tracked metric."""

    window: deque = field(default_factory=lambda: deque())
    best_value: Optional[float] = None
    steps_since_improvement: int = 0
    last_lr: Optional[float] = None


def _is_finite(v: float) -> bool:
    return not (math.isnan(v) or math.isinf(v))


def _rolling_stats(window: deque) -> tuple[float, float]:
    """Return (mean, std) of values in the window."""
    vals = list(window)
    if not vals:
        return 0.0, 1.0
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, 1.0
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return mean, math.sqrt(max(var, 1e-12))


def _z_score(value: float, mean: float, std: float) -> float:
    return (value - mean) / max(std, 1e-12)


class TrainingAnomalyDetector:
    """Detect anomalies in a stream of training metrics.

    Usage
    -----
    detector = TrainingAnomalyDetector()
    for step, metrics in training_loop:
        anomalies = detector.update(step, metrics)
        if anomalies:
            handle_anomalies(anomalies)
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()
        self._states: dict[str, DetectorState] = {}
        self.all_anomalies: list[Anomaly] = []

    def _get_state(self, metric: str) -> DetectorState:
        if metric not in self._states:
            self._states[metric] = DetectorState(
                window=deque(maxlen=self.config.window_size)
            )
        return self._states[metric]

    def update(self, step: int, metrics: dict[str, float]) -> list[Anomaly]:
        """Process one step's worth of metrics and return any detected anomalies.

        Parameters
        ----------
        step:
            Current training step.
        metrics:
            Dict of metric_name → float value.  Common keys:
            "loss", "grad_norm", "learning_rate", "val_loss", etc.

        Returns
        -------
        List of Anomaly objects detected this step (may be empty).
        """
        anomalies: list[Anomaly] = []

        for metric, value in metrics.items():
            detected = self._check_metric(step, metric, value)
            anomalies.extend(detected)

        self.all_anomalies.extend(anomalies)
        return anomalies

    def _check_metric(self, step: int, metric: str, value: float) -> list[Anomaly]:
        anomalies: list[Anomaly] = []
        state = self._get_state(metric)

        # 1. NaN / Inf check (always critical)
        if not _is_finite(value):
            anomalies.append(
                Anomaly(
                    step=step,
                    metric=metric,
                    anomaly_type=AnomalyType.NAN_INF,
                    value=value,
                    z_score=None,
                    message=f"{metric}={value} (NaN or Inf) at step {step}",
                    severity="critical",
                )
            )
            # Don't update window with bad values
            return anomalies

        # 2. Z-score check against rolling window
        if len(state.window) >= max(5, self.config.window_size // 4):
            mean, std = _rolling_stats(state.window)
            z = _z_score(value, mean, std)
            abs_z = abs(z)

            if abs_z > self.config.z_score_threshold:
                # Determine anomaly type from metric name
                if "grad" in metric.lower() or "norm" in metric.lower():
                    atype = AnomalyType.GRADIENT_EXPLOSION
                    severity = "critical" if abs_z > 2 * self.config.z_score_threshold else "warning"
                elif "loss" in metric.lower():
                    atype = AnomalyType.LOSS_SPIKE
                    severity = "critical" if z > 2 * self.config.z_score_threshold else "warning"
                elif "lr" in metric.lower() or "learning_rate" in metric.lower():
                    atype = AnomalyType.LR_ANOMALY
                    severity = "warning"
                else:
                    atype = AnomalyType.METRIC_REGRESSION
                    severity = "warning"

                anomalies.append(
                    Anomaly(
                        step=step,
                        metric=metric,
                        anomaly_type=atype,
                        value=value,
                        z_score=z,
                        message=(
                            f"{metric}={value:.6g} at step {step} "
                            f"(z={z:.2f}, mean={mean:.6g}, std={std:.6g})"
                        ),
                        severity=severity,
                    )
                )

        # 3. Gradient explosion: absolute threshold
        if ("grad" in metric.lower() or "norm" in metric.lower()) and value > self.config.grad_norm_threshold:
            # Only add if not already flagged by z-score
            if not any(a.anomaly_type == AnomalyType.GRADIENT_EXPLOSION and a.step == step for a in anomalies):
                anomalies.append(
                    Anomaly(
                        step=step,
                        metric=metric,
                        anomaly_type=AnomalyType.GRADIENT_EXPLOSION,
                        value=value,
                        z_score=None,
                        message=f"Gradient explosion: {metric}={value:.4g} > {self.config.grad_norm_threshold} at step {step}",
                        severity="critical",
                    )
                )

        # 4. Learning rate jump
        if ("lr" in metric.lower() or "learning_rate" in metric.lower()) and state.last_lr is not None:
            ratio = value / max(abs(state.last_lr), 1e-12)
            if ratio > self.config.lr_jump_factor or ratio < 1.0 / self.config.lr_jump_factor:
                if not any(a.anomaly_type == AnomalyType.LR_ANOMALY and a.step == step for a in anomalies):
                    anomalies.append(
                        Anomaly(
                            step=step,
                            metric=metric,
                            anomaly_type=AnomalyType.LR_ANOMALY,
                            value=value,
                            z_score=None,
                            message=f"LR jump: {state.last_lr:.6g} → {value:.6g} (ratio={ratio:.2f}) at step {step}",
                            severity="warning",
                        )
                    )
        if "lr" in metric.lower() or "learning_rate" in metric.lower():
            state.last_lr = value

        # 5. Stagnation check for loss metrics
        if "loss" in metric.lower():
            if state.best_value is None or value < state.best_value - self.config.stagnation_min_delta:
                state.best_value = value
                state.steps_since_improvement = 0
            else:
                state.steps_since_improvement += 1

            if state.steps_since_improvement >= self.config.stagnation_steps:
                # Only fire once per stagnation window
                if state.steps_since_improvement == self.config.stagnation_steps:
                    anomalies.append(
                        Anomaly(
                            step=step,
                            metric=metric,
                            anomaly_type=AnomalyType.STAGNATION,
                            value=value,
                            z_score=None,
                            message=(
                                f"Stagnation: {metric} has not improved for "
                                f"{self.config.stagnation_steps} steps (best={state.best_value:.6g})"
                            ),
                            severity="warning",
                        )
                    )

        # Update rolling window
        state.window.append(value)
        return anomalies

    def summary(self) -> dict[str, Any]:
        """Return a summary of all detected anomalies grouped by type."""
        by_type: dict[str, list[dict]] = {}
        for a in self.all_anomalies:
            key = a.anomaly_type.value
            by_type.setdefault(key, []).append(a.as_dict())
        return {
            "total_anomalies": len(self.all_anomalies),
            "by_type": {k: len(v) for k, v in by_type.items()},
            "critical_count": sum(1 for a in self.all_anomalies if a.severity == "critical"),
            "warning_count": sum(1 for a in self.all_anomalies if a.severity == "warning"),
        }

    def reset(self) -> None:
        """Clear all state and history."""
        self._states.clear()
        self.all_anomalies.clear()

    def get_anomalies_by_type(self, anomaly_type: AnomalyType) -> list[Anomaly]:
        """Filter all detected anomalies by type."""
        return [a for a in self.all_anomalies if a.anomaly_type == anomaly_type]

    def get_anomalies_for_metric(self, metric: str) -> list[Anomaly]:
        """Return anomalies for a specific metric."""
        return [a for a in self.all_anomalies if a.metric == metric]
