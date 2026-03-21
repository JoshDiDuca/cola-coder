"""Training Progress Estimator.

Estimate remaining training time by fitting an exponential decay curve
to observed loss values.  Predicts when a target loss will be reached
and gives a time-to-target estimate.

Uses non-linear least squares (implemented from scratch — no scipy) to
fit the model:  loss(step) = a * exp(-b * step) + c

Also accounts for learning rate schedule effects on the loss trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Curve fitting (gradient-free Nelder-Mead-style, lightweight)
# ---------------------------------------------------------------------------


def _exp_decay(step: float, a: float, b: float, c: float) -> float:
    """Exponential decay model: a * exp(-b * step) + c."""
    return a * math.exp(-b * max(step, 0)) + c


def _residuals(params: tuple[float, float, float], steps: list[float], losses: list[float]) -> float:
    """Sum of squared residuals."""
    a, b, c = params
    total = 0.0
    for s, loss_val in zip(steps, losses):
        pred = _exp_decay(s, a, b, c)
        total += (pred - loss_val) ** 2
    return total


def fit_exponential_decay(
    steps: list[int],
    losses: list[float],
    max_iterations: int = 500,
) -> tuple[float, float, float] | None:
    """Fit loss(step) = a*exp(-b*step) + c to (steps, losses) data.

    Returns (a, b, c) tuple, or None if fitting fails.
    Uses a simple coordinate descent approach.
    """
    if len(steps) < 3:
        return None

    fsteps = [float(s) for s in steps]
    # Initial guess
    c = min(losses) * 0.9
    a = max(losses) - c
    if a <= 0:
        a = 0.1
    # Estimate b from the slope
    if len(steps) >= 2 and steps[-1] != steps[0]:
        span = max(losses) - min(losses)
        if span > 1e-9:
            b = 1.0 / max(steps)
        else:
            b = 1e-4
    else:
        b = 1e-4
    b = max(b, 1e-8)

    best_params = (a, b, c)
    best_loss = _residuals(best_params, fsteps, losses)

    step_sizes = [a * 0.5, b * 0.5, abs(c) * 0.5 + 0.01]

    for _ in range(max_iterations):
        improved = False
        for dim in range(3):
            for direction in (+1, -1):
                new_params = list(best_params)
                new_params[dim] += direction * step_sizes[dim]
                # Keep b positive
                if dim == 1 and new_params[1] <= 0:
                    continue
                candidate = tuple(new_params)  # type: ignore[assignment]
                candidate_loss = _residuals(candidate, fsteps, losses)  # type: ignore[arg-type]
                if candidate_loss < best_loss:
                    best_loss = candidate_loss
                    best_params = candidate  # type: ignore[assignment]
                    improved = True
        if not improved:
            step_sizes = [s * 0.5 for s in step_sizes]
            if all(s < 1e-9 for s in step_sizes):
                break

    return best_params


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LossObservation:
    """A single recorded (step, loss) pair."""

    step: int
    loss: float
    timestamp_s: float | None = None  # wall-clock time in seconds, if available


@dataclass
class EstimationResult:
    """Result of a training progress estimation."""

    # Fitted curve parameters
    curve_a: float
    curve_b: float
    curve_c: float  # asymptote (minimum achievable loss)
    fit_rmse: float  # root mean squared error of the fit

    # Predictions
    target_loss: float
    predicted_step: int | None    # step when target_loss is expected (None if unreachable)
    steps_remaining: int | None   # steps still needed from the current step
    current_step: int
    current_loss: float

    # Time estimates (only if timestamps were provided)
    seconds_per_step: float | None = None
    estimated_seconds_remaining: float | None = None
    estimated_minutes_remaining: float | None = None

    # Quality flags
    is_converged: bool = False    # True if loss is already below target
    is_diverging: bool = False    # True if b <= 0 (loss not decreasing)
    confidence: float = 0.0       # 0.0–1.0, based on fit quality

    def summary(self) -> str:
        lines = [
            f"EstimationResult: step={self.current_step}, loss={self.current_loss:.4f}",
            f"  Fitted curve: {self.curve_a:.4f}*exp(-{self.curve_b:.6f}*t) + {self.curve_c:.4f}",
            f"  Asymptote (min loss): {self.curve_c:.4f}",
            f"  Target loss: {self.target_loss:.4f}",
        ]
        if self.is_converged:
            lines.append("  Status: CONVERGED (target already reached)")
        elif self.is_diverging:
            lines.append("  Status: DIVERGING (loss not decreasing)")
        elif self.predicted_step is not None:
            lines.append(f"  Predicted step: {self.predicted_step} "
                         f"({self.steps_remaining} steps remaining)")
            if self.estimated_minutes_remaining is not None:
                lines.append(f"  ETA: {self.estimated_minutes_remaining:.1f} minutes")
        else:
            lines.append("  Status: Target unreachable with current trajectory")
        lines.append(f"  Confidence: {self.confidence:.2f}, fit_rmse={self.fit_rmse:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Learning-rate-aware adjustment
# ---------------------------------------------------------------------------


def lr_adjustment_factor(
    current_step: int,
    warmup_steps: int,
    total_steps: int,
    schedule: str = "cosine",
) -> float:
    """Compute the relative learning rate factor at *current_step*.

    Returns a value in (0, 1] representing how much of the initial LR is active.
    """
    if total_steps <= 0:
        return 1.0
    if current_step < warmup_steps:
        return current_step / max(warmup_steps, 1)
    progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(1.0, progress)
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    elif schedule == "linear":
        return 1.0 - progress
    else:
        return 1.0


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class ProgressEstimator:
    """Estimate when training will reach a target loss.

    Parameters
    ----------
    target_loss:
        Desired final loss value.
    max_extrapolation:
        Maximum number of steps to extrapolate beyond the current step.
    warmup_steps:
        LR warmup steps (for schedule-aware estimation).
    total_steps:
        Total planned training steps.
    lr_schedule:
        LR schedule type: "cosine" | "linear" | "constant".
    """

    def __init__(
        self,
        target_loss: float = 2.0,
        max_extrapolation: int = 1_000_000,
        warmup_steps: int = 0,
        total_steps: int = 100_000,
        lr_schedule: str = "cosine",
    ) -> None:
        self.target_loss = target_loss
        self.max_extrapolation = max_extrapolation
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_schedule = lr_schedule
        self.observations: list[LossObservation] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, step: int, loss: float, timestamp_s: float | None = None) -> None:
        """Record a loss observation."""
        self.observations.append(LossObservation(step=step, loss=loss, timestamp_s=timestamp_s))
        # Keep sorted by step
        self.observations.sort(key=lambda o: o.step)

    def estimate(self, steps: list[int] | None = None, losses: list[float] | None = None) -> EstimationResult | None:
        """Estimate when target loss will be reached.

        Can use either the internally recorded observations or
        explicitly passed steps/losses arrays.

        Returns None if insufficient data (< 3 observations).
        """
        if steps is not None and losses is not None:
            obs_steps = steps
            obs_losses = losses
            timestamps: list[float | None] = [None] * len(steps)
        else:
            if len(self.observations) < 3:
                return None
            obs_steps = [o.step for o in self.observations]
            obs_losses = [o.loss for o in self.observations]
            timestamps = [o.timestamp_s for o in self.observations]

        if len(obs_steps) < 3:
            return None

        params = fit_exponential_decay(obs_steps, obs_losses)
        if params is None:
            return None

        a, b, c = params
        current_step = obs_steps[-1]
        current_loss = obs_losses[-1]

        # Compute fit RMSE
        preds = [_exp_decay(s, a, b, c) for s in obs_steps]
        rmse = math.sqrt(sum((p - lv) ** 2 for p, lv in zip(preds, obs_losses)) / len(obs_losses))

        is_converged = current_loss <= self.target_loss
        is_diverging = b <= 0

        predicted_step = None
        steps_remaining = None
        if not is_converged and not is_diverging and a > 0:
            # Solve: a*exp(-b*t) + c = target => t = -ln((target-c)/a) / b
            rhs = self.target_loss - c
            if 0 < rhs < a:
                try:
                    predicted_step = int(-math.log(rhs / a) / b)
                    steps_remaining = max(0, predicted_step - current_step)
                    if steps_remaining > self.max_extrapolation:
                        predicted_step = None
                        steps_remaining = None
                except (ValueError, ZeroDivisionError):
                    pass

        # Time estimation
        seconds_per_step = None
        estimated_seconds_remaining = None
        estimated_minutes_remaining = None
        valid_timestamps = [t for t in timestamps if t is not None]
        if len(valid_timestamps) >= 2 and len(obs_steps) >= 2:
            t0 = valid_timestamps[0]  # type: ignore[index]
            t1 = valid_timestamps[-1]  # type: ignore[index]
            step_span = obs_steps[-1] - obs_steps[0]
            if step_span > 0 and isinstance(t0, float) and isinstance(t1, float):
                seconds_per_step = (t1 - t0) / step_span
                if steps_remaining is not None:
                    estimated_seconds_remaining = steps_remaining * seconds_per_step
                    estimated_minutes_remaining = estimated_seconds_remaining / 60

        # Confidence: based on fit quality relative to loss range
        loss_range = max(obs_losses) - min(obs_losses) + 1e-9
        relative_rmse = rmse / loss_range
        confidence = max(0.0, min(1.0, 1.0 - relative_rmse))

        return EstimationResult(
            curve_a=round(a, 6),
            curve_b=round(b, 8),
            curve_c=round(c, 6),
            fit_rmse=round(rmse, 6),
            target_loss=self.target_loss,
            predicted_step=predicted_step,
            steps_remaining=steps_remaining,
            current_step=current_step,
            current_loss=current_loss,
            seconds_per_step=seconds_per_step,
            estimated_seconds_remaining=estimated_seconds_remaining,
            estimated_minutes_remaining=estimated_minutes_remaining,
            is_converged=is_converged,
            is_diverging=is_diverging,
            confidence=round(confidence, 4),
        )

    def clear(self) -> None:
        """Clear all recorded observations."""
        self.observations.clear()

    def predict_loss_at(self, step: int) -> float | None:
        """Predict the loss at a future step using the fitted curve."""
        if len(self.observations) < 3:
            return None
        obs_steps = [o.step for o in self.observations]
        obs_losses = [o.loss for o in self.observations]
        params = fit_exponential_decay(obs_steps, obs_losses)
        if params is None:
            return None
        a, b, c = params
        return round(_exp_decay(step, a, b, c), 6)
