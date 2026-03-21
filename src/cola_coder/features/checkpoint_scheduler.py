"""Training Checkpoint Scheduler.

Provides smart checkpoint scheduling:
  - Save more frequently early in training (when loss is volatile)
  - Save less frequently as loss stabilises
  - Exponential decay schedule for checkpoint intervals
  - Override support for manual checkpoints
  - Tracks checkpoint history with loss values

For a TS dev: like debouncing save events — rapid changes get saved often;
once things stabilise, saves become less frequent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class CheckpointRecord(NamedTuple):
    """Record of a checkpoint that was (or should be) saved."""

    step: int
    loss: float
    reason: str  # "scheduled" | "override" | "best" | "final"
    interval_used: int  # steps between this and previous scheduled checkpoint


@dataclass
class SchedulerState:
    """Mutable state tracked by the scheduler."""

    current_step: int = 0
    last_checkpoint_step: int = 0
    best_loss: float = float("inf")
    checkpoints: list[CheckpointRecord] = field(default_factory=list)
    overrides_pending: list[int] = field(default_factory=list)  # steps to force-save

    @property
    def total_checkpoints(self) -> int:
        return len(self.checkpoints)


# ---------------------------------------------------------------------------
# Schedule functions
# ---------------------------------------------------------------------------


def exponential_decay_interval(
    step: int,
    initial_interval: int,
    min_interval: int,
    decay_rate: float,
    warmup_steps: int = 0,
) -> int:
    """Compute checkpoint interval at *step* using exponential decay.

    The interval grows as training progresses (save less often later).

    Parameters
    ----------
    step:
        Current training step.
    initial_interval:
        Interval at step 0 (most frequent).
    min_interval:
        Floor — never save less often than this.
    decay_rate:
        Multiplier applied per ``initial_interval`` steps.
        E.g., 1.5 means interval grows 1.5× every initial_interval steps.
    warmup_steps:
        During warmup, use the initial_interval unchanged.
    """
    if step < warmup_steps:
        return initial_interval
    effective_step = step - warmup_steps
    exponent = effective_step / max(initial_interval, 1)
    # Clamp exponent to avoid overflow
    try:
        growth = decay_rate ** min(exponent, 300.0)
    except OverflowError:
        growth = float("inf")
    raw_interval = initial_interval * growth
    interval = int(min(raw_interval, 2 ** 30))
    return max(min_interval, interval)


def linear_decay_interval(
    step: int,
    initial_interval: int,
    final_interval: int,
    total_steps: int,
) -> int:
    """Linearly interpolate checkpoint interval from initial to final."""
    if total_steps <= 0:
        return initial_interval
    t = min(1.0, step / total_steps)
    return int(initial_interval + t * (final_interval - initial_interval))


def loss_adaptive_interval(
    recent_losses: list[float],
    base_interval: int,
    volatility_scale: float = 2.0,
) -> int:
    """Compute interval inversely proportional to loss volatility.

    High volatility → more frequent checkpoints.
    Low volatility → less frequent checkpoints.
    """
    if len(recent_losses) < 2:
        return base_interval
    diffs = [abs(recent_losses[i] - recent_losses[i - 1]) for i in range(1, len(recent_losses))]
    mean_diff = sum(diffs) / len(diffs)
    if mean_diff < 1e-9:
        return int(base_interval * volatility_scale)
    # Interval shrinks when differences are large
    factor = 1.0 / (1.0 + mean_diff * volatility_scale)
    return max(1, int(base_interval * factor))


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class CheckpointScheduler:
    """Determine when to save checkpoints during training.

    Parameters
    ----------
    initial_interval:
        Steps between checkpoints at the start of training.
    min_interval:
        Minimum steps between any two checkpoints.
    max_interval:
        Maximum steps between checkpoints (caps the growth).
    decay_rate:
        Exponential growth factor for the interval per initial_interval steps.
    save_best:
        If True, always save when a new best loss is achieved.
    total_steps:
        Total expected training steps (used for schedule preview).
    warmup_steps:
        Steps during which the initial_interval is kept constant.
    """

    def __init__(
        self,
        initial_interval: int = 100,
        min_interval: int = 50,
        max_interval: int = 2000,
        decay_rate: float = 1.5,
        save_best: bool = True,
        total_steps: int = 10_000,
        warmup_steps: int = 0,
    ) -> None:
        self.initial_interval = initial_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.decay_rate = decay_rate
        self.save_best = save_best
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.state = SchedulerState()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_checkpoint(self, step: int, loss: float) -> bool:
        """Return True if a checkpoint should be saved at *step*."""
        self.state.current_step = step
        if step in self.state.overrides_pending:
            return True
        if self.save_best and loss < self.state.best_loss:
            return True
        interval = self._current_interval(step)
        steps_since = step - self.state.last_checkpoint_step
        return steps_since >= interval

    def record_checkpoint(self, step: int, loss: float, reason: str = "scheduled") -> CheckpointRecord:
        """Record that a checkpoint was saved at *step* with *loss*."""
        interval_used = step - self.state.last_checkpoint_step

        record = CheckpointRecord(
            step=step,
            loss=loss,
            reason=reason,
            interval_used=interval_used,
        )
        self.state.checkpoints.append(record)
        self.state.last_checkpoint_step = step

        if loss < self.state.best_loss:
            self.state.best_loss = loss

        # Remove the override if it was pending
        if step in self.state.overrides_pending:
            self.state.overrides_pending.remove(step)

        return record

    def step(self, step: int, loss: float) -> CheckpointRecord | None:
        """Convenience: check and record in one call. Returns record if saved."""
        if not self.should_checkpoint(step, loss):
            return None

        if step in self.state.overrides_pending:
            reason = "override"
        elif self.save_best and loss < self.state.best_loss:
            reason = "best"
        else:
            reason = "scheduled"

        return self.record_checkpoint(step, loss, reason=reason)

    def add_override(self, step: int) -> None:
        """Force a checkpoint at a specific step."""
        if step not in self.state.overrides_pending:
            self.state.overrides_pending.append(step)

    def preview_schedule(self, total_steps: int | None = None) -> list[int]:
        """Preview which steps would have checkpoints (ignoring best/override)."""
        total = total_steps or self.total_steps
        schedule: list[int] = []
        last = 0
        step = 0
        while step < total:
            interval = self._current_interval(step)
            step = last + interval
            if step <= total:
                schedule.append(step)
                last = step
            else:
                break
        return schedule

    def reset(self) -> None:
        """Reset scheduler state (e.g., for a new training run)."""
        self.state = SchedulerState()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _current_interval(self, step: int) -> int:
        interval = exponential_decay_interval(
            step=step,
            initial_interval=self.initial_interval,
            min_interval=self.min_interval,
            decay_rate=self.decay_rate,
            warmup_steps=self.warmup_steps,
        )
        return min(interval, self.max_interval)

    def summary(self) -> str:
        s = self.state
        lines = [
            f"CheckpointScheduler: {s.total_checkpoints} checkpoints",
            f"  Best loss: {s.best_loss:.4f}",
            f"  Last checkpoint: step {s.last_checkpoint_step}",
            f"  Current interval at step {s.current_step}: "
            f"{self._current_interval(s.current_step)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def compute_checkpoint_density(
    schedule: list[int],
    total_steps: int,
) -> list[float]:
    """Return the checkpoint density (checkpoints per 1000 steps) in windows.

    Splits training into 10 equal windows and counts checkpoints per window.
    """
    if total_steps <= 0:
        return []
    window_size = total_steps / 10
    densities = []
    for w in range(10):
        start = w * window_size
        end = start + window_size
        count = sum(1 for s in schedule if start <= s < end)
        densities.append(round(count / (window_size / 1000), 4))
    return densities


def estimate_checkpoints_needed(
    total_steps: int,
    initial_interval: int,
    decay_rate: float,
    min_interval: int = 50,
) -> int:
    """Estimate how many checkpoints will be saved over a training run."""
    count = 0
    last = 0
    step = 0
    while step < total_steps:
        interval = exponential_decay_interval(
            step=step,
            initial_interval=initial_interval,
            min_interval=min_interval,
            decay_rate=decay_rate,
        )
        step = last + interval
        if step <= total_steps:
            count += 1
            last = step
        else:
            break
    return count
