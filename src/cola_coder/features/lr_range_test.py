"""Learning Rate Range Test — feature 42.

Implements Leslie Smith's LR range test (Smith 2017, "Cyclical Learning Rates
for Training Neural Networks").  The test sweeps learning rates exponentially
over a fixed number of steps and records the loss at each step.  The optimal
LR range is:

    min_lr: where the loss first starts decreasing noticeably
    max_lr: where the loss diverges / starts increasing

This module provides a pure-Python simulation layer (no PyTorch required at
import time) so it can be tested in isolation, plus a torch-based runner when
PyTorch is available.

Feature toggle pattern (project convention):
    FEATURE_ENABLED = False → runner returns an empty LRRangeResult.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if LR range testing is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LRRangeResult:
    """Results of an LR range test sweep."""

    lr_values: List[float] = field(default_factory=list)
    """Learning rates tested, in order."""

    losses: List[float] = field(default_factory=list)
    """Smoothed loss at each LR step."""

    suggested_min_lr: Optional[float] = None
    """Suggested lower bound of the LR range (steepest descent onset)."""

    suggested_max_lr: Optional[float] = None
    """Suggested upper bound of the LR range (just before divergence)."""

    divergence_lr: Optional[float] = None
    """LR at which the loss exceeded the divergence threshold."""

    num_steps: int = 0
    """Total steps in the sweep."""

    def is_valid(self) -> bool:
        """True if the sweep produced usable results."""
        return bool(self.lr_values) and self.suggested_max_lr is not None

    def summary(self) -> str:
        if not self.is_valid():
            return "LRRangeResult: insufficient data"
        return (
            f"LRRangeResult: min_lr={self.suggested_min_lr:.2e} "
            f"max_lr={self.suggested_max_lr:.2e} "
            f"steps={self.num_steps}"
        )


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------


class LRRangeTest:
    """Runs (or simulates) an LR range test.

    Can operate in two modes:

    1. **Simulation mode** — pass a ``loss_fn`` that takes an LR value and
       returns a synthetic loss.  Useful for testing and offline analysis.

    2. **Iterator mode** — call ``next_lr()`` in your training loop, set the
       optimizer LR, run a forward/backward pass, then call ``record_loss()``.

    Example (simulation)::

        def fake_loss(lr):
            return 5.0 * math.exp(lr / 0.01) - 4.9

        test = LRRangeTest(min_lr=1e-7, max_lr=1.0, num_steps=200)
        result = test.run(loss_fn=fake_loss)
        print(result.summary())

    Example (training loop)::

        test = LRRangeTest(min_lr=1e-7, max_lr=1.0, num_steps=200)
        for step in range(test.num_steps):
            lr = test.next_lr()
            set_lr(optimizer, lr)
            loss = train_step(batch)
            test.record_loss(loss)
            if test.has_diverged():
                break
        result = test.finish()
    """

    def __init__(
        self,
        min_lr: float = 1e-7,
        max_lr: float = 10.0,
        num_steps: int = 200,
        smoothing: float = 0.05,
        diverge_threshold: float = 5.0,
    ) -> None:
        """
        Args:
            min_lr: Starting (minimum) learning rate.
            max_lr: Ending (maximum) learning rate.
            num_steps: Number of steps in the sweep.
            smoothing: Exponential moving average factor for loss smoothing.
            diverge_threshold: If loss exceeds best_loss * diverge_threshold,
                sweep is stopped.
        """
        if min_lr <= 0:
            raise ValueError("min_lr must be positive")
        if max_lr <= min_lr:
            raise ValueError("max_lr must be greater than min_lr")
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2")

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.smoothing = smoothing
        self.diverge_threshold = diverge_threshold

        # State
        self._step: int = 0
        self._best_loss: float = float("inf")
        self._smooth_loss: float = 0.0
        self._lr_values: List[float] = []
        self._losses: List[float] = []
        self._diverged: bool = False

        # Precompute LR schedule (log-linear sweep)
        log_min = math.log(min_lr)
        log_max = math.log(max_lr)
        self._schedule: List[float] = [
            math.exp(log_min + (log_max - log_min) * i / max(num_steps - 1, 1))
            for i in range(num_steps)
        ]

    # ------------------------------------------------------------------
    # Iterator-mode API
    # ------------------------------------------------------------------

    def next_lr(self) -> float:
        """Return the next LR in the schedule (does NOT advance step)."""
        if self._step >= len(self._schedule):
            return self._schedule[-1]
        return self._schedule[self._step]

    def record_loss(self, loss: float) -> None:
        """Record the loss for the current step and advance."""
        if not math.isfinite(loss):
            self._diverged = True
            return

        lr = self._schedule[min(self._step, len(self._schedule) - 1)]

        # Exponential moving average smoothing
        if self._step == 0:
            self._smooth_loss = loss
        else:
            self._smooth_loss = self.smoothing * loss + (1 - self.smoothing) * self._smooth_loss

        self._lr_values.append(lr)
        self._losses.append(self._smooth_loss)

        if self._smooth_loss < self._best_loss:
            self._best_loss = self._smooth_loss

        if self._smooth_loss > self._best_loss * self.diverge_threshold:
            self._diverged = True

        self._step += 1

    def has_diverged(self) -> bool:
        """True if the loss has diverged during the sweep."""
        return self._diverged

    def finish(self) -> LRRangeResult:
        """Compute and return the final result."""
        return _analyse_sweep(
            self._lr_values,
            self._losses,
            self.num_steps,
        )

    # ------------------------------------------------------------------
    # Simulation API
    # ------------------------------------------------------------------

    def run(self, loss_fn: Callable[[float], float]) -> LRRangeResult:
        """Run the sweep using a synthetic loss function.

        Args:
            loss_fn: Callable taking an LR value, returning a loss scalar.

        Returns:
            LRRangeResult with suggested LR boundaries.
        """
        if not FEATURE_ENABLED:
            return LRRangeResult()

        self._step = 0
        self._best_loss = float("inf")
        self._smooth_loss = 0.0
        self._lr_values = []
        self._losses = []
        self._diverged = False

        for _ in range(self.num_steps):
            lr = self.next_lr()
            loss = loss_fn(lr)
            self.record_loss(loss)
            if self._diverged:
                break

        return self.finish()


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _analyse_sweep(
    lr_values: List[float],
    losses: List[float],
    num_steps: int,
) -> LRRangeResult:
    """Identify min/max LR from a completed sweep."""
    result = LRRangeResult(
        lr_values=lr_values,
        losses=losses,
        num_steps=num_steps,
    )

    if len(losses) < 5:
        return result

    # Find the index of minimum loss
    min_idx = losses.index(min(losses))

    # max_lr: use LR just before loss starts rising significantly
    # We use 10% past the minimum as the "steepest descent" region.
    max_lr_idx = min(min_idx + max(1, len(losses) // 10), len(losses) - 1)
    result.suggested_max_lr = lr_values[max_lr_idx]

    # Detect actual divergence point
    best = min(losses)
    diverge_idx: Optional[int] = None
    for i, loss in enumerate(losses):
        if loss > best * 5.0 and i > min_idx:
            diverge_idx = i
            break
    if diverge_idx is not None:
        result.divergence_lr = lr_values[diverge_idx]
        result.suggested_max_lr = lr_values[max(0, diverge_idx - 1)]

    # min_lr: LR where loss is 10% above the loss at the optimal point,
    # but on the descent side (i.e., before the minimum)
    descent_threshold = losses[min_idx] * 1.1
    min_lr_idx = 0
    for i in range(min_idx):
        if losses[i] <= descent_threshold:
            min_lr_idx = i
            break
    result.suggested_min_lr = lr_values[min_lr_idx]

    return result


def find_optimal_lr(
    loss_fn: Callable[[float], float],
    min_lr: float = 1e-7,
    max_lr: float = 10.0,
    num_steps: int = 200,
) -> Tuple[Optional[float], Optional[float]]:
    """Convenience wrapper that returns (min_lr, max_lr) tuple.

    Args:
        loss_fn: Function mapping LR → loss (for simulation).
        min_lr: Start of LR sweep.
        max_lr: End of LR sweep.
        num_steps: Steps in the sweep.

    Returns:
        (suggested_min_lr, suggested_max_lr) or (None, None) if disabled.
    """
    if not FEATURE_ENABLED:
        return None, None
    test = LRRangeTest(min_lr=min_lr, max_lr=max_lr, num_steps=num_steps)
    result = test.run(loss_fn)
    return result.suggested_min_lr, result.suggested_max_lr
