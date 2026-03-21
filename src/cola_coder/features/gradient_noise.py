"""Gradient Noise Estimator — feature 41.

Estimates the gradient noise scale (Simple/Fisher ratio) to detect when
batch size should change during training.  The noise scale B_simple is
defined as:

    B_simple = tr(Sigma) / |G|^2

where Sigma is the gradient covariance and G is the true gradient.
In practice we estimate this from two quantities that can be computed
from per-step gradient norms:

    B_simple ≈  (batch_simple - batch_big * E[g^2]) / (batch_big * E[g]^2 - batch_simple)

Reference: McCandlish et al. "An Empirical Model of Large-Batch Training" (2018).

Feature toggle pattern (project convention):
    FEATURE_ENABLED = False  → estimator returns None / no-ops.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if gradient noise estimation is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GradientNoiseReport:
    """Summary of gradient noise estimation."""

    noise_scale: float
    """Estimated B_simple (gradient noise scale).  Higher → more noisy."""

    grad_variance: float
    """Empirical variance of gradient norms over the observation window."""

    grad_mean: float
    """Mean gradient norm over the observation window."""

    snr: float
    """Signal-to-noise ratio (mean^2 / variance).  Higher → less noisy."""

    recommended_batch_multiplier: float
    """Suggested batch size multiplier relative to current batch size.
    1.0 means current size is fine; >1 means increase; <1 means decrease."""

    step: int
    """Training step at the time of estimation."""

    def is_batch_too_small(self, threshold: float = 1.5) -> bool:
        """Return True if the noise scale suggests the batch is too small."""
        return self.recommended_batch_multiplier > threshold

    def is_batch_too_large(self, threshold: float = 0.5) -> bool:
        """Return True if the noise scale suggests the batch is wastefully large."""
        return self.recommended_batch_multiplier < threshold

    def summary(self) -> str:
        return (
            f"step={self.step} noise_scale={self.noise_scale:.3f} "
            f"snr={self.snr:.3f} batch_mult={self.recommended_batch_multiplier:.2f}"
        )


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------


class GradientNoiseEstimator:
    """Tracks gradient statistics to estimate noise scale.

    Usage::

        estimator = GradientNoiseEstimator(window=100)
        for step, batch in enumerate(dataloader):
            loss.backward()
            grad_norm = compute_grad_norm(model)
            estimator.record(grad_norm, step=step)
            report = estimator.estimate(step=step)
            if report and report.is_batch_too_small():
                print("Consider increasing batch size")
    """

    def __init__(
        self,
        window: int = 100,
        min_samples: int = 20,
        target_snr: float = 10.0,
    ) -> None:
        """
        Args:
            window: Number of recent gradient norms to keep.
            min_samples: Minimum samples needed before estimation.
            target_snr: SNR value considered adequate (used to compute multiplier).
        """
        self.window = window
        self.min_samples = min_samples
        self.target_snr = target_snr
        self._norms: Deque[float] = deque(maxlen=window)
        self._step: int = 0

    def record(self, grad_norm: float, step: Optional[int] = None) -> None:
        """Record a gradient norm value.

        Args:
            grad_norm: L2 norm of the gradient at this step.
            step: Optional explicit step counter; otherwise auto-incremented.
        """
        if not FEATURE_ENABLED:
            return
        if not math.isfinite(grad_norm):
            return
        self._norms.append(float(grad_norm))
        self._step = step if step is not None else self._step + 1

    def estimate(self, step: Optional[int] = None) -> Optional[GradientNoiseReport]:
        """Compute a noise scale estimate from the current window.

        Returns None if fewer than ``min_samples`` have been collected or if
        the feature is disabled.
        """
        if not FEATURE_ENABLED:
            return None
        if len(self._norms) < self.min_samples:
            return None

        norms = list(self._norms)
        n = len(norms)

        mean_norm = sum(norms) / n
        variance = sum((x - mean_norm) ** 2 for x in norms) / max(n - 1, 1)

        # Noise scale: variance / mean^2 (unnormalised version of B_simple)
        if mean_norm < 1e-12:
            noise_scale = float("inf")
            snr = 0.0
        else:
            noise_scale = math.sqrt(variance) / (mean_norm + 1e-12)
            snr = (mean_norm**2) / (variance + 1e-12)

        # Recommend batch multiplier: if SNR < target, increase batch.
        # Ratio of target_snr / actual_snr gives a rough scaling factor.
        if snr > 1e-9:
            batch_mult = math.sqrt(self.target_snr / snr)
        else:
            batch_mult = 4.0  # heavily noisy, suggest large increase

        # Clamp to reasonable range
        batch_mult = max(0.1, min(10.0, batch_mult))

        current_step = step if step is not None else self._step
        return GradientNoiseReport(
            noise_scale=noise_scale,
            grad_variance=variance,
            grad_mean=mean_norm,
            snr=snr,
            recommended_batch_multiplier=batch_mult,
            step=current_step,
        )

    def reset(self) -> None:
        """Clear recorded history."""
        self._norms.clear()
        self._step = 0

    @property
    def num_samples(self) -> int:
        """Number of gradient norms currently in the window."""
        return len(self._norms)


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------


def estimate_noise_scale(
    grad_norms: List[float],
    target_snr: float = 10.0,
) -> Optional[GradientNoiseReport]:
    """One-shot estimation from a list of gradient norms.

    Args:
        grad_norms: List of gradient L2 norms.
        target_snr: Target signal-to-noise ratio for batch size recommendation.

    Returns:
        GradientNoiseReport or None if insufficient data.
    """
    if not FEATURE_ENABLED:
        return None
    estimator = GradientNoiseEstimator(
        window=len(grad_norms),
        min_samples=min(20, len(grad_norms)),
        target_snr=target_snr,
    )
    for i, norm in enumerate(grad_norms):
        estimator.record(norm, step=i)
    return estimator.estimate(step=len(grad_norms) - 1)
