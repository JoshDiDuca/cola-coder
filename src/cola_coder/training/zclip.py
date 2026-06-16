"""ZClip — adaptive gradient-norm spike mitigation (MODEL-054).

Implements ZClip (Kumar & Owen, arXiv:2504.02507): instead of a single fixed
``grad_clip`` value, track the gradient norm's running mean and variance via an
EMA and clip only the statistical *spikes* — the norms whose z-score exceeds a
threshold — to ``mean + z_thresh * std``. Fixed clipping either throttles every
step (too low) or lets a catastrophic spike through (too high); ZClip adapts to
the run's own gradient-norm distribution and reportedly removes loss spikes that
fixed clipping misses.

This module is a SELF-CONTAINED, pure-Python numeric utility — it imports no
torch and is NOT wired into the live trainer (which keeps its fixed
``torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)``). A future cycle can
adopt it behind a config flag in a worktree: compute the grad norm, ask the
clipper for the max-norm to apply, then call ``clip_grad_norm_(params, that)``.
Keeping it unwired means the running ``small_react_best`` job and any resume are
completely unaffected.

Usage::

    clipper = ZClipper(alpha=0.97, z_thresh=2.5, warmup=25)
    # each optimizer step, with the *pre-clip* total grad norm:
    max_norm = clipper.observe(grad_norm)        # float to clip to
    torch.nn.utils.clip_grad_norm_(params, max_norm)

State is checkpointable via ``state_dict`` / ``load_state_dict`` so a resumed run
keeps its adapted statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ZClipResult:
    """Outcome of one ``observe`` call (returned for telemetry/tests)."""

    max_norm: float       # the value to pass to clip_grad_norm_ (>= 0)
    clipped: bool         # whether this step's norm was a spike that got clipped
    z: float | None       # z-score of this norm vs the EMA (None during warmup)
    mean: float           # current EMA mean of the grad norm
    std: float            # current EMA std of the grad norm


class ZClipper:
    """Adaptive gradient-norm clipper using an EMA z-score spike test.

    Args:
        alpha: EMA decay for the mean/variance (paper default 0.97; higher = slower
            to adapt, more history).
        z_thresh: z-score above which a norm is a spike and gets clipped (paper
            default 2.5).
        warmup: steps to seed the statistics before any clipping (the EMA is
            meaningless until it has seen a few norms).
        eps: numerical floor for the std (avoids divide-by-zero on a flat start).
        clip_to_mean: if True, clip a spike down to ``mean + z_thresh*std``
            (paper); the returned ``max_norm`` is that value. If a norm is not a
            spike, ``max_norm`` is the norm itself (a no-op clip).
    """

    def __init__(
        self,
        alpha: float = 0.97,
        z_thresh: float = 2.5,
        warmup: int = 25,
        eps: float = 1e-6,
        clip_to_mean: bool = True,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if z_thresh <= 0.0:
            raise ValueError(f"z_thresh must be positive, got {z_thresh}")
        if warmup < 1:
            raise ValueError(f"warmup must be >= 1, got {warmup}")
        self.alpha = alpha
        self.z_thresh = z_thresh
        self.warmup = warmup
        self.eps = eps
        self.clip_to_mean = clip_to_mean

        self._mean: float = 0.0
        self._var: float = 0.0
        self._count: int = 0  # number of norms seen

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        return math.sqrt(max(self._var, 0.0))

    @property
    def count(self) -> int:
        return self._count

    @property
    def threshold(self) -> float | None:
        """Current spike threshold ``mean + z_thresh*std``, or None pre-warmup."""
        if self._count < self.warmup:
            return None
        return self._mean + self.z_thresh * max(self.std, self.eps)

    def observe(self, grad_norm: float) -> ZClipResult:
        """Record one grad norm; return the max-norm to clip to (and telemetry).

        During warmup the statistics are seeded and ``max_norm == grad_norm`` (no
        clipping). After warmup, a norm whose z-score exceeds ``z_thresh`` is a
        spike: it is clipped to ``mean + z_thresh*std`` and the EMA is updated with
        the CLIPPED value (so the spike does not poison the running statistics).
        Non-finite or negative norms are treated as 0.0 (defensive).
        """
        if not math.isfinite(grad_norm) or grad_norm < 0.0:
            grad_norm = 0.0

        # Warmup: seed stats, never clip.
        if self._count < self.warmup:
            self._update(grad_norm)
            return ZClipResult(
                max_norm=grad_norm, clipped=False, z=None, mean=self._mean, std=self.std,
            )

        std = max(self.std, self.eps)
        z = (grad_norm - self._mean) / std
        if z > self.z_thresh:
            max_norm = self._mean + self.z_thresh * std if self.clip_to_mean else self._mean
            used = max_norm  # update stats with the clipped value, not the spike
            self._update(used)
            return ZClipResult(
                max_norm=max_norm, clipped=True, z=z, mean=self._mean, std=self.std,
            )

        self._update(grad_norm)
        return ZClipResult(
            max_norm=grad_norm, clipped=False, z=z, mean=self._mean, std=self.std,
        )

    def _update(self, value: float) -> None:
        """EMA-update mean and variance with ``value`` (Welford-style EMA var)."""
        if self._count == 0:
            self._mean = value
            self._var = 0.0
        else:
            prev_mean = self._mean
            self._mean = self.alpha * prev_mean + (1.0 - self.alpha) * value
            # EMA of squared deviation from the PRE-update mean (stable, bias-light).
            self._var = self.alpha * self._var + (1.0 - self.alpha) * (value - prev_mean) ** 2
        self._count += 1

    def state_dict(self) -> dict[str, float | int]:
        """Serialisable state for checkpoint resume."""
        return {
            "alpha": self.alpha,
            "z_thresh": self.z_thresh,
            "warmup": self.warmup,
            "eps": self.eps,
            "clip_to_mean": float(self.clip_to_mean),
            "mean": self._mean,
            "var": self._var,
            "count": self._count,
        }

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        """Restore statistics (and config) from ``state_dict``."""
        self.alpha = float(state.get("alpha", self.alpha))
        self.z_thresh = float(state.get("z_thresh", self.z_thresh))
        self.warmup = int(state.get("warmup", self.warmup))
        self.eps = float(state.get("eps", self.eps))
        self.clip_to_mean = bool(state.get("clip_to_mean", self.clip_to_mean))
        self._mean = float(state.get("mean", 0.0))
        self._var = float(state.get("var", 0.0))
        self._count = int(state.get("count", 0))
