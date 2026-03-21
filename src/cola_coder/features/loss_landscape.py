"""Loss Landscape Analyzer: explore the geometry of the loss surface.

Samples loss values along random directions in weight space, computes
sharpness metrics (Hessian trace estimate via finite differences), and
classifies minima as flat or sharp.

Designed to work with any callable loss function — no framework required.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the loss landscape analyzer feature is active."""
    return FEATURE_ENABLED


# Type alias: a weight vector is a flat list of floats
WeightVector = list[float]
LossFn = Callable[[WeightVector], float]


@dataclass
class LandscapeSample:
    """A single point sampled along a direction in weight space."""

    alpha: float  # Step size along the direction
    loss: float  # Loss value at (theta + alpha * direction)


@dataclass
class DirectionProfile:
    """Loss profile along a single random direction."""

    samples: list[LandscapeSample] = field(default_factory=list)
    base_loss: float = 0.0
    min_loss: float = 0.0
    max_loss: float = 0.0
    curvature_estimate: float = 0.0  # Second-order finite diff at alpha=0

    def as_dict_list(self) -> list[dict]:
        return [{"alpha": s.alpha, "loss": s.loss} for s in self.samples]


@dataclass
class SharpnessMetrics:
    """Sharpness metrics for the current parameter point."""

    hessian_trace_estimate: float  # Sum of diagonal Hessian approximations
    max_curvature: float  # Maximum curvature across random directions
    mean_curvature: float  # Mean curvature across random directions
    n_directions: int
    base_loss: float
    classification: str  # "flat", "moderate", or "sharp"

    def summary(self) -> str:
        return (
            f"SharpnessMetrics(trace={self.hessian_trace_estimate:.4f}, "
            f"max_curv={self.max_curvature:.4f}, "
            f"mean_curv={self.mean_curvature:.4f}, "
            f"dirs={self.n_directions}, "
            f"class={self.classification})"
        )


def _normalize(v: WeightVector) -> WeightVector:
    """Return a unit vector."""
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-12:
        return v
    return [x / norm for x in v]


def _add_scaled(base: WeightVector, direction: WeightVector, alpha: float) -> WeightVector:
    """Compute base + alpha * direction."""
    return [b + alpha * d for b, d in zip(base, direction)]


def _random_unit_vector(dim: int, rng: random.Random) -> WeightVector:
    """Draw a random unit vector using Gaussian sampling."""
    v = [rng.gauss(0, 1) for _ in range(dim)]
    return _normalize(v)


def _finite_diff_curvature(
    loss_fn: LossFn,
    theta: WeightVector,
    direction: WeightVector,
    eps: float = 1e-3,
) -> float:
    """Estimate second derivative along direction via central finite differences.

    d²L/dα² ≈ (L(θ+εd) - 2L(θ) + L(θ-εd)) / ε²
    """
    l_plus = loss_fn(_add_scaled(theta, direction, eps))
    l_minus = loss_fn(_add_scaled(theta, direction, -eps))
    l_center = loss_fn(theta)
    return (l_plus - 2 * l_center + l_minus) / (eps * eps)


class LossLandscapeAnalyzer:
    """Analyze the local geometry of the loss surface around a weight vector."""

    def __init__(self, loss_fn: LossFn, theta: WeightVector, seed: int = 42) -> None:
        """
        Parameters
        ----------
        loss_fn:
            Callable that accepts a flat weight vector and returns a scalar loss.
        theta:
            Current parameter vector (the point to analyze around).
        seed:
            Random seed for reproducible direction sampling.
        """
        self.loss_fn = loss_fn
        self.theta = list(theta)
        self.dim = len(theta)
        self.rng = random.Random(seed)

    def sample_direction(
        self,
        direction: Optional[WeightVector] = None,
        n_points: int = 11,
        alpha_range: float = 1.0,
    ) -> DirectionProfile:
        """Sample loss along one direction.

        Parameters
        ----------
        direction:
            The direction vector.  If None, a random unit vector is used.
        n_points:
            Number of alpha values to sample (including centre).
        alpha_range:
            Half-range for alpha: alphas are in [-alpha_range, +alpha_range].
        """
        if direction is None:
            direction = _random_unit_vector(self.dim, self.rng)
        else:
            direction = _normalize(direction)

        if n_points < 1:
            n_points = 1

        step = (2 * alpha_range) / max(n_points - 1, 1)
        alphas = [-alpha_range + i * step for i in range(n_points)]

        base_loss = self.loss_fn(self.theta)
        samples = []
        for alpha in alphas:
            perturbed = _add_scaled(self.theta, direction, alpha)
            loss = self.loss_fn(perturbed)
            samples.append(LandscapeSample(alpha=alpha, loss=loss))

        losses = [s.loss for s in samples]
        curv = _finite_diff_curvature(self.loss_fn, self.theta, direction)

        return DirectionProfile(
            samples=samples,
            base_loss=base_loss,
            min_loss=min(losses),
            max_loss=max(losses),
            curvature_estimate=curv,
        )

    def sharpness(
        self,
        n_directions: int = 10,
        eps: float = 1e-3,
    ) -> SharpnessMetrics:
        """Estimate Hessian trace and sharpness via random direction sampling.

        Uses Hutchinson's estimator: E[v^T H v] = trace(H) for unit Gaussian v.

        Parameters
        ----------
        n_directions:
            Number of random directions to use for the estimate.
        eps:
            Step size for finite-difference curvature estimate.
        """
        base_loss = self.loss_fn(self.theta)
        curvatures = []
        for _ in range(n_directions):
            direction = _random_unit_vector(self.dim, self.rng)
            curv = _finite_diff_curvature(self.loss_fn, self.theta, direction, eps)
            curvatures.append(curv)

        # Hessian trace estimate = dim * mean(curvature) by Hutchinson
        trace_est = self.dim * (sum(curvatures) / len(curvatures)) if curvatures else 0.0
        max_curv = max(curvatures) if curvatures else 0.0
        mean_curv = sum(curvatures) / len(curvatures) if curvatures else 0.0

        # Classify
        if abs(mean_curv) < 0.1:
            classification = "flat"
        elif abs(mean_curv) < 1.0:
            classification = "moderate"
        else:
            classification = "sharp"

        return SharpnessMetrics(
            hessian_trace_estimate=trace_est,
            max_curvature=max_curv,
            mean_curvature=mean_curv,
            n_directions=n_directions,
            base_loss=base_loss,
            classification=classification,
        )

    def multi_direction_profiles(
        self,
        n_directions: int = 5,
        n_points: int = 11,
        alpha_range: float = 1.0,
    ) -> list[DirectionProfile]:
        """Sample loss along multiple random directions."""
        return [
            self.sample_direction(n_points=n_points, alpha_range=alpha_range)
            for _ in range(n_directions)
        ]

    def flatness_ratio(
        self,
        n_directions: int = 10,
        alpha: float = 0.1,
    ) -> float:
        """Compute flatness as the fraction of directions where loss barely increases.

        A direction is "flat" if |L(θ + α*d) - L(θ)| / max(|L(θ)|, 1) < 0.01.
        Higher ratio → flatter minimum.
        """
        base_loss = self.loss_fn(self.theta)
        flat_count = 0
        for _ in range(n_directions):
            direction = _random_unit_vector(self.dim, self.rng)
            perturbed = _add_scaled(self.theta, direction, alpha)
            perturbed_loss = self.loss_fn(perturbed)
            rel_change = abs(perturbed_loss - base_loss) / max(abs(base_loss), 1.0)
            if rel_change < 0.01:
                flat_count += 1
        return flat_count / n_directions if n_directions > 0 else 0.0
