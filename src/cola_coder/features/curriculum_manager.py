"""Training Curriculum Manager — feature 48.

Manages training data ordering by difficulty.  Supports three strategies:

- **easy_to_hard**: start with low-difficulty samples, gradually introduce
  harder ones (classical curriculum learning, Bengio et al. 2009).
- **hard_to_easy**: reverse — start with hard samples (anti-curriculum).
- **mixed**: interleave easy and hard samples with a configurable easy/hard
  ratio that shifts over time.

Each sample is associated with a difficulty score (0.0 = easiest, 1.0 =
hardest).  The manager tracks which difficulty bands have been "seen" and
exposes statistics for monitoring.

Feature toggle pattern (project convention):
    FEATURE_ENABLED = False → manager returns samples in original order.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Sequence, TypeVar


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if curriculum management is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Enums & types
# ---------------------------------------------------------------------------


class CurriculumStrategy(str, Enum):
    """Ordering strategies for training samples."""

    EASY_TO_HARD = "easy_to_hard"
    HARD_TO_EASY = "hard_to_easy"
    MIXED = "mixed"
    RANDOM = "random"


T = TypeVar("T")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CurriculumStats:
    """Progress snapshot for a curriculum run."""

    step: int
    total_samples: int
    samples_seen: int
    current_difficulty_ceiling: float
    """Max difficulty currently allowed (for easy_to_hard mode)."""
    band_coverage: Dict[str, float]
    """Fraction of each difficulty band that has been sampled."""
    strategy: str

    def summary(self) -> str:
        return (
            f"CurriculumStats step={self.step} "
            f"strategy={self.strategy} "
            f"seen={self.samples_seen}/{self.total_samples} "
            f"diff_ceiling={self.current_difficulty_ceiling:.2f}"
        )


# ---------------------------------------------------------------------------
# Sample record
# ---------------------------------------------------------------------------


@dataclass
class ScoredSample(Generic[T]):
    """A training sample paired with its difficulty score."""

    data: T
    difficulty: float
    """0.0 = easiest, 1.0 = hardest."""
    index: int
    """Original index in the dataset."""


# ---------------------------------------------------------------------------
# Core manager
# ---------------------------------------------------------------------------


class CurriculumManager(Generic[T]):
    """Manages difficulty-ordered training data sampling.

    Usage::

        samples = [ScoredSample(data=x, difficulty=d, index=i)
                   for i, (x, d) in enumerate(zip(dataset, difficulties))]
        manager = CurriculumManager(
            samples=samples,
            strategy=CurriculumStrategy.EASY_TO_HARD,
            warmup_steps=1000,
            total_steps=10000,
        )
        for step in range(total_steps):
            batch = manager.next_batch(batch_size=32, step=step)
            train(batch)
    """

    def __init__(
        self,
        samples: Sequence[ScoredSample[T]],
        strategy: CurriculumStrategy = CurriculumStrategy.EASY_TO_HARD,
        warmup_steps: int = 0,
        total_steps: int = 10000,
        num_bands: int = 5,
        mixed_easy_ratio: float = 0.7,
        seed: int = 42,
    ) -> None:
        """
        Args:
            samples: All training samples with difficulty scores.
            strategy: Ordering strategy.
            warmup_steps: Steps before full difficulty unlocked (easy_to_hard).
            total_steps: Total training steps (used to compute difficulty schedule).
            num_bands: Number of difficulty buckets for band-coverage tracking.
            mixed_easy_ratio: For MIXED strategy — starting fraction of easy samples.
                Decays linearly to 0.5 over total_steps.
            seed: Random seed for reproducible shuffling.
        """
        self.samples = list(samples)
        self.strategy = strategy
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.num_bands = num_bands
        self.mixed_easy_ratio_start = mixed_easy_ratio
        self._rng = random.Random(seed)

        # Sort by difficulty once
        self._sorted_easy_to_hard = sorted(self.samples, key=lambda s: s.difficulty)
        self._sorted_hard_to_easy = list(reversed(self._sorted_easy_to_hard))

        # Track which samples (by index) have been seen
        self._seen: set = set()
        self._step: int = 0

        # Pre-compute bands
        self._bands = _compute_bands(self.samples, num_bands)

    def next_batch(
        self,
        batch_size: int,
        step: Optional[int] = None,
    ) -> List[ScoredSample[T]]:
        """Draw the next batch of samples according to the current curriculum.

        Args:
            batch_size: Number of samples to return.
            step: Current training step (auto-incremented if None).

        Returns:
            List of ScoredSample objects.
        """
        current_step = step if step is not None else self._step
        self._step = current_step + 1

        if not FEATURE_ENABLED or self.strategy == CurriculumStrategy.RANDOM:
            pool = self.samples
        elif self.strategy == CurriculumStrategy.EASY_TO_HARD:
            ceiling = self._difficulty_ceiling(current_step)
            pool = [s for s in self._sorted_easy_to_hard if s.difficulty <= ceiling]
            if not pool:
                pool = self._sorted_easy_to_hard[:1]  # always at least one
        elif self.strategy == CurriculumStrategy.HARD_TO_EASY:
            # Invert: start with hardest, floor rises toward easiest
            floor = 1.0 - self._difficulty_ceiling(current_step)
            pool = [s for s in self._sorted_hard_to_easy if s.difficulty >= floor]
            if not pool:
                pool = self._sorted_hard_to_easy[:1]
        elif self.strategy == CurriculumStrategy.MIXED:
            pool = self._mixed_pool(current_step)
        else:
            pool = self.samples

        # Sample with replacement if pool < batch_size
        if len(pool) <= batch_size:
            batch = list(pool)
            # Pad with random from full pool
            batch += self._rng.choices(self.samples, k=batch_size - len(batch))
        else:
            batch = self._rng.sample(pool, batch_size)

        for s in batch:
            self._seen.add(s.index)

        return batch

    def stats(self, step: Optional[int] = None) -> CurriculumStats:
        """Return a progress snapshot."""
        current_step = step if step is not None else self._step
        ceiling = self._difficulty_ceiling(current_step)
        band_coverage = self._band_coverage()
        return CurriculumStats(
            step=current_step,
            total_samples=len(self.samples),
            samples_seen=len(self._seen),
            current_difficulty_ceiling=ceiling,
            band_coverage=band_coverage,
            strategy=self.strategy.value,
        )

    def reset(self) -> None:
        """Reset the seen-sample tracking and step counter."""
        self._seen.clear()
        self._step = 0

    @property
    def num_seen(self) -> int:
        """Number of unique samples seen so far."""
        return len(self._seen)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _difficulty_ceiling(self, step: int) -> float:
        """Compute current max difficulty (min_start→1 over warmup_steps).

        Starts at 0.05 (5%) so the very first step is not empty, then ramps
        up to 1.0 linearly over warmup_steps.
        """
        if self.warmup_steps <= 0:
            return 1.0
        min_start = 0.05
        progress = min(step / self.warmup_steps, 1.0)
        return min_start + (1.0 - min_start) * progress

    def _mixed_pool(self, step: int) -> List[ScoredSample[T]]:
        """Return a pool with decreasing easy/hard ratio over training."""
        if not self.samples:
            return []
        # Easy ratio: starts at mixed_easy_ratio_start, decays to 0.5
        progress = min(step / max(self.total_steps, 1), 1.0)
        easy_ratio = self.mixed_easy_ratio_start - (self.mixed_easy_ratio_start - 0.5) * progress
        threshold = 0.5  # split point
        easy = [s for s in self.samples if s.difficulty <= threshold]
        hard = [s for s in self.samples if s.difficulty > threshold]
        if not easy:
            return hard or self.samples
        if not hard:
            return easy or self.samples
        n_easy = max(1, int(len(self.samples) * easy_ratio))
        n_hard = max(1, len(self.samples) - n_easy)
        pool_easy = self._rng.sample(easy, min(n_easy, len(easy)))
        pool_hard = self._rng.sample(hard, min(n_hard, len(hard)))
        return pool_easy + pool_hard

    def _band_coverage(self) -> Dict[str, float]:
        """Fraction of each difficulty band sampled."""
        coverage: Dict[str, float] = {}
        for band_key, indices in self._bands.items():
            if not indices:
                coverage[band_key] = 0.0
                continue
            seen_in_band = sum(1 for idx in indices if idx in self._seen)
            coverage[band_key] = seen_in_band / len(indices)
        return coverage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_bands(
    samples: List[ScoredSample],
    num_bands: int,
) -> Dict[str, List[int]]:
    """Assign each sample to a difficulty band and return band → index lists."""
    bands: Dict[str, List[int]] = {str(i): [] for i in range(num_bands)}
    for s in samples:
        band_idx = min(int(s.difficulty * num_bands), num_bands - 1)
        bands[str(band_idx)].append(s.index)
    return bands


def make_scored_samples(
    data: Sequence[Any],
    difficulties: Sequence[float],
) -> List[ScoredSample]:
    """Convenience function to pair data with difficulty scores."""
    return [
        ScoredSample(data=d, difficulty=float(diff), index=i)
        for i, (d, diff) in enumerate(zip(data, difficulties))
    ]
