"""Token Statistics Tracker: record and analyse token streams during inference.

Tracks per-run statistics and accumulates them across many inference calls.
Useful for understanding model output diversity, vocabulary usage patterns,
and repetition tendencies.

For a TS dev: think of it like an analytics collector — you call .record()
each generation, then .summary() to get aggregated metrics.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TokenStats:
    """Statistics for a single recorded token sequence."""

    total_tokens: int
    unique_tokens: int
    type_token_ratio: float  # unique / total  (higher = more diverse)
    top_tokens: list[tuple[int, int]]  # (token_id, count) sorted by freq desc
    repeated_fraction: float  # fraction of tokens that appear more than once


@dataclass
class AggregateStats:
    """Aggregated statistics across all recorded sequences."""

    num_sequences: int
    total_tokens: int
    global_unique_tokens: int
    global_type_token_ratio: float
    avg_tokens_per_sequence: float
    avg_type_token_ratio: float
    avg_diversity_score: float
    global_top_tokens: list[tuple[int, int]]  # (token_id, count) sorted desc


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class TokenStatsTracker:
    """Record and analyse token sequences produced during inference.

    Example::

        tracker = TokenStatsTracker()
        for output_ids in generated_batches:
            tracker.record(output_ids)

        print(tracker.summary())
        print(f"diversity = {tracker.diversity_score():.3f}")
    """

    def __init__(self, max_top_k: int = 20) -> None:
        """
        Parameters
        ----------
        max_top_k:
            How many top tokens to include in reports.
        """
        self.max_top_k = max_top_k
        self._global_counter: Counter[int] = Counter()
        self._per_sequence: list[TokenStats] = []
        self._total_tokens: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record(self, tokens: list[int]) -> TokenStats:
        """Record a token sequence and return its per-sequence stats."""
        if not tokens:
            stats = TokenStats(
                total_tokens=0,
                unique_tokens=0,
                type_token_ratio=0.0,
                top_tokens=[],
                repeated_fraction=0.0,
            )
            self._per_sequence.append(stats)
            return stats

        counter: Counter[int] = Counter(tokens)
        total = len(tokens)
        unique = len(counter)
        ttr = unique / total if total > 0 else 0.0
        top_k = counter.most_common(self.max_top_k)
        repeated = sum(1 for t in tokens if counter[t] > 1)
        repeated_frac = repeated / total if total > 0 else 0.0

        stats = TokenStats(
            total_tokens=total,
            unique_tokens=unique,
            type_token_ratio=ttr,
            top_tokens=top_k,
            repeated_fraction=repeated_frac,
        )
        self._per_sequence.append(stats)
        self._global_counter.update(counter)
        self._total_tokens += total
        return stats

    def summary(self) -> AggregateStats:
        """Return aggregated statistics across all recorded sequences."""
        n = len(self._per_sequence)
        if n == 0:
            return AggregateStats(
                num_sequences=0,
                total_tokens=0,
                global_unique_tokens=0,
                global_type_token_ratio=0.0,
                avg_tokens_per_sequence=0.0,
                avg_type_token_ratio=0.0,
                avg_diversity_score=0.0,
                global_top_tokens=[],
            )

        avg_ttr = sum(s.type_token_ratio for s in self._per_sequence) / n
        avg_len = self._total_tokens / n
        global_unique = len(self._global_counter)
        global_ttr = global_unique / max(self._total_tokens, 1)
        global_top = self._global_counter.most_common(self.max_top_k)
        avg_div = sum(self._diversity_for_stats(s) for s in self._per_sequence) / n

        return AggregateStats(
            num_sequences=n,
            total_tokens=self._total_tokens,
            global_unique_tokens=global_unique,
            global_type_token_ratio=global_ttr,
            avg_tokens_per_sequence=avg_len,
            avg_type_token_ratio=avg_ttr,
            avg_diversity_score=avg_div,
            global_top_tokens=global_top,
        )

    def diversity_score(self) -> float:
        """Return a single 0.0–1.0 diversity score across all sequences.

        Combines type-token ratio with vocabulary entropy.  Higher means the
        model uses a wider, more varied vocabulary.
        """
        if not self._global_counter:
            return 0.0
        # Normalised entropy of global token distribution
        total = sum(self._global_counter.values())
        entropy = -sum(
            (c / total) * math.log2(c / total)
            for c in self._global_counter.values()
            if c > 0
        )
        max_entropy = math.log2(len(self._global_counter)) if len(self._global_counter) > 1 else 1.0
        normalised_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Blend with average TTR
        agg = self.summary()
        ttr_score = agg.avg_type_token_ratio
        return 0.5 * normalised_entropy + 0.5 * ttr_score

    def reset(self) -> None:
        """Clear all recorded data."""
        self._global_counter.clear()
        self._per_sequence.clear()
        self._total_tokens = 0

    def num_sequences(self) -> int:
        """Return the number of sequences recorded so far."""
        return len(self._per_sequence)

    def __repr__(self) -> str:
        agg = self.summary()
        return (
            f"TokenStatsTracker(sequences={agg.num_sequences}, "
            f"total_tokens={agg.total_tokens}, "
            f"diversity={self.diversity_score():.3f})"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _diversity_for_stats(stats: TokenStats) -> float:
        """Per-sequence diversity score based on TTR."""
        return stats.type_token_ratio
