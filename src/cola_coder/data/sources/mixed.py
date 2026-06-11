"""Mixed data source that combines multiple sources with configurable weights.

Like a weighted round-robin load balancer — each source gets a
proportional share of the output stream.
"""

from __future__ import annotations

import random
from typing import Iterator

from cola_coder.data.pipeline import DataRecord, DataSource
from cola_coder.data.registry import register_source


@register_source("mixed")
class MixedSource(DataSource):
    """Combine multiple data sources with configurable weight ratios.

    Sources are sampled proportionally to their weights. For example,
    weights [0.7, 0.2, 0.1] means ~70% from source 1, ~20% from
    source 2, ~10% from source 3.

    Uses a simple buffered interleaving strategy: pull from each source
    in proportion to its weight, yielding records in a mixed order.

    Args:
        sources: List of (DataSource, weight) tuples.
        seed: Random seed for reproducible mixing.
    """

    def __init__(
        self,
        sources: list[tuple[DataSource, float]],
        seed: int | None = None,
    ):
        if not sources:
            raise ValueError("MixedSource requires at least one source")

        self._sources = sources
        self._seed = seed

        # Normalize weights to sum to 1.0
        total_weight = sum(w for _, w in sources)
        if total_weight <= 0:
            raise ValueError("Weights must sum to a positive number")
        self._weights = [w / total_weight for _, w in sources]

    def name(self) -> str:
        parts = []
        for (src, _), weight in zip(self._sources, self._weights):
            parts.append(f"{src.name()}:{weight:.0%}")
        return f"mixed([{', '.join(parts)}])"

    def stream(self) -> Iterator[DataRecord]:
        """Yield records from all sources, weighted by their ratios.

        Strategy: create iterators for all sources, then randomly pick
        which source to pull from next based on the weight distribution.
        When a source is exhausted, redistribute its weight to remaining
        sources.
        """
        rng = random.Random(self._seed)

        # Create iterators
        iters: list[Iterator[DataRecord] | None] = [
            src.stream() for src, _ in self._sources
        ]
        weights = list(self._weights)

        while True:
            # Find active sources
            active_indices = [i for i, it in enumerate(iters) if it is not None]
            if not active_indices:
                break

            # Pick a source based on weights
            active_weights = [weights[i] for i in active_indices]
            total = sum(active_weights)
            if total <= 0:
                break

            # Weighted random selection
            normalized = [w / total for w in active_weights]
            chosen_pos = rng.choices(
                range(len(active_indices)),
                weights=normalized,
                k=1,
            )[0]
            chosen_idx = active_indices[chosen_pos]

            # Try to get a record from the chosen source
            try:
                record = next(iters[chosen_idx])  # type: ignore[arg-type]
                yield record
            except StopIteration:
                # Source exhausted — mark it as done
                iters[chosen_idx] = None

    def estimate_size(self) -> int | None:
        """Sum of all source estimates, or None if any is unknown."""
        total = 0
        for src, _ in self._sources:
            est = src.estimate_size()
            if est is None:
                return None
            total += est
        return total
