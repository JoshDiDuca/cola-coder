"""Shared utilities for scorers."""
from __future__ import annotations

import hashlib


def code_hash(code: str) -> str:
    """MD5 hash of code for dedup/caching. Consistent encoding."""
    return hashlib.md5(code.encode("utf-8")).hexdigest()


class ScoreMapper:
    """Map integer counts (errors, warnings) to 0.0-1.0 scores via threshold table."""

    def __init__(self, thresholds: list[tuple[int, float]], floor: float = 0.1) -> None:
        """
        Args:
            thresholds: List of (max_count, score) tuples, ascending by max_count.
                        E.g. [(0, 1.0), (2, 0.9), (5, 0.7), (10, 0.5), (20, 0.3)]
            floor: Score for counts exceeding all thresholds.
        """
        self._thresholds = sorted(thresholds, key=lambda t: t[0])
        self._floor = floor

    def __call__(self, count: int) -> float:
        """Map count to score."""
        for threshold, score in self._thresholds:
            if count <= threshold:
                return score
        return self._floor

    def map(self, count: int) -> float:
        """Map count to score (alias for __call__)."""
        return self(count)
