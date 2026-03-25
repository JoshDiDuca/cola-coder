"""Shared utilities for the scoring pipeline."""

from __future__ import annotations

import hashlib


def code_hash(code: str) -> str:
    """MD5 hash of code for dedup/caching. Used across all scorers."""
    return hashlib.md5(code.encode("utf-8")).hexdigest()


class ScoreMapper:
    """Map integer counts (errors, warnings) to 0.0-1.0 quality scores.

    Reusable across scorers that convert issue counts to scores.
    """

    def __init__(self, thresholds: list[tuple[int, float]], fallback: float = 0.1) -> None:
        """
        Args:
            thresholds: List of (max_count, score) tuples. If count <= max_count, return score.
            fallback: Score if count exceeds all thresholds.
        """
        self._thresholds = thresholds
        self._fallback = fallback

    def map(self, count: int) -> float:
        """Map a count to a score."""
        for threshold, score in self._thresholds:
            if count <= threshold:
                return score
        return self._fallback
