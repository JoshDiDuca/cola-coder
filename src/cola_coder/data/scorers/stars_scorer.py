"""GitHub stars scorer — use repository star count as a quality proxy."""

from __future__ import annotations

import math

from cola_coder.data.scorers.protocol import ScorerResult


class StarsScorer:
    """Score code quality based on GitHub repository star count.

    Uses log-scale normalization:
        0 stars    → 0.1
        10 stars   → 0.3
        100 stars  → 0.5
        1000 stars → 0.8
        10000+     → 1.0

    When star data is unavailable (e.g. starcoderdata parquet files),
    returns a configurable default score.
    """

    name: str = "stars"

    def __init__(self, default_score: float = 0.3) -> None:
        self._default_score = max(0.0, min(1.0, default_score))

    def score(self, code: str, metadata: dict[str, object] | None = None) -> ScorerResult:
        if metadata is None or "repo_stars" not in metadata:
            return ScorerResult(
                score=self._default_score,
                scorer_name=self.name,
                details={"stars": None, "source": "default"},
            )

        try:
            stars = int(metadata["repo_stars"])
        except (ValueError, TypeError):
            return ScorerResult(
                score=self._default_score,
                scorer_name=self.name,
                details={"stars": None, "source": "default", "parse_error": True},
            )

        normalized = self._normalize_stars(stars)
        return ScorerResult(
            score=normalized,
            scorer_name=self.name,
            details={"stars": stars, "source": "metadata"},
        )

    def score_batch(self, items: list[tuple[str, dict[str, object] | None]]) -> list[ScorerResult]:
        return [self.score(code, meta) for code, meta in items]

    @staticmethod
    def is_available() -> bool:
        return True  # Pure Python, no external dependencies

    @staticmethod
    def _normalize_stars(stars: int) -> float:
        """Map star count to 0.0-1.0 using log scale.

        Mapping:
            0      → 0.10
            1      → 0.10
            10     → 0.30
            100    → 0.50
            1000   → 0.80
            10000  → 1.00
            100000 → 1.00
        """
        if stars <= 0:
            return 0.1

        log_stars = math.log10(max(stars, 1))

        if log_stars <= 1:       # 1-10 stars
            return 0.1 + (log_stars / 1.0) * 0.2
        elif log_stars <= 2:     # 10-100 stars
            return 0.3 + ((log_stars - 1) / 1.0) * 0.2
        elif log_stars <= 3:     # 100-1000 stars
            return 0.5 + ((log_stars - 2) / 1.0) * 0.3
        elif log_stars <= 4:     # 1000-10000 stars
            return 0.8 + ((log_stars - 3) / 1.0) * 0.2
        else:                    # 10000+ stars
            return 1.0
