from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ScorerResult:
    """Result from a single scorer for a single code sample."""
    score: float            # 0.0 - 1.0 normalized
    scorer_name: str        # e.g. "tsc", "eslint", "stars"
    details: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class ScorerProtocol(Protocol):
    """Interface that all scorers must implement."""

    name: str

    def score(self, code: str, metadata: dict[str, object] | None = None) -> ScorerResult: ...
    def score_batch(self, items: list[tuple[str, dict[str, object] | None]]) -> list[ScorerResult]: ...
    @staticmethod
    def is_available() -> bool: ...


@dataclass
class CompositeResult:
    """Combined result from multiple scorers."""
    overall: float                       # Weighted average 0.0-1.0
    per_scorer: dict[str, ScorerResult]  # scorer_name -> individual result
    weight: float                        # Training weight (tier mapped)


class CompositeScorer:
    """Combine multiple scorers with configurable weights."""

    # Tier-to-training-weight mapping (from CodeScorer pattern)
    DEFAULT_TIER_WEIGHTS: dict[str, float] = {
        "excellent": 2.0,   # score >= 0.8
        "good": 1.5,        # score >= 0.6
        "average": 1.0,     # score >= 0.4
        "poor": 0.3,        # score >= 0.2
        "reject": 0.0,      # score < 0.2
    }

    def __init__(
        self,
        scorers: list[tuple[ScorerProtocol, float]],
        tier_weights: dict[str, float] | None = None,
    ) -> None:
        self._scorers = scorers
        self._tier_weights = tier_weights or self.DEFAULT_TIER_WEIGHTS
        # Normalize scorer weights to sum to 1.0
        total = sum(w for _, w in self._scorers)
        self._scorers = [(s, w / total) if total > 0 else (s, 0.0) for s, w in self._scorers]

    def score(self, code: str, metadata: dict[str, object] | None = None) -> CompositeResult:
        per_scorer: dict[str, ScorerResult] = {}
        overall = 0.0
        for scorer, weight in self._scorers:
            result = scorer.score(code, metadata)
            per_scorer[result.scorer_name] = result
            overall += result.score * weight
        overall = max(0.0, min(1.0, overall))
        return CompositeResult(
            overall=overall,
            per_scorer=per_scorer,
            weight=self._score_to_weight(overall),
        )

    def score_batch(self, items: list[tuple[str, dict[str, object] | None]]) -> list[CompositeResult]:
        # Collect per-scorer batch results indexed by scorer POSITION, not name
        # (DATA-031): keying by name meant two scorers sharing a `.name` collided
        # — the second's batch overwrote the first's and both read the second's
        # scores, so score_batch() diverged from the single-item score(). Zipping
        # the parallel lists guarantees score_batch == score regardless of names.
        per_scorer_batches = [scorer.score_batch(items) for scorer, _ in self._scorers]

        results: list[CompositeResult] = []
        for i in range(len(items)):
            per_scorer: dict[str, ScorerResult] = {}
            overall = 0.0
            for (scorer, weight), batch in zip(self._scorers, per_scorer_batches):
                result = batch[i]
                per_scorer[result.scorer_name] = result
                overall += result.score * weight
            overall = max(0.0, min(1.0, overall))
            results.append(CompositeResult(
                overall=overall,
                per_scorer=per_scorer,
                weight=self._score_to_weight(overall),
            ))
        return results

    def _score_to_weight(self, score: float) -> float:
        """Map 0.0-1.0 score to training weight via tier system."""
        if score >= 0.8:
            return self._tier_weights.get("excellent", 2.0)
        elif score >= 0.6:
            return self._tier_weights.get("good", 1.5)
        elif score >= 0.4:
            return self._tier_weights.get("average", 1.0)
        elif score >= 0.2:
            return self._tier_weights.get("poor", 0.3)
        else:
            return self._tier_weights.get("reject", 0.0)

    @staticmethod
    def score_to_tier(score: float) -> str:
        """Map 0.0-1.0 score to tier name."""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "average"
        elif score >= 0.2:
            return "poor"
        else:
            return "reject"
