"""Heuristic scorer — adapts existing CodeScorer to ScorerProtocol."""

from __future__ import annotations

from cola_coder.data.scorers.protocol import ScorerResult


class HeuristicScorer:
    """Wraps the existing 13-signal CodeScorer as a ScorerProtocol implementor."""

    name: str = "heuristic"

    def __init__(self) -> None:
        self._scorer: object | None = None

    def _get_scorer(self):
        if self._scorer is None:
            from cola_coder.features.code_scorer import CodeScorer
            self._scorer = CodeScorer()
        return self._scorer

    def score(self, code: str, metadata: dict[str, object] | None = None) -> ScorerResult:
        language = ""
        if metadata:
            language = str(metadata.get("language", ""))

        scorer = self._get_scorer()
        result = scorer.score(code, language)

        return ScorerResult(
            score=result.overall,
            scorer_name=self.name,
            details={
                "tier": result.tier,
                "breakdown": result.breakdown,
            },
        )

    def score_batch(self, items: list[tuple[str, dict[str, object] | None]]) -> list[ScorerResult]:
        return [self.score(code, meta) for code, meta in items]

    @staticmethod
    def is_available() -> bool:
        try:
            from cola_coder.features.code_scorer import CodeScorer
            return True
        except ImportError:
            return False
