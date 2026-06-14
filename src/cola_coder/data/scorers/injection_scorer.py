"""Prompt-injection safety scorer (DATA-063 soft-weight variant).

The SOFT counterpart to the hard-drop `InjectionFilter` (data/filters/injection.py):
instead of removing a sample carrying prompt-injection payloads, this scorer assigns
it a LOW quality score so the composite quality weight is reduced (down-weighting),
preserving the project's reweight-over-filter preference for borderline content while
still steering the model away from learning injection payloads.

Reuses the canonical SEC-019 scanner (`security.injection_patterns.scan_injection`)
and the shared `ScoreMapper` — more payload patterns → lower score. Pure Python,
no execution.
"""

from __future__ import annotations

from cola_coder.data.scorers.protocol import ScorerResult
from cola_coder.data.scorers.utils import ScoreMapper
from cola_coder.security.injection_patterns import scan_injection

# Injection-pattern count → quality score. Clean (0) stays 1.0; each extra
# corroborating pattern drives the score (and thus the training weight) down.
_INJECTION_SCORE = ScoreMapper([(0, 1.0), (1, 0.4), (2, 0.15)], floor=0.05)


class InjectionScorer:
    """Down-weight samples carrying prompt-injection payloads (graded by severity)."""

    name: str = "injection_safety"

    def score(self, code: str, metadata: dict[str, object] | None = None) -> ScorerResult:
        hits = scan_injection(code)
        return ScorerResult(
            score=_INJECTION_SCORE(len(hits)),
            scorer_name=self.name,
            details={"injection_hits": hits, "num_hits": len(hits)},
        )

    def score_batch(
        self, items: list[tuple[str, dict[str, object] | None]]
    ) -> list[ScorerResult]:
        return [self.score(code, meta) for code, meta in items]

    @staticmethod
    def is_available() -> bool:
        return True  # Pure Python, no external dependencies
