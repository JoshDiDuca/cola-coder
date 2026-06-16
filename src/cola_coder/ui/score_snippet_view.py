"""Quality-scorer playground endpoint helper (UI-101).

Runs the project's PURE-PYTHON code scorers on an ad-hoc snippet so the UI can
show the per-scorer quality breakdown that drives training-data weighting —
WITHOUT Docker, node, a model, or the network. Execution-based scorers (``tsc``,
``eslint``) and model/network scorers (``classifier``, ``llm_judge``, ``stars``)
are intentionally EXCLUDED: they need the sandbox/toolchain or remote calls and
could contend with the live trainer. This view is therefore MAIN-SAFE.

Returns the per-scorer 0–1 score + tier, plus the unweighted mean across the
scorers that ran. (The training pipeline uses configured per-scorer weights; the
UI shows the raw signals so a user can see WHY a snippet scores as it does.)
"""

from __future__ import annotations

import logging

from cola_coder.data.scorers.protocol import CompositeScorer, ScorerProtocol

logger = logging.getLogger(__name__)


def _safe_scorers() -> list[ScorerProtocol]:
    """Instantiate the deterministic, dependency-free scorers (skip unavailable)."""
    scorers: list[ScorerProtocol] = []
    try:
        from cola_coder.data.scorers.heuristic_scorer import HeuristicScorer

        scorers.append(HeuristicScorer())
    except Exception:  # noqa: BLE001 — a missing optional scorer must not break the view
        logger.debug("heuristic scorer unavailable", exc_info=True)
    try:
        from cola_coder.data.scorers.educational_value import EducationalValueScorer

        scorers.append(EducationalValueScorer())
    except Exception:  # noqa: BLE001
        logger.debug("educational_value scorer unavailable", exc_info=True)
    try:
        from cola_coder.data.scorers.repetition_scorer import RepetitionScorer

        scorers.append(RepetitionScorer())
    except Exception:  # noqa: BLE001
        logger.debug("repetition scorer unavailable", exc_info=True)
    try:
        from cola_coder.data.scorers.injection_scorer import InjectionScorer

        scorers.append(InjectionScorer())
    except Exception:  # noqa: BLE001
        logger.debug("injection scorer unavailable", exc_info=True)
    try:
        from cola_coder.data.scorers.cwe_security import CweSecurityScorer

        scorers.append(CweSecurityScorer())
    except Exception:  # noqa: BLE001
        logger.debug("cwe_security scorer unavailable", exc_info=True)

    return [s for s in scorers if s.is_available()]


def score_snippet(code: str, metadata: dict[str, object] | None = None) -> dict:
    """Score ``code`` with the safe pure-Python scorers; return the breakdown.

    ``{"scorers": [{name, score, tier}], "mean_score", "mean_tier", "count"}``.
    Empty/whitespace ``code`` returns ``{"error": str}``. A scorer that raises is
    skipped (logged), never crashing the view. Never raises.
    """
    if not code or not code.strip():
        return {"error": "code is required"}

    breakdown: list[dict[str, object]] = []
    total = 0.0
    for scorer in _safe_scorers():
        try:
            result = scorer.score(code, metadata)
        except Exception:  # noqa: BLE001 — one bad scorer must not sink the rest
            logger.warning("scorer %s raised on snippet", getattr(scorer, "name", "?"), exc_info=True)
            continue
        score = max(0.0, min(1.0, float(result.score)))
        breakdown.append(
            {
                "name": result.scorer_name,
                "score": round(score, 4),
                "tier": CompositeScorer.score_to_tier(score),
            }
        )
        total += score

    count = len(breakdown)
    mean = total / count if count else 0.0
    return {
        "scorers": breakdown,
        "mean_score": round(mean, 4),
        "mean_tier": CompositeScorer.score_to_tier(mean),
        "count": count,
    }
