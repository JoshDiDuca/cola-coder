"""Domain-detection endpoint helper for the local UI (UI-097).

Thin, MAIN-SAFE wrapper around :func:`cola_coder.features.domain_detector.detect_domain`
— a pure regex/keyword heuristic that classifies a TS/JS snippet into a framework
domain (react/nextjs/graphql/prisma/zod/testing/general). No model, no GPU, no
training-path contact. Pairs with the specialist registry (same domain keys): it
shows which specialist the confidence router would dispatch a snippet to.
"""

from __future__ import annotations

import logging

from cola_coder.features.domain_detector import detect_domain

logger = logging.getLogger(__name__)


def detect_domain_view(code: str, filename: str = "") -> dict:
    """Classify ``code`` and return ranked domain scores for the UI.

    Returns ``{"top_domain": str, "scores": [{domain, import_matches,
    keyword_matches, raw_score, confidence}]}`` sorted by confidence (highest
    first). Empty/whitespace ``code`` returns ``{"error": str}``. Never raises.
    """
    if not code.strip():
        return {"error": "code is required"}

    try:
        scores = detect_domain(code, filename)
    except Exception as exc:  # defensive — the heuristic should never raise
        logger.exception("detect_domain failed")
        return {"error": str(exc)}

    score_dicts = [
        {
            "domain": s.domain,
            "import_matches": s.import_matches,
            "keyword_matches": s.keyword_matches,
            "raw_score": round(float(s.raw_score), 4),
            "confidence": round(float(s.confidence), 4),
        }
        for s in scores
    ]
    top_domain = score_dicts[0]["domain"] if score_dicts else "general"
    return {"top_domain": top_domain, "scores": score_dicts}
