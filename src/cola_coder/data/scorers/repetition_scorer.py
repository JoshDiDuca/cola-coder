"""Graded Gopher-repetition quality scorer (DATA-072).

The project already has a hard-reject ``RepetitionFilter`` (Gopher/MassiveText
"any-of" thresholds) and reuses ``compute_repetition_metrics`` as one sub-signal
inside ``EducationalValueScorer``. This module exposes repetition as a STANDALONE,
independently-weightable ``CodeScorer`` signal so the pipeline can DOWN-WEIGHT
repetitive code on a continuous 0–1 scale instead of only binary-dropping it —
the data-centric "prefer reweighting over hard filtering to preserve diversity"
principle (FineWeb / DataComp-LM).

Scoring: for every Gopher metric we take ``ratio = value / threshold`` (the
reject boundary is ``ratio == 1``); the score is ``1 - min(1, max_ratio)`` over
all considered metrics. Clean code (all metrics well under threshold) scores near
1.0; code at/over any threshold scores 0.0 (it's exactly what the hard filter
would drop). Short documents skip the n-gram metrics (their statistics are noisy
below ~50 words), matching the filter's short-doc guard.

Pure-Python, CPU-only, no model load, no network — safe alongside live training.
Reuses ``compute_repetition_metrics`` / ``RepetitionThresholds`` (DRY — never
reimplements the Gopher math).
"""

from __future__ import annotations

import re

from cola_coder.data.filters.repetition import (
    RepetitionThresholds,
    compute_repetition_metrics,
)
from cola_coder.data.scorers.protocol import ScorerResult

# Below this word count the n-gram statistics are too noisy to be meaningful
# (matches RepetitionFilter._DEFAULT_MIN_WORDS); line/paragraph metrics still apply.
_DEFAULT_MIN_WORDS = 50
_WORD_RE = re.compile(r"\w+", re.UNICODE)


class RepetitionScorer:
    """Continuous 0–1 quality signal from the Gopher repetition metrics.

    1.0 = no detectable repetition; 0.0 = at or beyond the Gopher reject
    boundary on at least one metric. ``details`` reports the dominant
    (worst-ratio) metric so a low score is explainable.
    """

    name: str = "repetition"

    def __init__(
        self,
        thresholds: RepetitionThresholds | None = None,
        min_words: int = _DEFAULT_MIN_WORDS,
    ) -> None:
        self._thresholds = thresholds or RepetitionThresholds()
        self._min_words = min_words

    def score(self, code: str, metadata: dict[str, object] | None = None) -> ScorerResult:
        if not code or not code.strip():
            # No text → no repetition evidence. Other scorers handle empties.
            return ScorerResult(score=1.0, scorer_name=self.name, details={"reason": "empty"})

        metrics = compute_repetition_metrics(code)
        t = self._thresholds

        # (metric_name, value, threshold) — line/paragraph metrics always apply.
        ratios: list[tuple[str, float, float]] = [
            ("dup_line_frac", metrics.dup_line_frac, t.dup_line_frac),
            ("dup_para_frac", metrics.dup_para_frac, t.dup_para_frac),
            ("dup_line_char_frac", metrics.dup_line_char_frac, t.dup_line_char_frac),
            ("dup_para_char_frac", metrics.dup_para_char_frac, t.dup_para_char_frac),
        ]

        word_count = len(_WORD_RE.findall(code))
        if word_count >= self._min_words:
            for n, limit in t.top_ngram_char_frac.items():
                ratios.append(
                    (f"top_{n}gram_char_frac", metrics.top_ngram_char_frac.get(n, 0.0), limit)
                )
            for n, limit in t.dup_ngram_char_frac.items():
                ratios.append(
                    (f"dup_{n}gram_char_frac", metrics.dup_ngram_char_frac.get(n, 0.0), limit)
                )

        dominant_name = "none"
        max_ratio = 0.0
        for metric_name, value, threshold in ratios:
            ratio = (value / threshold) if threshold > 0 else 0.0
            if ratio > max_ratio:
                max_ratio = ratio
                dominant_name = metric_name

        score = max(0.0, 1.0 - min(1.0, max_ratio))

        return ScorerResult(
            score=score,
            scorer_name=self.name,
            details={
                "dominant_metric": dominant_name,
                "max_ratio": round(max_ratio, 4),
                "dup_line_frac": round(metrics.dup_line_frac, 4),
                "word_count": word_count,
                "ngrams_evaluated": word_count >= self._min_words,
            },
        )

    def score_batch(
        self, items: list[tuple[str, dict[str, object] | None]]
    ) -> list[ScorerResult]:
        return [self.score(code, meta) for code, meta in items]

    @staticmethod
    def is_available() -> bool:
        return True  # Pure Python, no external dependencies
