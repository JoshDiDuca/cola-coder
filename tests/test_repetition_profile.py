"""Corpus repetition-profile aggregation (DATA-075).

`repetition_profile` aggregates the graded RepetitionScorer over a set of samples
into a curriculum/cleanup signal: a severity histogram (distance-to-degenerate),
a dominant-metric tally, mean score, and a degenerate count. Pure-Python, MAIN-SAFE.
"""

from cola_coder.data.scorers.repetition_scorer import (
    RepetitionProfile,
    repetition_profile,
)

_CLEAN = (
    "export function add(a: number, b: number): number {\n"
    "  const sum = a + b\n"
    "  return sum\n"
    "}\n"
    "export function multiply(x: number, y: number): number {\n"
    "  const product = x * y\n"
    "  return product\n"
    "}\n"
    "const greeting = 'hello world'\n"
    "console.log(greeting.toUpperCase())\n"
)
_DEGENERATE = "console.log('spam')\n" * 40
_SEVERITY_LABELS = {"clean", "low", "medium", "high", "degenerate"}


class TestRepetitionProfile:
    def test_empty_corpus_is_zeroed(self) -> None:
        prof = repetition_profile([])
        assert isinstance(prof, RepetitionProfile)
        assert prof.count == 0
        assert prof.mean_score == 0.0
        assert prof.degenerate_count == 0
        # Histogram still carries all five bins (all zero).
        assert set(prof.severity_histogram) == _SEVERITY_LABELS
        assert sum(prof.severity_histogram.values()) == 0

    def test_histogram_bins_sum_to_count(self) -> None:
        codes = [_CLEAN, _DEGENERATE, _CLEAN]
        prof = repetition_profile(codes)
        assert prof.count == 3
        assert sum(prof.severity_histogram.values()) == 3
        assert set(prof.severity_histogram) == _SEVERITY_LABELS

    def test_clean_corpus_scores_high_no_degenerate(self) -> None:
        prof = repetition_profile([_CLEAN, _CLEAN, _CLEAN])
        assert prof.mean_score >= 0.6
        assert prof.degenerate_count == 0
        # Clean code lands in the two cleanest severity bins (none high/degenerate).
        assert prof.severity_histogram["clean"] + prof.severity_histogram["low"] == 3
        assert prof.severity_histogram["high"] == 0
        assert prof.severity_histogram["degenerate"] == 0

    def test_degenerate_sample_counted(self) -> None:
        prof = repetition_profile([_CLEAN, _DEGENERATE])
        assert prof.degenerate_count >= 1
        assert prof.severity_histogram["degenerate"] >= 1
        # The degenerate sample drags the mean below the clean-only case.
        assert prof.mean_score < repetition_profile([_CLEAN]).mean_score

    def test_dominant_metric_tally(self) -> None:
        prof = repetition_profile([_DEGENERATE])
        # Every sample contributes exactly one dominant-metric vote.
        assert sum(prof.dominant_metric_counts.values()) == 1
        # The degenerate (repeated-line) sample's worst metric is a line/char metric.
        assert any(k != "none" for k in prof.dominant_metric_counts)

    def test_mean_score_in_unit_range(self) -> None:
        prof = repetition_profile([_CLEAN, _DEGENERATE, _CLEAN])
        assert 0.0 <= prof.mean_score <= 1.0
