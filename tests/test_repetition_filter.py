"""Tests for the Gopher/MassiveText repetition filter (DATA-071).

Pure-Python, CPU-only — never imports torch, never loads a model. Exercises the
metric math, the published thresholds, the short-doc n-gram guard, config
overrides, and the FilterPlugin registry wiring.
"""

from __future__ import annotations

import pytest

from cola_coder.data.filters.repetition import (
    RepetitionFilter,
    RepetitionThresholds,
    compute_repetition_metrics,
)
from cola_coder.data.pipeline import DataRecord
from cola_coder.data.registry import get_filter, list_filters


def _rec(content: str) -> DataRecord:
    return DataRecord(content=content, metadata={})


# ---------------------------------------------------------------------------
# Metric math
# ---------------------------------------------------------------------------

def test_no_repetition_metrics_are_zero() -> None:
    # Every word distinct → no repeated lines, paragraphs, or n-grams.
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    m = compute_repetition_metrics(text)
    assert m.dup_line_frac == 0.0
    assert m.dup_para_frac == 0.0
    assert m.dup_line_char_frac == 0.0
    assert m.dup_para_char_frac == 0.0
    assert all(v == 0.0 for v in m.top_ngram_char_frac.values())
    assert all(v == 0.0 for v in m.dup_ngram_char_frac.values())


def test_duplicate_line_fraction_counts_extra_occurrences() -> None:
    # 4 lines, one of which is the same line repeated → 1 line is a duplicate.
    text = "alpha\nbeta\nalpha\ngamma\n"
    m = compute_repetition_metrics(text)
    # 1 duplicate out of 4 lines = 0.25.
    assert m.dup_line_frac == pytest.approx(0.25)
    assert m.dup_line_char_frac > 0.0


def test_all_identical_lines_saturates_fraction() -> None:
    text = "x = 1\n" * 10
    m = compute_repetition_metrics(text)
    # 9 of 10 lines are duplicates.
    assert m.dup_line_frac == pytest.approx(0.9)
    assert m.dup_line_char_frac == pytest.approx(0.9)


def test_duplicate_paragraph_fraction() -> None:
    para = "first paragraph line one\nfirst paragraph line two"
    text = f"{para}\n\nother paragraph\n\n{para}"
    m = compute_repetition_metrics(text)
    # 3 paragraphs, the repeated one contributes 1 duplicate → 1/3.
    assert m.dup_para_frac == pytest.approx(1 / 3)
    assert m.dup_para_char_frac > 0.0


def test_top_ngram_fraction_high_for_looping_phrase() -> None:
    # A looping 2-gram dominates the document.
    text = ("foo bar " * 40).strip()
    m = compute_repetition_metrics(text)
    assert m.top_ngram_char_frac[2] > 0.5


def test_dup_ngram_fraction_high_for_repeated_span() -> None:
    text = ("the quick brown fox jumps " * 20).strip()
    m = compute_repetition_metrics(text)
    # Long repeated spans → most 5-gram positions are covered.
    assert m.dup_ngram_char_frac[5] > 0.5


def test_metrics_bounded_zero_to_one() -> None:
    text = "lorem ipsum dolor sit amet " * 30
    m = compute_repetition_metrics(text)
    for value in (
        m.dup_line_frac,
        m.dup_para_frac,
        m.dup_line_char_frac,
        m.dup_para_char_frac,
        *m.top_ngram_char_frac.values(),
        *m.dup_ngram_char_frac.values(),
    ):
        assert 0.0 <= value <= 1.0


def test_empty_and_whitespace_are_safe() -> None:
    for text in ("", "   \n\n  \t  "):
        m = compute_repetition_metrics(text)
        assert m.dup_line_frac == 0.0
        assert m.top_ngram_char_frac[2] == 0.0
        assert m.dup_ngram_char_frac[10] == 0.0


# ---------------------------------------------------------------------------
# Filter accept/reject behavior
# ---------------------------------------------------------------------------

def test_clean_code_passes() -> None:
    f = RepetitionFilter()
    code = (
        "import math\n\n"
        "def area(radius: float) -> float:\n"
        "    return math.pi * radius * radius\n\n"
        "def perimeter(radius: float) -> float:\n"
        "    return 2 * math.pi * radius\n"
    )
    keep, reason = f.check(_rec(code))
    assert keep is True
    assert reason == ""


def test_empty_content_passes() -> None:
    f = RepetitionFilter()
    keep, reason = f.check(_rec(""))
    assert keep is True
    assert reason == ""


def test_repeated_lines_rejected_with_line_reason() -> None:
    f = RepetitionFilter()
    text = "log.info('processing')\n" * 30
    keep, reason = f.check(_rec(text))
    assert keep is False
    assert "dup_line" in reason


def test_looping_ngram_rejected() -> None:
    f = RepetitionFilter()
    # All distinct lines (no line/paragraph dup) but a dominant repeated phrase.
    text = "\n".join(f"value_{i} = compute the same the same the same" for i in range(60))
    keep, reason = f.check(_rec(text))
    assert keep is False
    assert "gram" in reason


def test_short_doc_skips_ngram_screen() -> None:
    f = RepetitionFilter()
    # < min_words: a dominant n-gram should NOT trigger rejection because the
    # n-gram screen is skipped for tiny docs (line/para still apply, but these
    # lines are all distinct).
    text = "a b a b a b"  # 6 words, well under default min_words=50
    keep, _ = f.check(_rec(text))
    assert keep is True


def test_min_words_guard_is_configurable() -> None:
    f = RepetitionFilter()
    f.setup({"min_words": 4})
    # Now the same dominant 2-gram document is long enough to be screened.
    text = "a b a b a b a b a b"  # 10 words, top 2-gram dominates
    keep, reason = f.check(_rec(text))
    assert keep is False
    assert "gram" in reason


# ---------------------------------------------------------------------------
# Thresholds & config
# ---------------------------------------------------------------------------

def test_default_thresholds_match_published_gopher_values() -> None:
    t = RepetitionThresholds()
    assert t.dup_line_frac == 0.30
    assert t.dup_para_frac == 0.30
    assert t.dup_line_char_frac == 0.20
    assert t.dup_para_char_frac == 0.20
    assert t.top_ngram_char_frac == {2: 0.20, 3: 0.18, 4: 0.16}
    assert t.dup_ngram_char_frac == {
        5: 0.15, 6: 0.14, 7: 0.13, 8: 0.12, 9: 0.11, 10: 0.10
    }


def test_setup_overrides_line_threshold() -> None:
    f = RepetitionFilter()
    # 12 distinct lines + 2 duplicate occurrences → dup_line_frac ≈ 0.14:
    # below the 0.30 default (passes) but above a stricter 0.10 bound (fails).
    distinct = "\n".join(f"step_{i} = run_phase_{i}()" for i in range(12))
    text = distinct + "\nstep_0 = run_phase_0()\nstep_1 = run_phase_1()\n"
    metrics = compute_repetition_metrics(text)
    assert 0.10 < metrics.dup_line_frac < 0.30
    assert f.check(_rec(text))[0] is True
    f.setup({"dup_line_frac": 0.10})
    keep, reason = f.check(_rec(text))
    assert keep is False
    assert "dup_line_frac" in reason


def test_setup_overrides_ngram_thresholds() -> None:
    f = RepetitionFilter()
    # Disable BOTH n-gram screens (top 2-4 and duplicate 5-10) with 1.0 caps;
    # only STRICTLY greater rejects, so a fully looping phrase now passes.
    f.setup(
        {
            "top_ngram_char_frac": {2: 1.0, 3: 1.0, 4: 1.0},
            "dup_ngram_char_frac": {5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0},
        }
    )
    text = "foo bar " * 60
    keep, _ = f.check(_rec(text))
    assert keep is True


def test_loose_thresholds_via_dataclass_pass_everything() -> None:
    loose = RepetitionThresholds(
        dup_line_frac=1.0,
        dup_para_frac=1.0,
        dup_line_char_frac=1.0,
        dup_para_char_frac=1.0,
        top_ngram_char_frac={2: 1.0, 3: 1.0, 4: 1.0},
        dup_ngram_char_frac={5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0},
    )
    f = RepetitionFilter(thresholds=loose)
    keep, _ = f.check(_rec("x = 1\n" * 50))
    assert keep is True


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------

def test_filter_is_registered() -> None:
    # Importing the package fires the @register_filter decorator.
    import cola_coder.data.filters  # noqa: F401

    assert "repetition" in list_filters()
    cls = get_filter("repetition")
    assert cls is RepetitionFilter
    assert cls().name() == "repetition"
