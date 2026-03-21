"""Tests for complexity_heatmap.py (feature 54)."""

import pytest

from cola_coder.features.complexity_heatmap import (
    FEATURE_ENABLED,
    ComplexityHeatmapGenerator,
    NODE_WEIGHTS,
    is_enabled,
)


@pytest.fixture
def gen():
    return ComplexityHeatmapGenerator()


SIMPLE_CODE = """\
x = 1
y = 2
z = x + y
"""

COMPLEX_CODE = """\
def factorial(n):
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        try:
            result *= i
        except Exception as e:
            raise ValueError(f"Unexpected: {e}")
    return result
"""


def test_feature_flag():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_generate_line_count(gen):
    hm = gen.generate(SIMPLE_CODE)
    assert len(hm.lines) == len(SIMPLE_CODE.splitlines())


def test_generate_line_numbers(gen):
    hm = gen.generate(SIMPLE_CODE)
    for i, lc in enumerate(hm.lines):
        assert lc.line_number == i + 1


def test_generate_source_preserved(gen):
    hm = gen.generate(SIMPLE_CODE)
    for lc, src in zip(hm.lines, SIMPLE_CODE.splitlines()):
        assert lc.source == src


def test_generate_scores_non_negative(gen):
    hm = gen.generate(COMPLEX_CODE)
    assert all(lc.score >= 0 for lc in hm.lines)


def test_complex_code_higher_total_than_simple(gen):
    hm_simple = gen.generate(SIMPLE_CODE)
    hm_complex = gen.generate(COMPLEX_CODE)
    assert hm_complex.total_score > hm_simple.total_score


def test_hottest_line_is_valid(gen):
    hm = gen.generate(COMPLEX_CODE)
    assert hm.hottest_line is not None
    assert 1 <= hm.hottest_line <= len(hm.lines)


def test_normalized_scores_range(gen):
    hm = gen.generate(COMPLEX_CODE)
    normed = hm.normalized_scores()
    assert all(0.0 <= s <= 1.0 for s in normed)
    assert max(normed) == pytest.approx(1.0)


def test_normalized_scores_all_zero_for_empty(gen):
    hm = gen.generate("")
    normed = hm.normalized_scores()
    assert all(s == 0.0 for s in normed)


def test_as_records(gen):
    hm = gen.generate(SIMPLE_CODE)
    records = hm.as_records()
    assert len(records) == len(hm.lines)
    assert all("line" in r and "score" in r and "source" in r for r in records)


def test_top_n_lines(gen):
    hm = gen.generate(COMPLEX_CODE)
    top = hm.top_n_lines(n=3)
    assert len(top) == 3
    assert top[0].score >= top[1].score >= top[2].score


def test_invalid_python_returns_zero_scores(gen):
    bad_code = "def bad(:\n    pass\n"
    hm = gen.generate(bad_code)
    assert all(lc.score == 0.0 for lc in hm.lines)
    assert hm.total_score == 0.0


def test_compare_returns_more_complex(gen):
    result = gen.compare(SIMPLE_CODE, COMPLEX_CODE)
    assert result["more_complex"] == "b"
    assert result["b_total"] > result["a_total"]
    assert result["diff"] > 0


def test_compare_equal(gen):
    result = gen.compare(SIMPLE_CODE, SIMPLE_CODE)
    assert result["more_complex"] == "equal"
    assert result["diff"] == pytest.approx(0.0)


def test_bucket_lines_count(gen):
    hm = gen.generate(COMPLEX_CODE)
    buckets = gen.bucket_lines(hm, n_buckets=5)
    assert len(buckets) == 5
    total = sum(len(b) for b in buckets)
    assert total == len(hm.lines)


def test_node_weights_has_if(gen):
    assert "If" in NODE_WEIGHTS
    assert NODE_WEIGHTS["If"] > 0


def test_custom_weights():
    weights = {"If": 100.0, "__default__": 0.0}
    gen2 = ComplexityHeatmapGenerator(node_weights=weights)
    code = "if True:\n    pass\n"
    hm = gen2.generate(code)
    # The if line should have a very high score
    assert hm.max_score >= 100.0
