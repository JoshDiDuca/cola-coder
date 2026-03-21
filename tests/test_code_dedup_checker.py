"""Tests for code_dedup_checker.py."""

import pytest

from cola_coder.features.code_dedup_checker import CodeDedupChecker


@pytest.fixture
def checker():
    return CodeDedupChecker()


_SAMPLE_A = """\
def add(a, b):
    return a + b
"""

_SAMPLE_B = """\
def multiply(x, y):
    return x * y
"""

_SAMPLE_C = """\
def add(a, b):
    return a + b
"""  # Exact copy of SAMPLE_A


def test_feature_enabled():
    from cola_coder.features.code_dedup_checker import FEATURE_ENABLED, is_enabled

    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_identical_code_high_similarity(checker):
    result = checker.check(_SAMPLE_A, [_SAMPLE_C])
    assert result.max_similarity > 0.8


def test_different_code_low_similarity(checker):
    result = checker.check(_SAMPLE_A, [_SAMPLE_B])
    assert result.max_similarity < 0.6


def test_empty_training_samples(checker):
    result = checker.check(_SAMPLE_A, [])
    assert result.max_similarity == 0.0
    assert result.nearest_index == -1


def test_nearest_index_correct(checker):
    result = checker.check(_SAMPLE_A, [_SAMPLE_B, _SAMPLE_C])
    assert result.nearest_index == 1  # SAMPLE_C is identical


def test_is_duplicate_threshold(checker):
    result = checker.check(_SAMPLE_A, [_SAMPLE_C])
    # Identical code — should be a duplicate at any reasonable threshold
    assert result.is_duplicate(threshold=0.8) is True
    # Structurally different code — should not be a duplicate at moderate threshold
    very_different = "import os\nimport sys\n\nclass Foo:\n    def __init__(self):\n        self.data = []\n"
    different_result = checker.check(_SAMPLE_A, [very_different])
    assert different_result.is_duplicate(threshold=0.8) is False


def test_per_sample_length(checker):
    training = [_SAMPLE_B, _SAMPLE_C, "x = 1\n"]
    result = checker.check(_SAMPLE_A, training)
    assert len(result.per_sample) == 3


def test_similarity_scores_in_range(checker):
    training = [_SAMPLE_A, _SAMPLE_B, "hello world"]
    result = checker.check(_SAMPLE_A, training)
    for s in result.per_sample:
        assert 0.0 <= s <= 1.0


def test_pairwise_similarity_method(checker):
    sim = checker.similarity(_SAMPLE_A, _SAMPLE_C)
    assert sim > 0.8
    sim2 = checker.similarity(_SAMPLE_A, _SAMPLE_B)
    assert sim2 < sim


def test_summary_returns_string(checker):
    result = checker.check(_SAMPLE_A, [_SAMPLE_B])
    s = result.summary()
    assert isinstance(s, str)
    assert "max=" in s
