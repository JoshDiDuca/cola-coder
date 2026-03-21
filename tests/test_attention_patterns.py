"""Tests for attention_patterns.py."""

from __future__ import annotations

import pytest

from cola_coder.features.attention_patterns import (
    FEATURE_ENABLED,
    PatternComparison,
    causal_pattern,
    compare_attention,
    dilated_pattern,
    entropy_of_attention,
    get_pattern,
    global_pattern,
    is_enabled,
    list_patterns,
    local_pattern,
    softmax_rows,
    strided_pattern,
)


class TestIsEnabled:
    def test_feature_enabled(self):
        assert FEATURE_ENABLED is True
        assert is_enabled() is True


class TestPatternBuilders:
    def test_local_pattern_shape(self):
        p = local_pattern(8, window=2)
        assert p.seq_len == 8
        assert len(p.mask) == 8
        assert all(len(row) == 8 for row in p.mask)

    def test_local_pattern_self_attend(self):
        p = local_pattern(8, window=2)
        for i in range(8):
            assert p.mask[i][i] == 1.0

    def test_global_pattern_all_ones(self):
        p = global_pattern(4)
        for row in p.mask:
            assert all(v == 1.0 for v in row)

    def test_causal_lower_triangular(self):
        p = causal_pattern(5)
        for i in range(5):
            for j in range(5):
                if j <= i:
                    assert p.mask[i][j] == 1.0
                else:
                    assert p.mask[i][j] == 0.0

    def test_strided_pattern_coverage(self):
        p = strided_pattern(8, stride=2)
        # Every row must have positions 0, 2, 4, 6 attended
        for row in p.mask:
            for pos in range(0, 8, 2):
                assert row[pos] == 1.0

    def test_dilated_self_attend(self):
        p = dilated_pattern(8, dilation_base=2)
        for i in range(8):
            assert p.mask[i][i] == 1.0

    def test_pattern_names(self):
        names = list_patterns()
        assert "local" in names
        assert "global" in names
        assert "causal" in names
        assert "strided" in names
        assert "dilated" in names

    def test_get_pattern_unknown_raises(self):
        with pytest.raises(ValueError):
            get_pattern("nonexistent", seq_len=4)

    def test_get_pattern_returns_correct(self):
        p = get_pattern("global", seq_len=4)
        assert p.name == "global"


class TestCompareAttention:
    def _make_global_weights(self, num_heads: int, seq_len: int) -> list[list[list[float]]]:
        """Create uniform attention weights (each position attends equally)."""
        val = 1.0 / seq_len
        row = [val] * seq_len
        matrix = [list(row) for _ in range(seq_len)]
        return [list(matrix) for _ in range(num_heads)]

    def _make_causal_weights(self, num_heads: int, seq_len: int) -> list[list[list[float]]]:
        heads = []
        for _ in range(num_heads):
            matrix = []
            for i in range(seq_len):
                val = 1.0 / (i + 1)
                row = [val if j <= i else 0.0 for j in range(seq_len)]
                matrix.append(row)
            heads.append(matrix)
        return heads

    def test_compare_returns_comparison(self):
        weights = self._make_global_weights(4, 8)
        pattern = global_pattern(8)
        result = compare_attention(weights, pattern)
        assert isinstance(result, PatternComparison)

    def test_global_attention_matches_global_pattern(self):
        weights = self._make_global_weights(2, 6)
        pattern = global_pattern(6)
        result = compare_attention(weights, pattern)
        assert result.mean_score > 0.9

    def test_causal_attention_matches_causal_pattern(self):
        weights = self._make_causal_weights(2, 6)
        pattern = causal_pattern(6)
        result = compare_attention(weights, pattern)
        assert result.mean_score > 0.9

    def test_discrepancy_computed_when_requested(self):
        weights = self._make_global_weights(2, 4)
        pattern = global_pattern(4)
        result = compare_attention(weights, pattern, compute_discrepancy=True)
        assert result.discrepancy is not None
        assert len(result.discrepancy) == 4

    def test_empty_attention_returns_empty(self):
        result = compare_attention([], global_pattern(4))
        assert result.mean_score == 0.0

    def test_best_head_valid_index(self):
        weights = self._make_global_weights(3, 5)
        pattern = global_pattern(5)
        result = compare_attention(weights, pattern)
        assert 0 <= result.best_head < 3


class TestUtilities:
    def test_softmax_rows_sum_to_one(self):
        matrix = [[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]]
        result = softmax_rows(matrix)
        for row in result:
            assert abs(sum(row) - 1.0) < 1e-6

    def test_entropy_uniform_is_max(self):
        uniform = [[0.25, 0.25, 0.25, 0.25]]
        peaked = [[0.97, 0.01, 0.01, 0.01]]
        assert entropy_of_attention(uniform) > entropy_of_attention(peaked)

    def test_entropy_nonnegative(self):
        weights = [[0.5, 0.3, 0.2]]
        assert entropy_of_attention(weights) >= 0.0
