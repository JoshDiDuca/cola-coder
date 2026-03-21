"""Tests for loss_component_analyzer.py."""

from __future__ import annotations

import pytest

from cola_coder.features.loss_component_analyzer import (
    FEATURE_ENABLED,
    LossBreakdown,
    LossComponentAnalyzer,
    categorize_token,
    is_enabled,
)


class TestIsEnabled:
    def test_constant(self):
        assert FEATURE_ENABLED is True

    def test_function(self):
        assert is_enabled() is True


class TestCategorizeToken:
    def test_keyword(self):
        assert categorize_token("def") == "keyword"
        assert categorize_token("return") == "keyword"

    def test_operator(self):
        assert categorize_token("+") == "operator"

    def test_identifier(self):
        assert categorize_token("my_var") == "identifier"

    def test_whitespace(self):
        assert categorize_token("   ") == "whitespace"

    def test_numeric_literal(self):
        assert categorize_token("42") == "numeric_literal"
        assert categorize_token("3.14") == "numeric_literal"

    def test_string_literal(self):
        assert categorize_token('"hello"') == "string_literal"

    def test_other_fallback(self):
        cat = categorize_token("€")
        assert isinstance(cat, str)


class TestLossComponentAnalyzer:
    def test_analyze_returns_breakdown(self):
        analyzer = LossComponentAnalyzer()
        result = analyzer.analyze([1, 2, 3], [0.5, 1.0, 0.3])
        assert isinstance(result, LossBreakdown)

    def test_mean_loss_correct(self):
        analyzer = LossComponentAnalyzer()
        result = analyzer.analyze([1, 2, 3], [1.0, 2.0, 3.0])
        assert abs(result.mean_loss - 2.0) < 1e-6

    def test_per_position_length(self):
        analyzer = LossComponentAnalyzer()
        result = analyzer.analyze([10, 20, 30], [1.0, 2.0, 3.0])
        assert len(result.per_position) == 3

    def test_per_position_position_values(self):
        analyzer = LossComponentAnalyzer()
        result = analyzer.analyze([10, 20, 30], [1.0, 2.0, 3.0])
        positions = [p.position for p in result.per_position]
        assert positions == [0, 1, 2]

    def test_per_token_keyed_by_id(self):
        analyzer = LossComponentAnalyzer()
        result = analyzer.analyze([5, 5, 10], [1.0, 3.0, 2.0])
        assert 5 in result.per_token
        # Token 5 appears twice, mean = 2.0
        assert abs(result.per_token[5].mean_loss - 2.0) < 1e-6

    def test_length_mismatch_raises(self):
        analyzer = LossComponentAnalyzer()
        with pytest.raises(ValueError):
            analyzer.analyze([1, 2, 3], [1.0, 2.0])

    def test_empty_input_returns_empty(self):
        analyzer = LossComponentAnalyzer()
        result = analyzer.analyze([], [])
        assert result.total_tokens == 0

    def test_vocab_mapping(self):
        vocab = {1: "def", 2: "my_func"}
        analyzer = LossComponentAnalyzer(vocab=vocab)
        result = analyzer.analyze([1, 2], [0.5, 1.5])
        assert result.per_token[1].token_str == "def"
        assert result.per_token[2].token_str == "my_func"

    def test_per_category_populated(self):
        vocab = {1: "def", 2: "my_var"}
        analyzer = LossComponentAnalyzer(vocab=vocab)
        result = analyzer.analyze([1, 2], [0.5, 1.5])
        assert len(result.per_category) >= 1

    def test_analyze_batch(self):
        analyzer = LossComponentAnalyzer()
        batch_ids = [[1, 2, 3], [4, 5, 6]]
        batch_losses = [[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]]
        result = analyzer.analyze_batch(batch_ids, batch_losses)
        assert result.total_tokens == 6

    def test_hardest_tokens_sorted(self):
        analyzer = LossComponentAnalyzer(top_k=2)
        result = analyzer.analyze([1, 2, 3], [0.1, 5.0, 2.0])
        hardest = analyzer.hardest_tokens(result)
        assert hardest[0].mean_loss >= hardest[1].mean_loss

    def test_summary_str(self):
        analyzer = LossComponentAnalyzer()
        result = analyzer.analyze([1, 2, 3], [1.0, 2.0, 3.0])
        s = result.summary()
        assert "LossBreakdown" in s
