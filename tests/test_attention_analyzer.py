"""Tests for attention_analyzer.py."""


import pytest

from cola_coder.features.attention_analyzer import AttentionAnalyzer, AttentionReport


@pytest.fixture
def analyzer():
    return AttentionAnalyzer()


def _uniform_attn(seq: int) -> list[list[float]]:
    """Create a uniform attention matrix (each row sums to 1)."""
    val = 1.0 / seq
    return [[val] * seq for _ in range(seq)]


def _diagonal_attn(seq: int) -> list[list[float]]:
    """Create a diagonal attention matrix (full self-attention)."""
    mat = [[0.0] * seq for _ in range(seq)]
    for i in range(seq):
        mat[i][i] = 1.0
    return mat


def _local_attn(seq: int, window: int = 2) -> list[list[float]]:
    """Attention concentrated on a local window around each query."""
    mat = [[0.0] * seq for _ in range(seq)]
    for i in range(seq):
        neighbors = [j for j in range(seq) if abs(i - j) <= window]
        val = 1.0 / len(neighbors)
        for j in neighbors:
            mat[i][j] = val
    return mat


def test_feature_enabled():
    from cola_coder.features.attention_analyzer import FEATURE_ENABLED, is_enabled

    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_single_head_single_layer(analyzer):
    """Shape (seq, seq) — single head, single layer."""
    seq = 8
    matrix = _uniform_attn(seq)
    report = analyzer.analyze(matrix)
    assert isinstance(report, AttentionReport)
    assert report.num_layers == 1
    assert report.num_heads == 1


def test_multi_head_single_layer(analyzer):
    """Shape (heads, seq, seq)."""
    seq, heads = 8, 4
    weights = [_uniform_attn(seq) for _ in range(heads)]
    report = analyzer.analyze(weights)
    assert report.num_heads == heads


def test_multi_layer_multi_head(analyzer):
    seq, heads, layers = 6, 2, 3
    weights = [[_uniform_attn(seq) for _ in range(heads)] for _ in range(layers)]
    report = analyzer.analyze(weights)
    assert report.num_layers == layers
    assert report.num_heads == heads
    assert report.seq_len == seq


def test_uniform_pattern_detected(analyzer):
    seq = 10
    weights = [[_uniform_attn(seq)]]  # 1 layer, 1 head
    report = analyzer.analyze(weights)
    assert report.head_patterns[0].pattern_type == "uniform"


def test_diagonal_pattern_detected(analyzer):
    seq = 8
    weights = [[_diagonal_attn(seq)]]
    report = analyzer.analyze(weights)
    assert report.head_patterns[0].pattern_type in ("diagonal", "local", "mixed")


def test_entropy_range(analyzer):
    seq = 8
    weights = [[_uniform_attn(seq)]]
    report = analyzer.analyze(weights)
    assert 0.0 <= report.avg_entropy <= 1.0


def test_summary_returns_string(analyzer):
    seq = 6
    weights = [[[_uniform_attn(seq) for _ in range(2)] for _ in range(2)]]
    report = analyzer.analyze(weights[0])
    s = report.summary()
    assert isinstance(s, str)
    assert "layers=" in s


def test_local_pattern_detected(analyzer):
    seq = 12
    weights = [[_local_attn(seq, window=1)]]
    report = analyzer.analyze(weights)
    hp = report.head_patterns[0]
    assert hp.locality_score > 0.5
