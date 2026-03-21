"""Tests for features/architecture_visualizer.py.

All tests are CPU-only, no model weights loaded.
"""

from __future__ import annotations

import pytest

from cola_coder.features.architecture_visualizer import (
    FEATURE_ENABLED,
    ArchitectureReport,
    ArchitectureVisualizer,
    _count_attn_params,
    _count_ffn_params,
    _fmt_params,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


TINY_CONFIG = {
    "model": {
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 4,
        "n_kv_heads": 4,
        "vocab_size": 1000,
        "max_seq_len": 512,
        "d_ffn": 1024,
    }
}

FLAT_CONFIG = {
    "d_model": 128,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 2,
    "vocab_size": 500,
    "max_seq_len": 256,
    "d_ffn": 512,
}


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


def test_feature_enabled_default():
    assert FEATURE_ENABLED is True


def test_is_enabled():
    assert is_enabled() is True


# ---------------------------------------------------------------------------
# Parameter math
# ---------------------------------------------------------------------------


def test_count_attn_params_standard():
    # d_model=768, n_heads=12, n_kv_heads=12 → Q+K+V+O = 4 * 768^2
    params = _count_attn_params(768, 12, 12)
    assert params == 4 * 768 * 768


def test_count_attn_params_gqa():
    # With GQA (n_kv_heads < n_heads), K+V are smaller
    params = _count_attn_params(256, 4, 2)
    head_dim = 256 // 4  # 64
    expected = 256 * 256 + 2 * (2 * head_dim * 256) + 256 * 256
    assert params == expected


def test_count_ffn_params():
    # SwiGLU: 3 matrices of d_model × d_ffn
    params = _count_ffn_params(256, 1024)
    assert params == 3 * 256 * 1024


def test_fmt_params_billions():
    assert "B" in _fmt_params(1_500_000_000)


def test_fmt_params_millions():
    assert "M" in _fmt_params(125_000_000)


def test_fmt_params_thousands():
    assert "K" in _fmt_params(50_000)


def test_fmt_params_small():
    assert _fmt_params(100) == "100"


# ---------------------------------------------------------------------------
# ArchitectureVisualizer
# ---------------------------------------------------------------------------


@pytest.fixture
def viz():
    return ArchitectureVisualizer()


def test_visualize_returns_report(viz):
    report = viz.visualize(TINY_CONFIG)
    assert isinstance(report, ArchitectureReport)


def test_report_has_layers(viz):
    report = viz.visualize(TINY_CONFIG)
    assert len(report.layers) > 0


def test_report_layer_types(viz):
    report = viz.visualize(TINY_CONFIG)
    types = {layer.layer_type for layer in report.layers}
    assert "Embedding" in types
    assert "GQA Attention" in types
    assert "SwiGLU FFN" in types
    assert "RMSNorm" in types


def test_report_total_params_positive(viz):
    report = viz.visualize(TINY_CONFIG)
    assert report.total_params > 0


def test_report_fields_match_config(viz):
    report = viz.visualize(TINY_CONFIG)
    assert report.n_layers == 4
    assert report.d_model == 256
    assert report.n_heads == 4
    assert report.vocab_size == 1000
    assert report.max_seq_len == 512


def test_flat_config(viz):
    """ArchitectureVisualizer should handle flat config dicts too."""
    report = viz.visualize(FLAT_CONFIG)
    assert report.n_layers == 2
    assert report.d_model == 128


def test_architecture_text_not_empty(viz):
    report = viz.visualize(TINY_CONFIG)
    assert len(report.architecture_text) > 100


def test_architecture_text_contains_totals(viz):
    report = viz.visualize(TINY_CONFIG)
    assert "TOTAL" in report.architecture_text


def test_layer_info_params_non_negative(viz):
    report = viz.visualize(TINY_CONFIG)
    for lay in report.layers:
        assert lay.params >= 0, f"{lay.name} has negative params"


def test_embedding_params_correct(viz):
    report = viz.visualize(TINY_CONFIG)
    emb_layer = next(layer for layer in report.layers if layer.layer_type == "Embedding")
    expected = 1000 * 256  # vocab_size * d_model
    assert emb_layer.params == expected


def test_output_layer_params_zero_tied(viz):
    """Output (lm_head) should have params=0 since it's weight-tied."""
    report = viz.visualize(TINY_CONFIG)
    output = next((layer for layer in report.layers if "weight-tied" in layer.name), None)
    assert output is not None
    assert output.params == 0
