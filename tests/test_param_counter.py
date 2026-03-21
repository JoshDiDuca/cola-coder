"""Tests for features/param_counter.py — Feature 98.

All tests are CPU-only.  Torch model tests use a tiny mock.
"""

from __future__ import annotations


import pytest

from cola_coder.features.param_counter import (
    FEATURE_ENABLED,
    ComponentCount,
    ModelParamCounter,
    ParamBreakdown,
    ParamCountResult,
    count_from_model,
    is_enabled,
    theoretical_breakdown,
)


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


def test_feature_enabled():
    assert FEATURE_ENABLED is True


def test_is_enabled():
    assert is_enabled() is True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_counter():
    """Counter for a tiny model: 256-d, 2 layers, 4 heads."""
    return ModelParamCounter(
        vocab_size=1000,
        d_model=256,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_multiplier=8 / 3,
    )


# ---------------------------------------------------------------------------
# Theoretical breakdown
# ---------------------------------------------------------------------------


def test_theoretical_returns_breakdown():
    bd = theoretical_breakdown(
        vocab_size=1000,
        d_model=256,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
    )
    assert isinstance(bd, ParamBreakdown)


def test_theoretical_embedding_correct():
    bd = theoretical_breakdown(
        vocab_size=1000, d_model=256, n_layers=1, n_heads=4, n_kv_heads=4
    )
    assert bd.embedding == 1000 * 256


def test_theoretical_total_positive():
    bd = theoretical_breakdown(
        vocab_size=512, d_model=128, n_layers=2, n_heads=4, n_kv_heads=4
    )
    assert bd.total > 0


def test_attention_total_is_sum(tiny_counter):
    result = tiny_counter.count()
    bd = result.theoretical
    assert bd.attention_total == bd.attention_q + bd.attention_k + bd.attention_v + bd.attention_o


def test_ffn_total_is_sum(tiny_counter):
    result = tiny_counter.count()
    bd = result.theoretical
    assert bd.ffn_total == bd.ffn_gate + bd.ffn_up + bd.ffn_down


def test_total_is_sum_of_parts(tiny_counter):
    result = tiny_counter.count()
    bd = result.theoretical
    assert bd.total == bd.embedding + bd.attention_total + bd.ffn_total + bd.norm + bd.other


def test_gqa_kv_heads_smaller_than_q():
    """GQA: K/V params < Q params when n_kv_heads < n_heads."""
    bd = theoretical_breakdown(
        vocab_size=100, d_model=256, n_layers=1, n_heads=8, n_kv_heads=2
    )
    assert bd.attention_k < bd.attention_q
    assert bd.attention_v < bd.attention_q


def test_more_layers_more_params():
    bd2 = theoretical_breakdown(
        vocab_size=1000, d_model=256, n_layers=2, n_heads=4, n_kv_heads=4
    )
    bd4 = theoretical_breakdown(
        vocab_size=1000, d_model=256, n_layers=4, n_heads=4, n_kv_heads=4
    )
    assert bd4.total > bd2.total


# ---------------------------------------------------------------------------
# ModelParamCounter.count()
# ---------------------------------------------------------------------------


def test_count_returns_result(tiny_counter):
    result = tiny_counter.count()
    assert isinstance(result, ParamCountResult)


def test_count_no_model_actual_is_none(tiny_counter):
    result = tiny_counter.count(model=None)
    assert result.actual is None
    assert result.mismatch_pct is None


def test_theoretical_total_method(tiny_counter):
    t = tiny_counter.theoretical_total()
    assert t == tiny_counter.count().theoretical.total
    assert t > 0


# ---------------------------------------------------------------------------
# ParamBreakdown helpers
# ---------------------------------------------------------------------------


def test_breakdown_total_millions():
    bd = ParamBreakdown(embedding=50_000_000)
    assert bd.total_millions == pytest.approx(50.0)


def test_breakdown_as_dict_keys():
    bd = ParamBreakdown(embedding=1_000_000)
    d = bd.as_dict()
    assert "embedding_M" in d
    assert "total_M" in d
    assert "attention_total_M" in d


# ---------------------------------------------------------------------------
# ParamCountResult summary
# ---------------------------------------------------------------------------


def test_summary_theoretical_key(tiny_counter):
    result = tiny_counter.count()
    s = result.summary()
    assert "theoretical_total_M" in s


# ---------------------------------------------------------------------------
# ComponentCount
# ---------------------------------------------------------------------------


def test_component_millions():
    c = ComponentCount(name="tok_emb.weight", n_params=5_000_000)
    assert c.millions == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# count_from_model — mock-based
# ---------------------------------------------------------------------------


def test_count_from_model_no_torch_returns_none():
    """If torch is not importable, return None gracefully."""
    import sys
    original = sys.modules.get("torch")
    # Temporarily hide torch
    sys.modules["torch"] = None  # type: ignore[assignment]
    sys.modules["torch.nn"] = None  # type: ignore[assignment]
    try:
        result = count_from_model(object())
        assert result is None
    finally:
        if original is not None:
            sys.modules["torch"] = original
        else:
            del sys.modules["torch"]
        if "torch.nn" in sys.modules:
            del sys.modules["torch.nn"]


def test_count_from_model_with_mock():
    """Build a minimal mock nn.Module and verify counts are parsed."""
    import torch.nn as nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(100, 32)
            self.q_proj = nn.Linear(32, 32, bias=False)
            self.norm = nn.LayerNorm(32)

    model = TinyModel()
    bd = count_from_model(model)
    assert bd is not None
    assert bd.embedding > 0
    assert bd.attention_q > 0
    assert bd.norm > 0
