"""Tests for checkpoint_merger.py (feature 49)."""

from __future__ import annotations


import pytest

from cola_coder.features.checkpoint_merger import (
    FEATURE_ENABLED,
    CheckpointMerger,
    MergeConfig,
    MergeMethod,
    MergeReport,
    _dot,
    _norm,
    _slerp_param,
    _weighted_mean,
    is_enabled,
)


def test_feature_enabled():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


# ---------------------------------------------------------------------------
# Arithmetic helpers
# ---------------------------------------------------------------------------


def test_weighted_mean_equal_weights():
    result = _weighted_mean([[1.0, 2.0], [3.0, 4.0]], [0.5, 0.5])
    assert abs(result[0] - 2.0) < 1e-9
    assert abs(result[1] - 3.0) < 1e-9


def test_norm_zero_vector():
    assert _norm([0.0, 0.0]) == 0.0


def test_norm_unit_vector():
    v = [1.0, 0.0, 0.0]
    assert abs(_norm(v) - 1.0) < 1e-9


def test_dot_product():
    assert abs(_dot([1.0, 2.0], [3.0, 4.0]) - 11.0) < 1e-9


# ---------------------------------------------------------------------------
# SLERP
# ---------------------------------------------------------------------------


def test_slerp_t0_returns_a():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    result = _slerp_param(a, b, t=0.0)
    assert abs(result[0] - 1.0) < 1e-6
    assert abs(result[1] - 0.0) < 1e-6


def test_slerp_t1_returns_b():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    result = _slerp_param(a, b, t=1.0)
    assert abs(result[0] - 0.0) < 1e-6
    assert abs(result[1] - 1.0) < 1e-6


def test_slerp_midpoint_norm():
    """SLERP at t=0.5 should have norm between the two input norms."""
    a = [2.0, 0.0]
    b = [0.0, 4.0]
    result = _slerp_param(a, b, t=0.5)
    result_norm = _norm(result)
    assert abs(result_norm - 3.0) < 0.1  # midpoint between 2 and 4


# ---------------------------------------------------------------------------
# Merger
# ---------------------------------------------------------------------------


def _make_ckpt(values: dict) -> dict:
    return {k: v for k, v in values.items()}


def test_linear_merge_equal_weights():
    ckpt_a = {"w": [1.0, 2.0, 3.0]}
    ckpt_b = {"w": [3.0, 4.0, 5.0]}
    merger = CheckpointMerger()
    merged, report = merger.merge([ckpt_a, ckpt_b])
    assert abs(merged["w"][0] - 2.0) < 1e-9
    assert abs(merged["w"][1] - 3.0) < 1e-9
    assert report.method == "linear"
    assert report.num_checkpoints == 2


def test_linear_merge_custom_weights():
    ckpt_a = {"w": [0.0, 0.0]}
    ckpt_b = {"w": [4.0, 8.0]}
    merger = CheckpointMerger()
    config = MergeConfig(method=MergeMethod.LINEAR, weights=[1.0, 3.0])
    merged, _ = merger.merge([ckpt_a, ckpt_b], config)
    # weight_a=0.25, weight_b=0.75 → result = [3.0, 6.0]
    assert abs(merged["w"][0] - 3.0) < 1e-9
    assert abs(merged["w"][1] - 6.0) < 1e-9


def test_slerp_two_checkpoints():
    ckpt_a = {"v": [1.0, 0.0]}
    ckpt_b = {"v": [0.0, 1.0]}
    config = MergeConfig(method=MergeMethod.SLERP, slerp_t=0.5)
    merger = CheckpointMerger()
    merged, report = merger.merge([ckpt_a, ckpt_b], config)
    assert report.method == "slerp"
    # Result should be roughly [sqrt(0.5), sqrt(0.5)] with magnitude 1
    result = merged["v"]
    result_norm = _norm(result)
    assert abs(result_norm - 1.0) < 0.1


def test_task_arithmetic_adds_to_base():
    base = {"w": [0.0, 0.0]}
    ckpt = {"w": [2.0, 4.0]}
    config = MergeConfig(
        method=MergeMethod.TASK_ARITHMETIC,
        base_checkpoint=base,
        task_arithmetic_scale=1.0,
    )
    merger = CheckpointMerger()
    merged, report = merger.merge([ckpt], config)
    # task_vector = ckpt - base = [2, 4], scale=1 → merged = base + [2, 4] = [2, 4]
    assert abs(merged["w"][0] - 2.0) < 1e-9
    assert abs(merged["w"][1] - 4.0) < 1e-9
    assert report.method == "task_arithmetic"


def test_task_arithmetic_scale_half():
    base = {"w": [0.0, 0.0]}
    ckpt = {"w": [2.0, 4.0]}
    config = MergeConfig(
        method=MergeMethod.TASK_ARITHMETIC,
        base_checkpoint=base,
        task_arithmetic_scale=0.5,
    )
    merger = CheckpointMerger()
    merged, _ = merger.merge([ckpt], config)
    assert abs(merged["w"][0] - 1.0) < 1e-9


def test_mismatched_keys_raises():
    ckpt_a = {"w1": [1.0]}
    ckpt_b = {"w2": [2.0]}
    merger = CheckpointMerger()
    with pytest.raises(ValueError, match="mismatch"):
        merger.merge([ckpt_a, ckpt_b])


def test_empty_checkpoints_raises():
    merger = CheckpointMerger()
    with pytest.raises(ValueError):
        merger.merge([])


def test_wrong_weights_length_raises():
    ckpts = [{"w": [1.0]}, {"w": [2.0]}]
    config = MergeConfig(weights=[1.0])  # only 1 weight for 2 checkpoints
    merger = CheckpointMerger()
    with pytest.raises(ValueError, match="weights length"):
        merger.merge(ckpts, config)


def test_task_arithmetic_no_base_raises():
    with pytest.raises(ValueError, match="base_checkpoint"):
        MergeConfig(method=MergeMethod.TASK_ARITHMETIC)


def test_report_summary_format():
    report = MergeReport(
        method="linear",
        num_checkpoints=3,
        weights=[1.0, 1.0, 1.0],
        param_names=["w1", "w2"],
        num_params=200,
    )
    s = report.summary()
    assert "method=linear" in s
    assert "checkpoints=3" in s
