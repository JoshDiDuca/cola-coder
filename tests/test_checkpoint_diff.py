"""Tests for scripts/checkpoint_diff.py.

No GPU, no model weights — uses temporary directories with metadata.json.
"""

from __future__ import annotations

import json
from pathlib import Path


# Import the module under test from scripts/
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from checkpoint_diff import (  # noqa: E402
    _flat_config,
    _fmt_num,
    _read_metadata,
    build_diff_report,
    diff_configs,
    diff_params,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ckpt(tmp_path: Path, name: str, meta: dict) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


def test_flat_config_simple():
    d = {"a": 1, "b": {"c": 2}}
    result = _flat_config(d)
    assert result["a"] == 1
    assert result["b.c"] == 2


def test_flat_config_deeply_nested():
    d = {"x": {"y": {"z": 42}}}
    result = _flat_config(d)
    assert result["x.y.z"] == 42


def test_fmt_num_billions():
    assert "B" in _fmt_num(2_000_000_000)


def test_fmt_num_millions():
    assert "M" in _fmt_num(125_000_000)


def test_fmt_num_thousands():
    assert "K" in _fmt_num(50_000)


def test_fmt_num_small():
    assert _fmt_num(10) == "10"


def test_read_metadata_existing(tmp_path):
    meta = {"step": 100, "loss": 2.5}
    d = tmp_path / "step_00000100"
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps(meta))
    result = _read_metadata(d)
    assert result["step"] == 100


def test_read_metadata_missing(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    result = _read_metadata(d)
    assert result == {}


# ---------------------------------------------------------------------------
# diff_configs
# ---------------------------------------------------------------------------


def test_diff_configs_no_diff():
    meta = {"config": {"model": {"d_model": 256}}}
    diffs = diff_configs(meta, meta)
    assert diffs == []


def test_diff_configs_detects_change():
    a = {"config": {"model": {"d_model": 256}}}
    b = {"config": {"model": {"d_model": 512}}}
    diffs = diff_configs(a, b)
    assert len(diffs) == 1
    key, va, vb = diffs[0]
    assert "d_model" in key
    assert va == 256
    assert vb == 512


def test_diff_configs_detects_missing():
    a = {"config": {"model": {"d_model": 256}}}
    b = {"config": {"model": {"d_model": 256, "new_key": True}}}
    diffs = diff_configs(a, b)
    assert any("new_key" in k for k, _, _ in diffs)


# ---------------------------------------------------------------------------
# diff_params
# ---------------------------------------------------------------------------


def test_diff_params_identical():
    params = {"weight": ((4, 4), "torch.float32")}
    result = diff_params(params, params)
    assert result["added"] == []
    assert result["removed"] == []
    assert result["shape_changed"] == []


def test_diff_params_added():
    a = {"weight": ((4, 4), "torch.float32")}
    b = {"weight": ((4, 4), "torch.float32"), "bias": ((4,), "torch.float32")}
    result = diff_params(a, b)
    assert "bias" in result["added"]


def test_diff_params_removed():
    a = {"weight": ((4, 4), "torch.float32"), "bias": ((4,), "torch.float32")}
    b = {"weight": ((4, 4), "torch.float32")}
    result = diff_params(a, b)
    assert "bias" in result["removed"]


def test_diff_params_shape_changed():
    a = {"weight": ((4, 4), "torch.float32")}
    b = {"weight": ((8, 8), "torch.float32")}
    result = diff_params(a, b)
    assert len(result["shape_changed"]) == 1
    name, sha, shb = result["shape_changed"][0]
    assert sha == (4, 4)
    assert shb == (8, 8)


def test_diff_params_total_counts():
    a = {"w": ((10, 10), "f")}  # 100 params
    b = {"w": ((10, 10), "f"), "b": ((5,), "f")}  # 105 params
    result = diff_params(a, b)
    assert result["total_params_a"] == 100
    assert result["total_params_b"] == 105


# ---------------------------------------------------------------------------
# build_diff_report integration
# ---------------------------------------------------------------------------


def test_build_diff_report_step_progression(tmp_path):
    meta_a = {
        "step": 1000, "loss": 3.0,
        "config": {"model": {"d_model": 256, "n_layers": 4}},
    }
    meta_b = {
        "step": 2000, "loss": 2.5,
        "config": {"model": {"d_model": 256, "n_layers": 4}},
    }
    ckpt_a = _make_ckpt(tmp_path, "step_00001000", meta_a)
    ckpt_b = _make_ckpt(tmp_path, "step_00002000", meta_b)

    report = build_diff_report(ckpt_a, ckpt_b)
    assert report["step_a"] == 1000
    assert report["step_b"] == 2000
    assert report["loss_a"] == 3.0
    assert report["loss_b"] == 2.5


def test_build_diff_report_loss_delta(tmp_path):
    meta_a = {"step": 1000, "loss": 3.0, "config": {}}
    meta_b = {"step": 2000, "loss": 2.0, "config": {}}
    ckpt_a = _make_ckpt(tmp_path, "step_a", meta_a)
    ckpt_b = _make_ckpt(tmp_path, "step_b", meta_b)

    report = build_diff_report(ckpt_a, ckpt_b)
    # loss_delta = loss_a - loss_b = 3.0 - 2.0 = 1.0 (positive = improved)
    assert abs(report["loss_delta"] - 1.0) < 1e-9


def test_build_diff_report_perplexity(tmp_path):
    import math

    meta_a = {"step": 1, "loss": 2.0, "config": {}}
    meta_b = {"step": 2, "loss": 1.0, "config": {}}
    ckpt_a = _make_ckpt(tmp_path, "sa", meta_a)
    ckpt_b = _make_ckpt(tmp_path, "sb", meta_b)

    report = build_diff_report(ckpt_a, ckpt_b)
    assert abs(report["ppl_a"] - math.exp(2.0)) < 1e-6
    assert abs(report["ppl_b"] - math.exp(1.0)) < 1e-6
