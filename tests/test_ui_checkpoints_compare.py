"""Tests for cola_coder.ui.checkpoints_compare.

Hermetic: every fixture synthesizes a tiny safetensors file by hand (8-byte
little-endian header length + JSON header + a fake data region) plus sidecars in
``tmp_path``. Mirrors the helper pattern in test_ui_checkpoint_detail.py. No
torch/safetensors imports, no real checkpoints touched.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from cola_coder.ui.checkpoints_compare import compare_checkpoints


def _write_safetensors(path: Path, tensors: dict, metadata: dict | None = None) -> None:
    """Synthesize a minimal valid .safetensors file.

    ``tensors`` maps name -> (dtype, shape). The data region is zero-filled — only
    the header is ever read by the code under test.
    """
    header: dict = {}
    offset = 0
    width = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8}
    for name, (dtype, shape) in tensors.items():
        n = 1
        for d in shape:
            n *= d
        nbytes = n * width.get(dtype, 4)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    if metadata is not None:
        header["__metadata__"] = metadata

    header_json = json.dumps(header).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header_json)))
        handle.write(header_json)
        handle.write(b"\x00" * offset)


def _make_checkpoint(
    base: Path,
    tensors: dict,
    metadata: dict | None = None,
    moe_config: dict | None = None,
) -> Path:
    """Build a checkpoint dir with model.safetensors + optional sidecars."""
    base.mkdir(parents=True)
    _write_safetensors(base / "model.safetensors", tensors)
    if metadata is not None:
        (base / "metadata.json").write_text(json.dumps(metadata))
    if moe_config is not None:
        (base / "moe_config.json").write_text(json.dumps(moe_config))
    return base


# Standard small tensor sets reused across tests.
_SMALL = {"tok_emb.weight": ("F32", [100, 8]), "blocks.0.attn.wq.weight": ("F32", [8, 8])}  # 864
_BIG = {
    "tok_emb.weight": ("F32", [200, 8]),  # 1600
    "blocks.0.attn.wq.weight": ("F32", [8, 8]),  # 64
    "blocks.0.attn.wk.weight": ("F32", [8, 8]),  # 64
}  # 1728, 3 tensors


def test_num_params_delta_positive(tmp_path):
    a = _make_checkpoint(tmp_path / "a", _SMALL)
    b = _make_checkpoint(tmp_path / "b", _BIG)
    result = compare_checkpoints(str(a), str(b))
    assert "error" not in result
    assert result["diff"]["num_params_delta"] == 1728 - 864  # b - a, positive


def test_num_params_delta_negative(tmp_path):
    a = _make_checkpoint(tmp_path / "a", _BIG)
    b = _make_checkpoint(tmp_path / "b", _SMALL)
    result = compare_checkpoints(str(a), str(b))
    assert result["diff"]["num_params_delta"] == 864 - 1728  # negative


def test_num_params_delta_zero(tmp_path):
    a = _make_checkpoint(tmp_path / "a", _SMALL)
    b = _make_checkpoint(tmp_path / "b", _SMALL)
    result = compare_checkpoints(str(a), str(b))
    assert result["diff"]["num_params_delta"] == 0


def test_tensor_count_delta(tmp_path):
    a = _make_checkpoint(tmp_path / "a", _SMALL)  # 2 tensors
    b = _make_checkpoint(tmp_path / "b", _BIG)  # 3 tensors
    result = compare_checkpoints(str(a), str(b))
    assert result["diff"]["tensor_count_delta"] == 1  # 3 - 2


def test_sides_carry_full_detail(tmp_path):
    a = _make_checkpoint(tmp_path / "a", _SMALL, metadata={"loss": 1.5})
    b = _make_checkpoint(tmp_path / "b", _SMALL, metadata={"loss": 1.2})
    result = compare_checkpoints(str(a), str(b))
    assert result["a"]["num_params"] == 864
    assert result["b"]["num_params"] == 864
    assert result["a"]["metadata"] == {"loss": 1.5}
    assert result["b"]["metadata"] == {"loss": 1.2}


def test_metadata_changed_keys_differing_values(tmp_path):
    a = _make_checkpoint(tmp_path / "a", _SMALL, metadata={"loss": 1.5, "dim": 8})
    b = _make_checkpoint(tmp_path / "b", _SMALL, metadata={"loss": 1.2, "dim": 8})
    result = compare_checkpoints(str(a), str(b))
    assert result["diff"]["metadata_changed_keys"] == ["loss"]  # dim unchanged


def test_metadata_changed_keys_union_and_sorted(tmp_path):
    a = _make_checkpoint(tmp_path / "a", _SMALL, metadata={"loss": 1.5, "only_a": 1})
    b = _make_checkpoint(tmp_path / "b", _SMALL, metadata={"loss": 1.5, "only_b": 2})
    result = compare_checkpoints(str(a), str(b))
    # only_a and only_b appear in one side each; loss is identical -> excluded.
    assert result["diff"]["metadata_changed_keys"] == ["only_a", "only_b"]


def test_metadata_changed_keys_none_treated_as_empty(tmp_path):
    a = _make_checkpoint(tmp_path / "a", _SMALL)  # no metadata.json -> None
    b = _make_checkpoint(tmp_path / "b", _SMALL, metadata={"loss": 1.0})
    result = compare_checkpoints(str(a), str(b))
    assert result["a"]["metadata"] is None
    assert result["diff"]["metadata_changed_keys"] == ["loss"]


def test_metadata_no_changes(tmp_path):
    a = _make_checkpoint(tmp_path / "a", _SMALL, metadata={"loss": 1.0, "dim": 8})
    b = _make_checkpoint(tmp_path / "b", _SMALL, metadata={"loss": 1.0, "dim": 8})
    result = compare_checkpoints(str(a), str(b))
    assert result["diff"]["metadata_changed_keys"] == []


def test_is_moe_changed_true(tmp_path):
    a = _make_checkpoint(tmp_path / "a", _SMALL)  # dense
    b = _make_checkpoint(tmp_path / "b", _SMALL, moe_config={"num_experts": 8})  # MoE
    result = compare_checkpoints(str(a), str(b))
    assert result["a"]["is_moe"] is False
    assert result["b"]["is_moe"] is True
    assert result["diff"]["is_moe_changed"] is True


def test_is_moe_changed_false_when_both_dense(tmp_path):
    a = _make_checkpoint(tmp_path / "a", _SMALL)
    b = _make_checkpoint(tmp_path / "b", _BIG)
    result = compare_checkpoints(str(a), str(b))
    assert result["diff"]["is_moe_changed"] is False


def test_dtype_diffs(tmp_path):
    a = _make_checkpoint(tmp_path / "a", {"x.weight": ("F32", [4, 4])})  # F32 only
    b = _make_checkpoint(tmp_path / "b", {"x.weight": ("BF16", [4, 4])})  # BF16 only
    result = compare_checkpoints(str(a), str(b))
    assert result["diff"]["dtypes_only_a"] == ["F32"]
    assert result["diff"]["dtypes_only_b"] == ["BF16"]


def test_dtype_diffs_empty_when_identical(tmp_path):
    a = _make_checkpoint(tmp_path / "a", _SMALL)
    b = _make_checkpoint(tmp_path / "b", _SMALL)
    result = compare_checkpoints(str(a), str(b))
    assert result["diff"]["dtypes_only_a"] == []
    assert result["diff"]["dtypes_only_b"] == []


def test_error_passthrough_a_missing(tmp_path):
    b = _make_checkpoint(tmp_path / "b", _SMALL)
    result = compare_checkpoints(str(tmp_path / "missing_a"), str(b))
    assert "error" in result
    assert "a:" in result["error"]
    assert "error" in result["a"]
    assert "error" not in result["b"]
    assert "diff" not in result


def test_error_passthrough_b_missing(tmp_path):
    a = _make_checkpoint(tmp_path / "a", _SMALL)
    result = compare_checkpoints(str(a), str(tmp_path / "missing_b"))
    assert "error" in result
    assert "b:" in result["error"]
    assert "error" not in result["a"]
    assert "error" in result["b"]


def test_error_passthrough_both_missing(tmp_path):
    result = compare_checkpoints(str(tmp_path / "no_a"), str(tmp_path / "no_b"))
    assert "error" in result
    assert "a:" in result["error"] and "b:" in result["error"]
    assert "error" in result["a"]
    assert "error" in result["b"]
