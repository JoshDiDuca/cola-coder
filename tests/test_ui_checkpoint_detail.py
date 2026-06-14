"""Tests for cola_coder.ui.checkpoint_detail.

Hermetic: every fixture synthesizes a tiny safetensors file by hand (8-byte
little-endian header length + JSON header + a fake data region) plus sidecars in
``tmp_path``. No torch/safetensors imports, no real checkpoints touched.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from cola_coder.ui.checkpoint_detail import checkpoint_detail


def _write_safetensors(path: Path, tensors: dict, metadata: dict | None = None) -> None:
    """Synthesize a minimal valid .safetensors file.

    ``tensors`` maps name -> (dtype, shape). Data region is zero-filled but its
    actual contents are irrelevant — checkpoint_detail only reads the header.
    """
    header: dict = {}
    offset = 0
    # Bytes-per-element for the dtypes we use in tests.
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


def _make_checkpoint(tmp_path: Path, **kwargs) -> Path:
    """Build a checkpoint dir with model.safetensors (2 tensors) + optional sidecars."""
    ckpt = tmp_path / "step_00001000"
    ckpt.mkdir(parents=True)
    _write_safetensors(
        ckpt / "model.safetensors",
        {
            "tok_emb.weight": ("F32", [100, 8]),   # 800 params
            "blocks.0.attn.wq.weight": ("F32", [8, 8]),  # 64 params
        },
        metadata=kwargs.get("st_metadata"),
    )
    if kwargs.get("metadata") is not None:
        (ckpt / "metadata.json").write_text(json.dumps(kwargs["metadata"]))
    if kwargs.get("moe_config") is not None:
        (ckpt / "moe_config.json").write_text(json.dumps(kwargs["moe_config"]))
    if kwargs.get("training_state"):
        (ckpt / "training_state.pt").write_bytes(b"\x80\x04fake-pickle")
    return ckpt


def test_basic_param_and_tensor_counts(tmp_path):
    ckpt = _make_checkpoint(tmp_path, metadata={"loss": 1.5, "dim": 8})
    result = checkpoint_detail(str(ckpt))
    assert "error" not in result
    assert result["num_params"] == 864  # 800 + 64
    assert result["tensor_count"] == 2
    assert result["dtypes"] == ["F32"]


def test_metadata_parsed(tmp_path):
    ckpt = _make_checkpoint(tmp_path, metadata={"loss": 1.5, "dim": 8})
    result = checkpoint_detail(str(ckpt))
    assert result["metadata"] == {"loss": 1.5, "dim": 8}


def test_metadata_none_when_absent(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    result = checkpoint_detail(str(ckpt))
    assert result["metadata"] is None


def test_files_listed(tmp_path):
    ckpt = _make_checkpoint(tmp_path, metadata={"loss": 1.0})
    result = checkpoint_detail(str(ckpt))
    assert "model.safetensors" in result["files"]
    assert "metadata.json" in result["files"]
    assert result["path"] == str(ckpt)


def test_distinct_dtypes(tmp_path):
    ckpt = tmp_path / "step_2"
    ckpt.mkdir()
    _write_safetensors(
        ckpt / "model.safetensors",
        {
            "a.weight": ("F32", [4, 4]),   # 16
            "b.weight": ("BF16", [2, 3]),  # 6
            "c.weight": ("F16", [5]),      # 5
        },
    )
    result = checkpoint_detail(str(ckpt))
    assert result["num_params"] == 27
    assert result["tensor_count"] == 3
    assert result["dtypes"] == ["BF16", "F16", "F32"]


def test_skips_metadata_key_in_header(tmp_path):
    """The __metadata__ entry must not be counted as a tensor."""
    ckpt = _make_checkpoint(tmp_path, st_metadata={"format": "pt"})
    result = checkpoint_detail(str(ckpt))
    assert result["tensor_count"] == 2
    assert result["num_params"] == 864


def test_moe_via_sidecar(tmp_path):
    ckpt = _make_checkpoint(
        tmp_path, moe_config={"num_experts": 8, "moe_layers": "all"}
    )
    result = checkpoint_detail(str(ckpt))
    assert result["is_moe"] is True
    assert result["moe_config"] == {"num_experts": 8, "moe_layers": "all"}


def test_moe_via_tensor_keys(tmp_path):
    ckpt = tmp_path / "moe_dir"
    ckpt.mkdir()
    _write_safetensors(
        ckpt / "model.safetensors",
        {
            "tok_emb.weight": ("F32", [10, 4]),
            "blocks.0.ffn.experts.0.w1.weight": ("F32", [4, 4]),
        },
    )
    result = checkpoint_detail(str(ckpt))
    assert result["is_moe"] is True
    assert result["moe_config"] is None


def test_not_moe_by_default(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    result = checkpoint_detail(str(ckpt))
    assert result["is_moe"] is False
    assert result["moe_config"] is None


def test_training_state_presence(tmp_path):
    with_state = _make_checkpoint(tmp_path / "a", training_state=True)
    without_state = _make_checkpoint(tmp_path / "b")
    assert checkpoint_detail(str(with_state))["has_training_state"] is True
    assert checkpoint_detail(str(without_state))["has_training_state"] is False


def test_direct_safetensors_path(tmp_path):
    ckpt = _make_checkpoint(tmp_path, metadata={"loss": 2.0})
    st = ckpt / "model.safetensors"
    result = checkpoint_detail(str(st))
    assert result["num_params"] == 864
    assert result["path"] == str(ckpt)
    assert result["metadata"] == {"loss": 2.0}


def test_multiple_shards_summed(tmp_path):
    ckpt = tmp_path / "sharded"
    ckpt.mkdir()
    _write_safetensors(ckpt / "model-00001.safetensors", {"a": ("F32", [10, 10])})
    _write_safetensors(ckpt / "model-00002.safetensors", {"b": ("F32", [5, 5])})
    result = checkpoint_detail(str(ckpt))
    assert result["num_params"] == 125  # 100 + 25
    assert result["tensor_count"] == 2


def test_missing_dir_returns_error(tmp_path):
    result = checkpoint_detail(str(tmp_path / "does_not_exist"))
    assert "error" in result
    assert "metadata" not in result


def test_dir_without_safetensors_returns_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "metadata.json").write_text("{}")
    result = checkpoint_detail(str(empty))
    assert "error" in result


def test_bad_header_returns_error(tmp_path):
    ckpt = tmp_path / "bad"
    ckpt.mkdir()
    # Truncated: claims a huge header but provides no bytes.
    (ckpt / "model.safetensors").write_bytes(struct.pack("<Q", 9999))
    result = checkpoint_detail(str(ckpt))
    assert "error" in result
