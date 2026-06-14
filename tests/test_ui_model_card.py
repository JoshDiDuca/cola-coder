"""Tests for cola_coder.ui.model_card.

Hermetic: synthesizes a tiny valid .safetensors (8-byte little-endian header
length + JSON header + zero-filled data region) plus a metadata.json with the
canonical nested ``config.model`` / ``config.training`` shape in ``tmp_path``.
No torch/safetensors imports, no real checkpoints touched.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from cola_coder.ui.model_card import build_model_card


def _write_safetensors(path: Path, tensors: dict, metadata: dict | None = None) -> None:
    """Synthesize a minimal valid .safetensors file (header only matters)."""
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


def _canonical_metadata() -> dict:
    """The nested shape written by training/checkpoint.save_checkpoint."""
    return {
        "step": 100000,
        "loss": 0.9859,
        "config": {
            "model": {
                "vocab_size": 32768,
                "dim": 768,
                "n_layers": 12,
                "n_heads": 12,
                "n_kv_heads": 4,
                "max_seq_len": 2048,
                "rope_theta": 10000.0,
            },
            "training": {
                "batch_size": 12,
                "learning_rate": 0.0006,
                "max_steps": 100000,
                "precision": "bf16",
            },
        },
        "data_path": "data/train.npy",
        "tokenizer_path": "data/tokenizer.json",
    }


def _make_checkpoint(tmp_path: Path, **kwargs) -> Path:
    """Build a checkpoint dir with model.safetensors (2 tensors) + optional sidecars."""
    ckpt = tmp_path / "step_00100000"
    ckpt.mkdir(parents=True)
    _write_safetensors(
        ckpt / "model.safetensors",
        {
            "tok_emb.weight": ("F32", [100, 8]),  # 800 params
            "blocks.0.attn.wq.weight": ("F32", [8, 8]),  # 64 params
        },
    )
    if "metadata" in kwargs and kwargs["metadata"] is not None:
        (ckpt / "metadata.json").write_text(json.dumps(kwargs["metadata"]))
    if kwargs.get("moe_config") is not None:
        (ckpt / "moe_config.json").write_text(json.dumps(kwargs["moe_config"]))
    return ckpt


def test_basic_param_count(tmp_path):
    ckpt = _make_checkpoint(tmp_path, metadata=_canonical_metadata())
    result = build_model_card(str(ckpt))
    assert "error" not in result
    assert result["num_params"] == 864  # 800 + 64
    assert result["name"] == "step_00100000"
    assert result["path"] == str(ckpt)


def test_architecture_split(tmp_path):
    ckpt = _make_checkpoint(tmp_path, metadata=_canonical_metadata())
    arch = build_model_card(str(ckpt))["architecture"]
    assert arch["dim"] == 768
    assert arch["n_layers"] == 12
    assert arch["n_heads"] == 12
    assert arch["n_kv_heads"] == 4
    assert arch["vocab_size"] == 32768
    assert arch["max_seq_len"] == 2048
    # Training-only keys must not leak into architecture.
    assert "learning_rate" not in arch
    assert "loss" not in arch


def test_training_split(tmp_path):
    ckpt = _make_checkpoint(tmp_path, metadata=_canonical_metadata())
    training = build_model_card(str(ckpt))["training"]
    # Top-level step/loss are training provenance.
    assert training["step"] == 100000
    assert training["loss"] == 0.9859
    # config.training fields.
    assert training["learning_rate"] == 0.0006
    assert training["batch_size"] == 12
    assert training["precision"] == "bf16"
    # Architecture keys must not leak into training.
    assert "dim" not in training
    assert "vocab_size" not in training


def test_markdown_contains_key_facts(tmp_path):
    ckpt = _make_checkpoint(tmp_path, metadata=_canonical_metadata())
    md = build_model_card(str(ckpt))["markdown"]
    assert "# step_00100000" in md
    assert "Architecture" in md
    assert "Training" in md
    assert "768" in md  # dim
    assert "0.0006" in md  # learning rate
    assert "864" in md  # exact param count appears


def test_markdown_humanized_params(tmp_path):
    """A large fake checkpoint should humanize params to M."""
    ckpt = tmp_path / "big"
    ckpt.mkdir()
    # 2_000_000 params: 1000x1000 + 1000x1000.
    _write_safetensors(
        ckpt / "model.safetensors",
        {"a.weight": ("F32", [1000, 1000]), "b.weight": ("F32", [1000, 1000])},
    )
    (ckpt / "metadata.json").write_text(json.dumps(_canonical_metadata()))
    result = build_model_card(str(ckpt))
    assert result["num_params"] == 2_000_000
    assert "2.0M" in result["markdown"]


def test_is_moe_dense_default(tmp_path):
    ckpt = _make_checkpoint(tmp_path, metadata=_canonical_metadata())
    result = build_model_card(str(ckpt))
    assert result["is_moe"] is False
    assert "Dense" in result["markdown"]


def test_is_moe_via_sidecar(tmp_path):
    ckpt = _make_checkpoint(
        tmp_path,
        metadata=_canonical_metadata(),
        moe_config={"num_experts": 8, "moe_layers": "all"},
    )
    result = build_model_card(str(ckpt))
    assert result["is_moe"] is True
    assert "MoE" in result["markdown"]


def test_graceful_when_metadata_absent(tmp_path):
    """No metadata.json: still succeed with empty arch/training dicts."""
    ckpt = _make_checkpoint(tmp_path)  # no metadata
    result = build_model_card(str(ckpt))
    assert "error" not in result
    assert result["num_params"] == 864
    assert result["architecture"] == {}
    assert result["training"] == {}
    assert "step_00100000" in result["markdown"]
    assert "No data available" in result["markdown"]


def test_flat_metadata_shape(tmp_path):
    """A flat metadata.json (no nested config) is still split correctly."""
    flat = {"dim": 512, "n_layers": 8, "step": 5000, "learning_rate": 1e-4}
    ckpt = _make_checkpoint(tmp_path, metadata=flat)
    result = build_model_card(str(ckpt))
    assert result["architecture"]["dim"] == 512
    assert result["architecture"]["n_layers"] == 8
    assert result["training"]["step"] == 5000
    assert result["training"]["learning_rate"] == 1e-4


def test_tokenizer_field_present_or_none(tmp_path):
    """tokenizer is either a dict result or None — never an error dict."""
    ckpt = _make_checkpoint(tmp_path, metadata=_canonical_metadata())
    result = build_model_card(str(ckpt))
    tok = result["tokenizer"]
    assert tok is None or (isinstance(tok, dict) and "error" not in tok)


def test_missing_dir_returns_error(tmp_path):
    result = build_model_card(str(tmp_path / "does_not_exist"))
    assert "error" in result
    assert "num_params" not in result


def test_dir_without_safetensors_returns_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "metadata.json").write_text("{}")
    result = build_model_card(str(empty))
    assert "error" in result


def test_direct_safetensors_path(tmp_path):
    ckpt = _make_checkpoint(tmp_path, metadata=_canonical_metadata())
    st = ckpt / "model.safetensors"
    result = build_model_card(str(st))
    assert "error" not in result
    assert result["num_params"] == 864
    # name resolves to the containing dir.
    assert result["name"] == "step_00100000"
