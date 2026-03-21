"""Tests for features/checkpoint_converter.py.

Uses temporary directories with fake tensors — no GPU required.
Skips tests that require safetensors if the package is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cola_coder.features.checkpoint_converter import (
    FEATURE_ENABLED,
    CheckpointConverter,
    ConversionResult,
    SUPPORTED_FORMATS,
    _build_hf_config,
    _detect_format,
    _flat_config,
    _strip_compile_prefix,
    is_enabled,
)

# Try to import safetensors — tests that need it are skipped if not available
try:
    import safetensors  # noqa: F401

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    import torch as _torch_check  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


def test_feature_enabled():
    assert FEATURE_ENABLED is True


def test_is_enabled():
    assert is_enabled() is True


# ---------------------------------------------------------------------------
# Supported formats
# ---------------------------------------------------------------------------


def test_supported_formats_keys():
    assert "safetensors" in SUPPORTED_FORMATS
    assert "pytorch" in SUPPORTED_FORMATS
    assert "huggingface" in SUPPORTED_FORMATS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_strip_compile_prefix_removes():
    state = {"_orig_mod.weight": "a", "bias": "b"}
    stripped, found = _strip_compile_prefix(state)
    assert "weight" in stripped
    assert "_orig_mod.weight" not in stripped
    assert found is True


def test_strip_compile_prefix_no_prefix():
    state = {"weight": "a", "bias": "b"}
    stripped, found = _strip_compile_prefix(state)
    assert stripped == state
    assert found is False


def test_detect_format_safetensors_file(tmp_path):
    f = tmp_path / "model.safetensors"
    f.write_text("dummy")
    assert _detect_format(f) == "safetensors"


def test_detect_format_pt_file(tmp_path):
    f = tmp_path / "model.pt"
    f.write_text("dummy")
    assert _detect_format(f) == "pytorch"


def test_detect_format_hf_dir(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_text("dummy")
    assert _detect_format(tmp_path) == "huggingface"


def test_detect_format_unknown(tmp_path):
    assert _detect_format(tmp_path / "nonexistent.xyz") == "unknown"


def test_flat_config_simple():
    d = {"a": 1, "b": {"c": 2, "d": 3}}
    flat = _flat_config(d)
    assert flat["a"] == 1
    assert flat["b.c"] == 2
    assert flat["b.d"] == 3


def test_build_hf_config_fields():
    meta = {
        "config": {
            "model": {
                "d_model": 512,
                "n_layers": 8,
                "n_heads": 8,
                "vocab_size": 16_000,
                "max_seq_len": 1024,
            }
        }
    }
    cfg = _build_hf_config(meta)
    assert cfg["hidden_size"] == 512
    assert cfg["num_hidden_layers"] == 8
    assert cfg["vocab_size"] == 16_000
    assert cfg["model_type"] == "cola_coder"


def test_build_hf_config_empty():
    cfg = _build_hf_config({})
    assert "model_type" in cfg
    assert cfg["model_type"] == "cola_coder"


# ---------------------------------------------------------------------------
# CheckpointConverter (needs torch + safetensors for full tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def converter():
    return CheckpointConverter()


def test_unsupported_format_raises(converter, tmp_path):
    dummy_ckpt = tmp_path / "ckpt"
    dummy_ckpt.mkdir()
    with pytest.raises(ValueError, match="Unsupported target format"):
        converter.convert(dummy_ckpt, "onnx")


def test_missing_source_raises(converter, tmp_path):
    with pytest.raises(FileNotFoundError):
        converter.convert(tmp_path / "nonexistent", "pytorch")


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_pytorch_roundtrip(converter, tmp_path):
    """Save a .pt file and convert it to safetensors if available, else pytorch."""
    import torch

    state = {"weight": torch.zeros(4, 4), "bias": torch.zeros(4)}
    pt_file = tmp_path / "model.pt"
    torch.save(state, str(pt_file))

    # Convert to pytorch (no-op but exercises the code path)
    result = converter.convert(pt_file, "pytorch", output=tmp_path / "out.pt")
    assert isinstance(result, ConversionResult)
    assert result.tensor_count == 2
    assert Path(result.target_path).exists()


@pytest.mark.skipif(not (HAS_TORCH and HAS_SAFETENSORS), reason="torch/safetensors not available")
def test_safetensors_to_pytorch(converter, tmp_path):
    import torch
    from safetensors.torch import save_file

    tensors = {"weight": torch.zeros(4, 4), "bias": torch.zeros(4)}
    st_path = tmp_path / "model.safetensors"
    save_file(tensors, str(st_path))

    result = converter.convert(st_path, "pytorch", output=tmp_path / "model.pt")
    assert result.success
    assert result.tensor_count == 2
    assert Path(result.target_path).exists()


@pytest.mark.skipif(not (HAS_TORCH and HAS_SAFETENSORS), reason="torch/safetensors not available")
def test_to_huggingface_creates_config_json(converter, tmp_path):
    import torch
    from safetensors.torch import save_file

    # Create a minimal cola-coder checkpoint directory
    ckpt_dir = tmp_path / "step_00010000"
    ckpt_dir.mkdir()
    # Use non-shared tensors (cola-coder saves without output.weight due to weight tying)
    tensors = {"tok_emb.weight": torch.zeros(100, 64)}
    save_file(tensors, str(ckpt_dir / "model.safetensors"))
    (ckpt_dir / "metadata.json").write_text(
        json.dumps({"step": 10000, "config": {"model": {"d_model": 64, "n_layers": 2}}})
    )

    hf_dir = tmp_path / "hf_output"
    result = converter.convert(ckpt_dir, "huggingface", output=hf_dir)
    assert result.success
    assert (hf_dir / "config.json").exists()
    assert (hf_dir / "model.safetensors").exists()
    cfg = json.loads((hf_dir / "config.json").read_text())
    assert cfg["model_type"] == "cola_coder"
