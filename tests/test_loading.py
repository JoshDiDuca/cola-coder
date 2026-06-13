"""Tests for inference model-loading helpers (inference/loading.py).

Focus: load_generator must resolve a `latest` pointer FILE to the real
checkpoint dir BEFORE inspecting it for vocab size and MoE config. Otherwise
reading `latest/model.safetensors` / `latest/moe_config.json` silently misses,
leaving the config dense and crashing load_state_dict on an upcycled MoE
checkpoint loaded via its `latest` pointer (the documented usage, e.g.
`--checkpoint checkpoints/tiny/latest`).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cola_coder.inference import loading


def _make_moe_checkpoint(base: Path) -> tuple[Path, Path]:
    """Create a step_* dir holding a MoE sidecar plus a `latest` pointer file.

    Returns (checkpoint_dir, latest_pointer).
    """
    ckpt_dir = base / "step_00001000"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "moe_config.json").write_text(
        json.dumps({"num_experts": 8, "num_shared_experts": 1, "top_k": 2}),
        encoding="utf-8",
    )
    # A minimal (empty) model.safetensors so detect_moe_checkpoint's sidecar
    # branch is reached; the sidecar is enough for the assertion below.
    (ckpt_dir / "model.safetensors").write_text("", encoding="utf-8")
    latest = base / "latest"
    latest.write_text(str(ckpt_dir), encoding="utf-8")
    return ckpt_dir, latest


def test_apply_moe_config_resolves_dir_but_not_latest_pointer(tmp_path):
    """Regression guard: detection works on the dir, not on the pointer file.

    This documents WHY load_generator must resolve `latest` first — the MoE
    detector reads files inside the path it is handed, so a pointer file yields
    no detection.
    """
    ckpt_dir, latest = _make_moe_checkpoint(tmp_path)

    cfg_dir = _DummyConfig()
    assert loading.apply_moe_config_from_checkpoint(cfg_dir, ckpt_dir) is True
    assert cfg_dir.model.moe.enabled is True

    cfg_ptr = _DummyConfig()
    # Handing the pointer FILE directly detects nothing (the bug surface).
    assert loading.apply_moe_config_from_checkpoint(cfg_ptr, latest) is False
    assert cfg_ptr.model.moe.enabled is False


def test_load_generator_resolves_latest_pointer_before_moe_inspection(
    tmp_path, monkeypatch
):
    """load_generator(latest_pointer) must flip the config to MoE.

    Before the fix, load_generator inspected the unresolved pointer file, so an
    upcycled MoE checkpoint loaded via `.../latest` stayed dense and crashed in
    load_model_only. After the fix the pointer is resolved first, so MoE config
    is applied and apply_moe_config sees the real step_* dir.
    """
    ckpt_dir, latest = _make_moe_checkpoint(tmp_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: {}\n", encoding="utf-8")
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()

    captured: dict = {}

    def fake_apply_moe(config, checkpoint):
        captured["checkpoint"] = Path(checkpoint)
        config.model.moe.enabled = True
        return True

    monkeypatch.setattr(loading, "apply_moe_config_from_checkpoint", fake_apply_moe)

    dummy_config = _DummyConfig()

    # Stub out every torch-dependent import that load_generator pulls in lazily.
    _install_stub_modules(monkeypatch, dummy_config)

    generator, config, tokenizer = loading.load_generator(
        checkpoint=latest,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        device="cpu",
    )

    # apply_moe_config received the RESOLVED step_* dir, not the pointer file.
    assert captured["checkpoint"] == ckpt_dir
    assert captured["checkpoint"].name == "step_00001000"
    # Config was flipped to MoE.
    assert config.model.moe.enabled is True


def test_load_generator_missing_checkpoint_raises(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: {}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        loading.load_generator(
            checkpoint=tmp_path / "does_not_exist",
            config_path=config_path,
            tokenizer_path=tmp_path,
            device="cpu",
        )


# --------------------------------------------------------------------------- #
# Test doubles                                                                  #
# --------------------------------------------------------------------------- #


class _DummyMoE:
    enabled = False
    num_experts = 0
    num_shared_experts = 0
    top_k = 0
    moe_layers = None


class _DummyModelCfg:
    vocab_size = 32768

    def __init__(self):
        self.moe = _DummyMoE()


class _DummyConfig:
    def __init__(self):
        self.model = _DummyModelCfg()


def _install_stub_modules(monkeypatch, dummy_config):
    """Inject lightweight stand-ins for the torch-heavy modules load_generator
    imports lazily, so the function runs without building a real model."""

    # torch
    torch_mod = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available():
            return False

    torch_mod.cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    # safetensors.safe_open — make vocab inspection a no-op (no model.safetensors
    # tensors). It is wrapped in try/except in load_generator anyway.
    safetensors_mod = types.ModuleType("safetensors")

    def _safe_open(*_a, **_k):
        raise RuntimeError("no real tensors in test")

    safetensors_mod.safe_open = _safe_open
    monkeypatch.setitem(sys.modules, "safetensors", safetensors_mod)

    # cola_coder.inference.generator.CodeGenerator
    gen_mod = types.ModuleType("cola_coder.inference.generator")
    gen_mod.CodeGenerator = lambda **kwargs: MagicMock(name="CodeGenerator")
    monkeypatch.setitem(sys.modules, "cola_coder.inference.generator", gen_mod)

    # cola_coder.model.config.Config
    cfg_mod = types.ModuleType("cola_coder.model.config")

    class _Config:
        @staticmethod
        def from_yaml(_path):
            return dummy_config

    cfg_mod.Config = _Config
    cfg_mod.get_storage_config = lambda: MagicMock(tokenizer_path="unused")
    monkeypatch.setitem(sys.modules, "cola_coder.model.config", cfg_mod)

    # cola_coder.model.transformer.Transformer
    tf_mod = types.ModuleType("cola_coder.model.transformer")

    def _transformer(_model_cfg):
        m = MagicMock(name="Transformer")
        m.to = lambda _device: m
        return m

    tf_mod.Transformer = _transformer
    monkeypatch.setitem(sys.modules, "cola_coder.model.transformer", tf_mod)

    # cola_coder.tokenizer.tokenizer_utils.CodeTokenizer
    tok_mod = types.ModuleType("cola_coder.tokenizer.tokenizer_utils")
    tok_mod.CodeTokenizer = lambda _path: MagicMock(name="CodeTokenizer")
    monkeypatch.setitem(sys.modules, "cola_coder.tokenizer.tokenizer_utils", tok_mod)

    # cola_coder.training.checkpoint.load_model_only
    ckpt_mod = types.ModuleType("cola_coder.training.checkpoint")
    ckpt_mod.load_model_only = lambda _path, model, device=None: model
    monkeypatch.setitem(sys.modules, "cola_coder.training.checkpoint", ckpt_mod)
