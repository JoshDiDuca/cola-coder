"""BUG-128: build the model from the checkpoint's saved architecture, not a
possibly-wrong --config. The interactive-generation menu passed configs/tiny.yaml
(dim=512) for the dim=768 small_react_best checkpoint → load_state_dict size
mismatch crash. apply_model_config_from_checkpoint reads metadata.json and overrides
config.model so the built model always matches the checkpoint.
"""

import json
from pathlib import Path

import pytest

from cola_coder.inference.loading import apply_model_config_from_checkpoint
from cola_coder.model.config import Config


def _tiny_config():
    return Config.from_yaml("configs/tiny.yaml")


def _write_meta(d: Path, model_cfg: dict):
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(
        json.dumps({"step": 100, "config": {"model": model_cfg}}), encoding="utf-8"
    )


class TestApplyModelConfig:
    def test_overrides_architecture_from_metadata(self, tmp_path):
        cfg = _tiny_config()
        ckpt = tmp_path / "step_00000100"
        _write_meta(ckpt, {"dim": 768, "n_layers": 12, "n_heads": 12, "n_kv_heads": 4})
        assert apply_model_config_from_checkpoint(cfg, str(ckpt)) is True
        assert cfg.model.dim == 768
        assert cfg.model.n_layers == 12
        assert cfg.model.n_heads == 12
        assert cfg.model.n_kv_heads == 4

    def test_resolves_latest_pointer(self, tmp_path):
        real = tmp_path / "step_00000200"
        _write_meta(real, {"dim": 640})
        latest = tmp_path / "latest"
        latest.write_text(str(real), encoding="utf-8")
        cfg = _tiny_config()
        assert apply_model_config_from_checkpoint(cfg, str(latest)) is True
        assert cfg.model.dim == 640

    def test_no_metadata_returns_false_and_leaves_config(self, tmp_path):
        d = tmp_path / "no_meta"
        d.mkdir()
        cfg = _tiny_config()
        orig_dim = cfg.model.dim
        assert apply_model_config_from_checkpoint(cfg, str(d)) is False
        assert cfg.model.dim == orig_dim

    def test_nested_fields_skipped(self, tmp_path):
        # rope_scaling is a dict in metadata — must not clobber the config object.
        cfg = _tiny_config()
        ckpt = tmp_path / "step_x"
        _write_meta(ckpt, {"dim": 768, "rope_scaling": {"type": "yarn", "factor": 2.0}})
        apply_model_config_from_checkpoint(cfg, str(ckpt))
        assert cfg.model.dim == 768
        # rope_scaling stays the config's own (object), not a raw dict.
        assert not isinstance(cfg.model.rope_scaling, dict)

    def test_real_checkpoint_reproduces_and_fixes_the_crash(self):
        # The exact bug: tiny.yaml (dim=512) vs the small_react_best checkpoint (768).
        ckpt = Path("checkpoints/small_react_best/step_00004500")
        if not (ckpt / "metadata.json").exists():
            pytest.skip("real small_react_best checkpoint not present")
        cfg = _tiny_config()
        assert cfg.model.dim != 768  # tiny starts mismatched (would crash)
        assert apply_model_config_from_checkpoint(cfg, str(ckpt)) is True
        assert cfg.model.dim == 768  # now matches the checkpoint → no size mismatch
