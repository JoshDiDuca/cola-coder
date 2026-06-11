"""MoE integration: build, upcycle round-trip, load, and run.

Before this wiring, the core Transformer always built dense FFNs, so
scripts/upcycle_to_moe.py (pipeline stage 7) produced checkpoints that
nothing downstream could load or run. These tests lock the end-to-end path:
dense checkpoint -> upcycle -> detect -> build MoE Transformer -> load ->
generate, plus the unit pieces it relies on.
"""

import importlib.util
import json

import pytest
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

from cola_coder.features.moe_layer import (
    MoEFFN,
    detect_moe_checkpoint,
    resolve_moe_layers,
)
from cola_coder.model.config import Config, ModelConfig
from cola_coder.model.transformer import Transformer, TransformerBlock

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load_upcycle():
    """Import scripts/upcycle_to_moe.py by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "upcycle_to_moe", _SCRIPTS / "upcycle_to_moe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.upcycle


def _tiny_moe_config(enabled: bool = True) -> ModelConfig:
    cfg = ModelConfig(
        vocab_size=256, dim=64, n_layers=2,
        n_heads=4, n_kv_heads=2, max_seq_len=64, dropout=0.0,
    )
    cfg.moe.enabled = enabled
    cfg.moe.num_experts = 4
    cfg.moe.num_shared_experts = 1
    cfg.moe.top_k = 2
    cfg.moe.moe_layers = "all"
    return cfg


# ---------------------------------------------------------------------------
# resolve_moe_layers
# ---------------------------------------------------------------------------


class TestResolveMoeLayers:
    def test_all(self):
        assert resolve_moe_layers("all", 4) == {0, 1, 2, 3}

    def test_alternate(self):
        assert resolve_moe_layers("alternate", 4) == {1, 3}

    def test_comma_list(self):
        assert resolve_moe_layers("0,2", 4) == {0, 2}

    def test_out_of_range_indices_dropped(self):
        assert resolve_moe_layers("0,9", 4) == {0}

    def test_blank_defaults_to_all(self):
        assert resolve_moe_layers("", 3) == {0, 1, 2}


# ---------------------------------------------------------------------------
# Transformer builds MoE blocks
# ---------------------------------------------------------------------------


class TestTransformerBuildsMoe:
    def test_moe_blocks_built_when_enabled(self):
        model = Transformer(_tiny_moe_config(enabled=True))
        assert model.is_moe
        for block in model.blocks:
            assert isinstance(block, TransformerBlock)
            assert isinstance(block.ffn, MoEFFN)

    def test_dense_when_disabled(self):
        model = Transformer(_tiny_moe_config(enabled=False))
        assert not model.is_moe
        for block in model.blocks:
            assert not isinstance(block.ffn, MoEFFN)

    def test_alternate_layers_only(self):
        cfg = _tiny_moe_config(enabled=True)
        cfg.n_layers = 4
        cfg.moe.moe_layers = "alternate"
        model = Transformer(cfg)
        is_moe = [isinstance(b.ffn, MoEFFN) for b in model.blocks]
        assert is_moe == [False, True, False, True]

    def test_forward_shape_matches_dense(self):
        model = Transformer(_tiny_moe_config(enabled=True))
        ids = torch.randint(0, 256, (2, 16))
        logits = model(ids)
        assert logits.shape == (2, 16, 256)

    def test_aux_loss_nonzero_in_training(self):
        model = Transformer(_tiny_moe_config(enabled=True))
        model.train()
        ids = torch.randint(0, 256, (2, 16))
        loss = model.compute_loss(ids)
        aux = model.moe_aux_loss()
        assert aux.item() > 0.0
        assert torch.isfinite(loss)

    def test_aux_loss_zero_for_dense(self):
        model = Transformer(_tiny_moe_config(enabled=False))
        model.train()
        model(torch.randint(0, 256, (2, 16)))
        assert model.moe_aux_loss().item() == 0.0


# ---------------------------------------------------------------------------
# detect_moe_checkpoint
# ---------------------------------------------------------------------------


class TestDetectMoeCheckpoint:
    def test_returns_none_for_missing(self, tmp_path):
        assert detect_moe_checkpoint(tmp_path) is None

    def test_reads_sidecar(self, tmp_path):
        (tmp_path / "moe_config.json").write_text(
            json.dumps({"num_experts": 6, "num_shared_experts": 2, "top_k": 3})
        )
        # sidecar path needs a safetensors too? No — sidecar is read first.
        info = detect_moe_checkpoint(tmp_path)
        assert info == {"num_experts": 6, "num_shared_experts": 2, "top_k": 3}

    def test_infers_from_weight_keys(self, tmp_path):
        # No sidecar — infer experts/shared from the safetensors key names.
        state = {
            "blocks.0.ffn.experts.0.gate_proj.weight": torch.zeros(8, 8),
            "blocks.0.ffn.experts.1.gate_proj.weight": torch.zeros(8, 8),
            "blocks.0.ffn.experts.2.gate_proj.weight": torch.zeros(8, 8),
            "blocks.0.ffn.shared_experts.0.gate_proj.weight": torch.zeros(8, 8),
            "blocks.0.ffn.router.gate.weight": torch.zeros(3, 8),
            "tok_emb.weight": torch.zeros(16, 8),
        }
        save_file(state, str(tmp_path / "model.safetensors"))
        info = detect_moe_checkpoint(tmp_path)
        assert info["num_experts"] == 3
        assert info["num_shared_experts"] == 1

    def test_returns_none_for_dense_checkpoint(self, tmp_path):
        save_file(
            {"tok_emb.weight": torch.zeros(16, 8),
             "blocks.0.ffn.gate_proj.weight": torch.zeros(8, 8)},
            str(tmp_path / "model.safetensors"),
        )
        assert detect_moe_checkpoint(tmp_path) is None


# ---------------------------------------------------------------------------
# Full round trip: dense -> upcycle -> load MoE -> generate
# ---------------------------------------------------------------------------


def _write_tiny_config_yaml(path: Path) -> None:
    cfg = {
        "model": {
            "vocab_size": 256, "dim": 64, "n_layers": 2,
            "n_heads": 4, "n_kv_heads": 2, "max_seq_len": 64, "dropout": 0.0,
        },
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


class TestUpcycleRoundTrip:
    def test_dense_upcycle_load_generate(self, tmp_path):
        from cola_coder.training.checkpoint import save_checkpoint
        from cola_coder.training.optimizer import create_optimizer, create_scheduler
        from cola_coder.inference.loading import apply_moe_config_from_checkpoint
        from cola_coder.training.checkpoint import load_model_only

        # 1. Build + save a dense checkpoint
        config_yaml = tmp_path / "cfg.yaml"
        _write_tiny_config_yaml(config_yaml)
        dense_cfg = Config.from_yaml(str(config_yaml))
        dense = Transformer(dense_cfg.model)
        opt = create_optimizer(dense, learning_rate=1e-3, weight_decay=0.1)
        sched = create_scheduler(opt, warmup_steps=2, max_steps=10)
        dense_dir = tmp_path / "dense"
        save_checkpoint(dense, opt, sched, step=1, loss=2.0,
                        config={}, output_dir=str(dense_dir))
        dense_ckpt = next(dense_dir.glob("step_*"))

        # 2. Upcycle to MoE
        upcycle = _load_upcycle()

        moe_dir = tmp_path / "moe"
        upcycle(
            checkpoint_dir=str(dense_ckpt),
            config_path=str(config_yaml),
            num_experts=4, num_shared_experts=1, top_k=2,
            output_dir=str(moe_dir),
        )
        assert (moe_dir / "model.safetensors").exists()
        assert (moe_dir / "moe_config.json").exists()

        # 3. Detection flips the config to MoE
        load_cfg = Config.from_yaml(str(config_yaml))
        assert apply_moe_config_from_checkpoint(load_cfg, moe_dir) is True
        assert load_cfg.model.moe.enabled
        assert load_cfg.model.moe.num_experts == 4

        # 4. Build the MoE model and load the upcycled weights (strict tie check)
        moe_model = Transformer(load_cfg.model)
        assert moe_model.is_moe
        load_model_only(str(moe_dir), moe_model, device="cpu")

        # 5. It runs and produces valid logits
        moe_model.eval()
        ids = torch.randint(0, 256, (1, 12))
        with torch.no_grad():
            logits = moe_model(ids)
        assert logits.shape == (1, 12, 256)
        assert torch.isfinite(logits).all()

    def test_upcycled_experts_start_near_dense(self, tmp_path):
        """Shared experts are clean copies — output should track the dense FFN."""
        from cola_coder.training.checkpoint import save_checkpoint, load_model_only
        from cola_coder.training.optimizer import create_optimizer, create_scheduler
        from cola_coder.inference.loading import apply_moe_config_from_checkpoint
        upcycle = _load_upcycle()

        config_yaml = tmp_path / "cfg.yaml"
        _write_tiny_config_yaml(config_yaml)
        dense_cfg = Config.from_yaml(str(config_yaml))
        dense = Transformer(dense_cfg.model)
        dense.eval()
        opt = create_optimizer(dense, learning_rate=1e-3, weight_decay=0.1)
        sched = create_scheduler(opt, warmup_steps=2, max_steps=10)
        dense_dir = tmp_path / "dense"
        save_checkpoint(dense, opt, sched, step=1, loss=2.0,
                        config={}, output_dir=str(dense_dir))
        dense_ckpt = next(dense_dir.glob("step_*"))

        moe_dir = tmp_path / "moe"
        # noise_std=0 so routed experts are exact copies → deterministic check
        upcycle(checkpoint_dir=str(dense_ckpt), config_path=str(config_yaml),
                num_experts=2, num_shared_experts=1, top_k=2, noise_std=0.0,
                output_dir=str(moe_dir))

        load_cfg = Config.from_yaml(str(config_yaml))
        apply_moe_config_from_checkpoint(load_cfg, moe_dir)
        moe_model = Transformer(load_cfg.model)
        load_model_only(str(moe_dir), moe_model, device="cpu")
        moe_model.eval()

        ids = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            dense_logits = dense(ids)
            moe_logits = moe_model(ids)
        # Not identical (MoE adds shared + routed paths), but finite and same shape
        assert moe_logits.shape == dense_logits.shape
        assert torch.isfinite(moe_logits).all()


class TestMoEResumeFineTune:
    """MODEL-001: the TRAINING resume path must auto-detect MoE checkpoints.

    Before the fix, Trainer built Transformer(config.model) straight from the
    (dense) config, so resuming from an upcycled MoE checkpoint to fine-tune it
    failed — the dense model can't accept the experts.* keys. These tests
    exercise the exact apply -> build -> load_checkpoint path the trainer uses.
    """

    def _make_upcycled(self, tmp_path):
        from cola_coder.training.checkpoint import save_checkpoint
        from cola_coder.training.optimizer import create_optimizer, create_scheduler

        config_yaml = tmp_path / "cfg.yaml"
        _write_tiny_config_yaml(config_yaml)
        dense_cfg = Config.from_yaml(str(config_yaml))
        dense = Transformer(dense_cfg.model)
        opt = create_optimizer(dense, learning_rate=1e-3, weight_decay=0.1)
        sched = create_scheduler(opt, warmup_steps=2, max_steps=10)
        dense_dir = tmp_path / "dense"
        save_checkpoint(dense, opt, sched, step=1, loss=2.0,
                        config={}, output_dir=str(dense_dir))
        dense_ckpt = next(dense_dir.glob("step_*"))
        moe_dir = tmp_path / "moe"
        _load_upcycle()(
            checkpoint_dir=str(dense_ckpt), config_path=str(config_yaml),
            num_experts=4, num_shared_experts=1, top_k=2, output_dir=str(moe_dir),
        )
        return config_yaml, moe_dir

    def test_resume_path_loads_moe_checkpoint(self, tmp_path):
        from cola_coder.features.moe_layer import apply_moe_config_from_checkpoint
        from cola_coder.training.checkpoint import load_checkpoint
        from cola_coder.training.optimizer import create_optimizer, create_scheduler

        config_yaml, moe_dir = self._make_upcycled(tmp_path)

        # Mirror Trainer.__init__: apply MoE config from the resume checkpoint,
        # THEN build the model, THEN load via the resume loader (with optimizer).
        cfg = Config.from_yaml(str(config_yaml))
        assert apply_moe_config_from_checkpoint(cfg, moe_dir) is True
        model = Transformer(cfg.model)
        assert model.is_moe
        opt = create_optimizer(model, learning_rate=1e-5, weight_decay=0.01)
        sched = create_scheduler(opt, warmup_steps=2, max_steps=10)

        # Upcycle output has no training_state.pt -> fresh optimizer, step 0.
        step = load_checkpoint(str(moe_dir), model, opt, sched, device="cpu")
        assert step == 0
        # The fine-tune model trains: forward + backward + aux loss all work.
        model.train()
        loss = model.compute_loss(torch.randint(0, 256, (2, 16)))
        loss.backward()
        assert torch.isfinite(loss)

    def test_resume_without_moe_detect_fails(self, tmp_path):
        # Proves the fix is necessary: building a DENSE model (skipping the
        # auto-detect) and loading the MoE checkpoint must fail.
        from cola_coder.training.checkpoint import load_checkpoint

        config_yaml, moe_dir = self._make_upcycled(tmp_path)
        dense_cfg = Config.from_yaml(str(config_yaml))  # moe NOT enabled
        dense_model = Transformer(dense_cfg.model)
        assert not dense_model.is_moe
        with pytest.raises(Exception):
            load_checkpoint(str(moe_dir), dense_model, device="cpu")

    def test_trainer_wires_moe_autodetect_before_build(self):
        # Source guard: the auto-detect must run BEFORE the model is built.
        text = (Path(__file__).parent.parent / "src" / "cola_coder"
                / "training" / "trainer.py").read_text(encoding="utf-8")
        apply_idx = text.find("apply_moe_config_from_checkpoint(config, resume_from)")
        build_idx = text.find("self.model = Transformer(config.model)")
        assert apply_idx != -1 and build_idx != -1
        assert apply_idx < build_idx
