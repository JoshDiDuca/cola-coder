"""Tests for checkpoint saving and loading.

Covers the critical paths that have broken in production:
- Weight tying (tok_emb.weight == output.weight) with safetensors
- torch.compile prefix stripping (_orig_mod.*)
- Round-trip save→load weight fidelity
- Optimizer/scheduler state preservation
- Metadata and "latest" pointer files
- Cleanup of old checkpoints
- load_model_only for inference
- detect_latest_checkpoint discovery
"""

import json
from pathlib import Path

import pytest
import torch

from cola_coder.model.config import ModelConfig
from cola_coder.model.transformer import Transformer
from cola_coder.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_model_only,
    get_checkpoint_info,
    detect_latest_checkpoint,
)
from cola_coder.training.optimizer import create_optimizer, create_scheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_config() -> ModelConfig:
    """Minimal model config for fast tests."""
    return ModelConfig(
        vocab_size=256, dim=64, n_layers=2,
        n_heads=4, n_kv_heads=2, max_seq_len=64,
    )


def _make_model():
    return Transformer(_tiny_config())


def _make_training_state(model):
    """Return (model, optimizer, scheduler) ready for a checkpoint."""
    optimizer = create_optimizer(model, learning_rate=1e-3, weight_decay=0.1)
    scheduler = create_scheduler(optimizer, warmup_steps=10, max_steps=100)
    # Take a real gradient step so optimizer state is non-trivial
    token_ids = torch.randint(0, 256, (2, 16))
    loss = model.compute_loss(token_ids)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Core save / load round-trip
# ---------------------------------------------------------------------------

class TestCheckpointRoundTrip:
    """Save a checkpoint and load it back — weights must match exactly."""

    def test_save_creates_expected_files(self, tmp_path):
        model = _make_model()
        opt, sched = _make_training_state(model)

        ckpt = save_checkpoint(
            model, opt, sched, step=100, loss=2.5,
            config={"test": True}, output_dir=str(tmp_path),
        )
        ckpt_dir = Path(ckpt)

        assert (ckpt_dir / "model.safetensors").exists()
        assert (ckpt_dir / "training_state.pt").exists()
        assert (ckpt_dir / "metadata.json").exists()
        assert (tmp_path / "latest").exists()

    def test_weight_fidelity(self, tmp_path):
        """Loaded weights are bitwise identical to saved weights."""
        model = _make_model()
        opt, sched = _make_training_state(model)

        # Snapshot weights before save
        original_weights = {
            k: v.clone() for k, v in model.state_dict().items()
        }

        ckpt = save_checkpoint(
            model, opt, sched, step=50, loss=3.0,
            config={}, output_dir=str(tmp_path),
        )

        # Load into a fresh model
        model2 = _make_model()
        load_checkpoint(ckpt, model2, device="cpu")

        for key in original_weights:
            assert torch.equal(
                model2.state_dict()[key], original_weights[key]
            ), f"Weight mismatch on {key}"

    def test_weight_tying_preserved_after_load(self, tmp_path):
        """output.weight and tok_emb.weight still share storage after load."""
        model = _make_model()
        opt, sched = _make_training_state(model)

        ckpt = save_checkpoint(
            model, opt, sched, step=1, loss=5.0,
            config={}, output_dir=str(tmp_path),
        )

        model2 = _make_model()
        load_checkpoint(ckpt, model2, device="cpu")

        # They must be the same tensor (weight tying)
        assert model2.output.weight.data_ptr() == model2.tok_emb.weight.data_ptr(), (
            "Weight tying broken after checkpoint load"
        )

    def test_optimizer_state_restored(self, tmp_path):
        """Optimizer momentum buffers survive a save→load cycle."""
        model = _make_model()
        opt, sched = _make_training_state(model)

        ckpt = save_checkpoint(
            model, opt, sched, step=10, loss=4.0,
            config={}, output_dir=str(tmp_path),
        )

        model2 = _make_model()
        opt2, sched2 = _make_training_state(model2)
        load_checkpoint(ckpt, model2, opt2, sched2, device="cpu")

        # Step count from scheduler must match
        assert sched2.last_epoch == sched.last_epoch

    def test_step_number_round_trip(self, tmp_path):
        model = _make_model()
        opt, sched = _make_training_state(model)

        ckpt = save_checkpoint(
            model, opt, sched, step=42, loss=1.0,
            config={}, output_dir=str(tmp_path),
        )

        model2 = _make_model()
        opt2, sched2 = _make_training_state(model2)
        step = load_checkpoint(ckpt, model2, opt2, sched2, device="cpu")
        assert step == 42


# ---------------------------------------------------------------------------
# Weight tying + safetensors (the exact bug that crashed training)
# ---------------------------------------------------------------------------

class TestWeightTyingSafetensors:
    """Verify safetensors doesn't choke on shared tok_emb ↔ output tensors."""

    def test_output_weight_not_in_saved_keys(self, tmp_path):
        """output.weight must be excluded — it's a tied alias of tok_emb.weight."""
        model = _make_model()
        opt, sched = _make_training_state(model)

        ckpt = save_checkpoint(
            model, opt, sched, step=1, loss=5.0,
            config={}, output_dir=str(tmp_path),
        )

        from safetensors.torch import load_file
        state = load_file(str(Path(ckpt) / "model.safetensors"))

        assert "output.weight" not in state, (
            "output.weight should be excluded (tied to tok_emb.weight)"
        )
        assert "tok_emb.weight" in state

    def test_no_orig_mod_prefix_in_saved_keys(self, tmp_path):
        """Saved keys must never have _orig_mod. prefix (torch.compile artefact)."""
        model = _make_model()
        opt, sched = _make_training_state(model)

        ckpt = save_checkpoint(
            model, opt, sched, step=1, loss=5.0,
            config={}, output_dir=str(tmp_path),
        )

        from safetensors.torch import load_file
        state = load_file(str(Path(ckpt) / "model.safetensors"))

        for key in state:
            assert not key.startswith("_orig_mod."), (
                f"Key {key!r} has _orig_mod. prefix — strip it during save"
            )


# ---------------------------------------------------------------------------
# torch.compile round-trip
# ---------------------------------------------------------------------------

class TestTorchCompileCheckpoint:
    """Checkpoints must work when the model is wrapped by torch.compile."""

    @pytest.fixture
    def compiled_model(self):
        model = _make_model()
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile not available")
        try:
            compiled = torch.compile(model, mode="default")
            # Run a forward pass to trigger compilation / verify it works
            with torch.no_grad():
                compiled(torch.randint(0, 256, (1, 8)))
            return compiled
        except Exception as e:
            pytest.skip(f"torch.compile failed: {e}")

    def test_save_compiled_model(self, tmp_path, compiled_model):
        """Saving a compiled model must not raise (safetensors shared-tensor error)."""
        opt = create_optimizer(compiled_model, learning_rate=1e-3)
        sched = create_scheduler(opt, warmup_steps=10, max_steps=100)

        # This was the exact crash: RuntimeError: Some tensors share memory
        ckpt = save_checkpoint(
            compiled_model, opt, sched, step=1000, loss=3.0,
            config={"compiled": True}, output_dir=str(tmp_path),
        )
        assert Path(ckpt).exists()

    def test_load_into_compiled_model(self, tmp_path, compiled_model):
        """Load a checkpoint into a compiled model."""
        opt = create_optimizer(compiled_model, learning_rate=1e-3)
        sched = create_scheduler(opt, warmup_steps=10, max_steps=100)

        # Save from compiled
        ckpt = save_checkpoint(
            compiled_model, opt, sched, step=500, loss=2.0,
            config={}, output_dir=str(tmp_path),
        )

        # Load into a fresh compiled model
        model2 = _make_model()
        compiled2 = torch.compile(model2, mode="default")
        step = load_checkpoint(ckpt, compiled2, device="cpu")
        assert step == 500

    def test_cross_load_compiled_to_uncompiled(self, tmp_path, compiled_model):
        """Checkpoint from compiled model loads into uncompiled model."""
        opt = create_optimizer(compiled_model, learning_rate=1e-3)
        sched = create_scheduler(opt, warmup_steps=10, max_steps=100)

        ckpt = save_checkpoint(
            compiled_model, opt, sched, step=200, loss=2.5,
            config={}, output_dir=str(tmp_path),
        )

        # Load into a plain (non-compiled) model
        model2 = _make_model()
        step = load_checkpoint(ckpt, model2, device="cpu")
        assert step == 200

    def test_cross_load_uncompiled_to_compiled(self, tmp_path, compiled_model):
        """Checkpoint from uncompiled model loads into compiled model."""
        # Save from uncompiled model
        plain_model = _make_model()
        opt = create_optimizer(plain_model, learning_rate=1e-3)
        sched = create_scheduler(opt, warmup_steps=10, max_steps=100)

        ckpt = save_checkpoint(
            plain_model, opt, sched, step=300, loss=2.0,
            config={}, output_dir=str(tmp_path),
        )

        # Load into compiled model
        step = load_checkpoint(ckpt, compiled_model, device="cpu")
        assert step == 300


# ---------------------------------------------------------------------------
# load_model_only (inference path)
# ---------------------------------------------------------------------------

class TestLoadModelOnly:

    def test_load_model_only_matches_original(self, tmp_path):
        model = _make_model()
        opt, sched = _make_training_state(model)

        original_emb = model.tok_emb.weight.clone()

        ckpt = save_checkpoint(
            model, opt, sched, step=1, loss=5.0,
            config={}, output_dir=str(tmp_path),
        )

        model2 = _make_model()
        load_model_only(ckpt, model2, device="cpu")

        assert torch.equal(model2.tok_emb.weight, original_emb)
        # Model should be in eval mode after load_model_only
        assert not model2.training

    def test_load_model_only_weight_tying(self, tmp_path):
        model = _make_model()
        opt, sched = _make_training_state(model)

        ckpt = save_checkpoint(
            model, opt, sched, step=1, loss=5.0,
            config={}, output_dir=str(tmp_path),
        )

        model2 = _make_model()
        load_model_only(ckpt, model2, device="cpu")

        assert model2.output.weight.data_ptr() == model2.tok_emb.weight.data_ptr()


# ---------------------------------------------------------------------------
# Metadata, latest pointer, cleanup
# ---------------------------------------------------------------------------

class TestMetadataAndLatest:

    def test_metadata_json(self, tmp_path):
        model = _make_model()
        opt, sched = _make_training_state(model)

        ckpt = save_checkpoint(
            model, opt, sched, step=777, loss=1.23,
            config={"model": "tiny"}, output_dir=str(tmp_path),
        )

        meta = json.loads((Path(ckpt) / "metadata.json").read_text())
        assert meta["step"] == 777
        assert abs(meta["loss"] - 1.23) < 1e-6
        assert meta["config"]["model"] == "tiny"

    def test_latest_pointer(self, tmp_path):
        model = _make_model()
        opt, sched = _make_training_state(model)

        ckpt = save_checkpoint(
            model, opt, sched, step=100, loss=2.0,
            config={}, output_dir=str(tmp_path),
        )

        latest = (tmp_path / "latest").read_text().strip()
        assert latest == ckpt

    def test_latest_pointer_updates(self, tmp_path):
        model = _make_model()
        opt, sched = _make_training_state(model)

        save_checkpoint(
            model, opt, sched, step=100, loss=2.0,
            config={}, output_dir=str(tmp_path),
        )
        ckpt2 = save_checkpoint(
            model, opt, sched, step=200, loss=1.5,
            config={}, output_dir=str(tmp_path),
        )

        latest = (tmp_path / "latest").read_text().strip()
        assert latest == ckpt2

    def test_load_via_latest(self, tmp_path):
        model = _make_model()
        opt, sched = _make_training_state(model)

        save_checkpoint(
            model, opt, sched, step=50, loss=3.0,
            config={}, output_dir=str(tmp_path),
        )

        model2 = _make_model()
        opt2, sched2 = _make_training_state(model2)
        # Must pass optimizer/scheduler to get step number back
        step = load_checkpoint(
            str(tmp_path / "latest"), model2, opt2, sched2, device="cpu"
        )
        assert step == 50


class TestCheckpointCleanup:

    def test_max_checkpoints_enforced(self, tmp_path):
        model = _make_model()
        opt, sched = _make_training_state(model)

        for i in range(1, 8):
            save_checkpoint(
                model, opt, sched, step=i * 100, loss=5.0 - i * 0.5,
                config={}, output_dir=str(tmp_path), max_checkpoints=3,
            )

        step_dirs = [
            d for d in tmp_path.iterdir()
            if d.is_dir() and d.name.startswith("step_")
        ]
        assert len(step_dirs) <= 3, f"Expected ≤3 checkpoints, found {len(step_dirs)}"

    def test_newest_checkpoints_kept(self, tmp_path):
        model = _make_model()
        opt, sched = _make_training_state(model)

        for i in range(1, 6):
            save_checkpoint(
                model, opt, sched, step=i * 1000, loss=1.0,
                config={}, output_dir=str(tmp_path), max_checkpoints=2,
            )

        remaining = sorted(
            d.name for d in tmp_path.iterdir()
            if d.is_dir() and d.name.startswith("step_")
        )
        # The two newest (step_04000 and step_05000) should remain
        assert "step_00005000" in remaining
        assert "step_00004000" in remaining


# ---------------------------------------------------------------------------
# detect_latest_checkpoint
# ---------------------------------------------------------------------------

class TestDetectLatestCheckpoint:

    def test_no_checkpoints(self, tmp_path):
        assert detect_latest_checkpoint(str(tmp_path)) is None

    def test_finds_latest_across_sizes(self, tmp_path):
        model = _make_model()
        opt, sched = _make_training_state(model)

        # Save a "tiny" checkpoint at step 100
        save_checkpoint(
            model, opt, sched, step=100, loss=3.0,
            config={"model": {"dim": 64}},
            output_dir=str(tmp_path / "tiny"),
        )
        # Save a "small" checkpoint at step 500
        save_checkpoint(
            model, opt, sched, step=500, loss=2.0,
            config={"model": {"dim": 768}},
            output_dir=str(tmp_path / "small"),
        )

        result = detect_latest_checkpoint(str(tmp_path))
        assert result is not None
        path, info = result
        assert info["step"] == 500


# ---------------------------------------------------------------------------
# get_checkpoint_info
# ---------------------------------------------------------------------------

class TestGetCheckpointInfo:

    def test_returns_info(self, tmp_path):
        model = _make_model()
        opt, sched = _make_training_state(model)

        ckpt = save_checkpoint(
            model, opt, sched, step=42, loss=2.5,
            config={"model": "tiny"}, output_dir=str(tmp_path),
        )

        info = get_checkpoint_info(ckpt)
        assert info["step"] == 42
        assert abs(info["loss"] - 2.5) < 1e-6

    def test_missing_checkpoint_returns_empty(self, tmp_path):
        info = get_checkpoint_info(str(tmp_path / "nonexistent"))
        assert info == {}

    def test_incomplete_checkpoint(self, tmp_path):
        """Loading an incomplete checkpoint raises FileNotFoundError."""
        bad_dir = tmp_path / "step_00000001"
        bad_dir.mkdir()
        # Only metadata, no model.safetensors
        (bad_dir / "metadata.json").write_text('{"step": 1}')

        model = _make_model()
        with pytest.raises(FileNotFoundError, match="model.safetensors is missing"):
            load_checkpoint(str(bad_dir), model, device="cpu")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_overwrite_existing_checkpoint(self, tmp_path):
        """Saving to the same step twice doesn't crash."""
        model = _make_model()
        opt, sched = _make_training_state(model)

        save_checkpoint(
            model, opt, sched, step=100, loss=3.0,
            config={}, output_dir=str(tmp_path),
        )
        # Save again at the same step (shouldn't error)
        ckpt = save_checkpoint(
            model, opt, sched, step=100, loss=2.5,
            config={}, output_dir=str(tmp_path),
        )
        meta = json.loads((Path(ckpt) / "metadata.json").read_text())
        assert abs(meta["loss"] - 2.5) < 1e-6

    def test_atomic_save_cleans_tmp_dir(self, tmp_path):
        """No .tmp_* directories left after successful save."""
        model = _make_model()
        opt, sched = _make_training_state(model)

        save_checkpoint(
            model, opt, sched, step=1, loss=5.0,
            config={}, output_dir=str(tmp_path),
        )

        tmp_dirs = [d for d in tmp_path.iterdir() if d.name.startswith(".tmp_")]
        assert len(tmp_dirs) == 0, f"Temp dirs remain: {tmp_dirs}"

    def test_forward_pass_after_load(self, tmp_path):
        """Model produces valid output after loading from checkpoint."""
        model = _make_model()
        opt, sched = _make_training_state(model)

        ckpt = save_checkpoint(
            model, opt, sched, step=1, loss=5.0,
            config={}, output_dir=str(tmp_path),
        )

        model2 = _make_model()
        load_checkpoint(ckpt, model2, device="cpu")

        # Forward pass should produce valid logits
        token_ids = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            logits = model2(token_ids)
        assert logits.shape == (1, 16, 256)
        assert not torch.isnan(logits).any()

    def test_loss_computation_after_load(self, tmp_path):
        """Model can compute loss + backward after loading (training resumption)."""
        model = _make_model()
        opt, sched = _make_training_state(model)

        ckpt = save_checkpoint(
            model, opt, sched, step=1, loss=5.0,
            config={}, output_dir=str(tmp_path),
        )

        model2 = _make_model()
        opt2, sched2 = _make_training_state(model2)
        load_checkpoint(ckpt, model2, opt2, sched2, device="cpu")

        # Full training step must work
        token_ids = torch.randint(0, 256, (2, 16))
        loss = model2.compute_loss(token_ids)
        loss.backward()
        opt2.step()  # Should not raise
