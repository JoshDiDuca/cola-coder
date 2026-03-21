"""Tests for checkpoint averaging (uniform and EMA).

Covers:
- Uniform average produces correct values (mean of known weights)
- EMA produces expected decay behavior
- find_checkpoints sorts correctly by step number
- average_last_k picks the correct checkpoints
- Missing checkpoint raises FileNotFoundError
- Mismatched keys raises ValueError
- output.weight (tied weight) is excluded from saved state dict
- Averaged checkpoint can be loaded into a real model
- Average of identical checkpoints equals the original
- EMA with decay=0 collapses to last checkpoint
- EMA with decay approaching 1 stays near first checkpoint
- average_last_k with k > available raises ValueError
- Averaging 1 checkpoint is a no-op copy
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from cola_coder.model.config import ModelConfig
from cola_coder.model.transformer import Transformer
from cola_coder.training.checkpoint import save_checkpoint
from cola_coder.training.checkpoint_average import AverageResult, CheckpointAverager
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


def _make_model() -> Transformer:
    return Transformer(_tiny_config())


def _make_training_state(model: Transformer):
    """Return (optimizer, scheduler) after one real gradient step."""
    optimizer = create_optimizer(model, learning_rate=1e-3, weight_decay=0.1)
    scheduler = create_scheduler(optimizer, warmup_steps=10, max_steps=100)
    token_ids = torch.randint(0, 256, (2, 16))
    loss = model.compute_loss(token_ids)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return optimizer, scheduler


def _save_model_checkpoint(model: Transformer, step: int, output_dir: str) -> str:
    """Save a model checkpoint using the real save_checkpoint helper."""
    opt, sched = _make_training_state(model)
    return save_checkpoint(
        model, opt, sched,
        step=step, loss=1.0,
        config={"model": {"dim": 64}},
        output_dir=output_dir,
    )


def _make_simple_safetensors(weight_value: float, tmp_dir: Path, name: str) -> Path:
    """Create a minimal safetensors checkpoint with known constant weights.

    Used to test averaging arithmetic without needing a real model.
    Creates a ckpt dir with model.safetensors containing two keys:
      - 'tok_emb.weight': shape (4, 4) filled with weight_value
      - 'norm.weight': shape (4,) filled with weight_value
    (output.weight intentionally excluded to match the weight-tying invariant.)
    """
    ckpt_dir = tmp_dir / f"step_{name:0>8}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "tok_emb.weight": torch.full((4, 4), weight_value),
        "norm.weight": torch.full((4,), weight_value),
    }
    save_file(state, str(ckpt_dir / "model.safetensors"))
    return ckpt_dir


# ---------------------------------------------------------------------------
# 1. Uniform average produces correct values
# ---------------------------------------------------------------------------

class TestUniformAverage:

    def test_mean_of_known_weights(self, tmp_path):
        """Average of [1, 2, 3] should be 2.0 for every element."""
        dirs = [
            _make_simple_safetensors(1.0, tmp_path / "ckpts", "00001"),
            _make_simple_safetensors(2.0, tmp_path / "ckpts", "00002"),
            _make_simple_safetensors(3.0, tmp_path / "ckpts", "00003"),
        ]
        averager = CheckpointAverager()
        out_path = str(tmp_path / "averaged")
        averager.uniform_average([str(d) for d in dirs], out_path)

        averaged = load_file(str(Path(out_path) / "model.safetensors"))
        for key in averaged:
            assert torch.allclose(averaged[key], torch.full_like(averaged[key], 2.0)), (
                f"Key {key!r}: expected all 2.0, got {averaged[key]}"
            )

    def test_returns_average_result(self, tmp_path):
        dirs = [
            _make_simple_safetensors(1.0, tmp_path / "ckpts", "00001"),
            _make_simple_safetensors(3.0, tmp_path / "ckpts", "00002"),
        ]
        averager = CheckpointAverager()
        out_path = str(tmp_path / "averaged")
        result = averager.uniform_average([str(d) for d in dirs], out_path)

        assert isinstance(result, AverageResult)
        assert result.method == "uniform"
        assert result.num_checkpoints == 2
        assert len(result.checkpoint_paths) == 2
        assert result.output_path == out_path

    def test_average_of_identical_is_original(self, tmp_path):
        """Averaging N copies of the same weights returns those weights."""
        dirs = [
            _make_simple_safetensors(5.0, tmp_path / "ckpts", "00001"),
            _make_simple_safetensors(5.0, tmp_path / "ckpts", "00002"),
            _make_simple_safetensors(5.0, tmp_path / "ckpts", "00003"),
        ]
        averager = CheckpointAverager()
        out_path = str(tmp_path / "averaged")
        averager.uniform_average([str(d) for d in dirs], out_path)

        averaged = load_file(str(Path(out_path) / "model.safetensors"))
        for key in averaged:
            assert torch.allclose(averaged[key], torch.full_like(averaged[key], 5.0))

    def test_single_checkpoint_is_copy(self, tmp_path):
        """Averaging 1 checkpoint is identical to the original weights."""
        ckpt_dir = _make_simple_safetensors(7.0, tmp_path / "ckpts", "00001")
        averager = CheckpointAverager()
        out_path = str(tmp_path / "averaged")
        result = averager.uniform_average([str(ckpt_dir)], out_path)

        averaged = load_file(str(Path(out_path) / "model.safetensors"))
        for key in averaged:
            assert torch.allclose(averaged[key], torch.full_like(averaged[key], 7.0))
        assert result.num_checkpoints == 1

    def test_output_weight_excluded(self, tmp_path):
        """output.weight must NOT appear in the saved averaged checkpoint."""
        model = _make_model()
        ckpt1 = _save_model_checkpoint(model, step=100, output_dir=str(tmp_path / "run1"))
        ckpt2 = _save_model_checkpoint(model, step=200, output_dir=str(tmp_path / "run2"))

        averager = CheckpointAverager(_tiny_config())
        out_path = str(tmp_path / "averaged")
        averager.uniform_average([ckpt1, ckpt2], out_path)

        saved = load_file(str(Path(out_path) / "model.safetensors"))
        assert "output.weight" not in saved, (
            "output.weight must be excluded — it's tied to tok_emb.weight"
        )
        assert "tok_emb.weight" in saved


# ---------------------------------------------------------------------------
# 2. EMA produces expected decay behavior
# ---------------------------------------------------------------------------

class TestEMAAverage:

    def test_ema_two_checkpoints(self, tmp_path):
        """EMA of [0.0, 1.0] with decay=0.5 should equal 0.5."""
        # ema = decay * first + (1 - decay) * second
        # = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        dirs = [
            _make_simple_safetensors(0.0, tmp_path / "ckpts", "00001"),
            _make_simple_safetensors(1.0, tmp_path / "ckpts", "00002"),
        ]
        averager = CheckpointAverager()
        out_path = str(tmp_path / "averaged")
        result = averager.exponential_moving_average(
            [str(d) for d in dirs], out_path, decay=0.5
        )

        averaged = load_file(str(Path(out_path) / "model.safetensors"))
        for key in averaged:
            assert torch.allclose(
                averaged[key], torch.full_like(averaged[key], 0.5), atol=1e-5
            ), f"Key {key!r}: expected 0.5, got {averaged[key].flatten()[:3]}"
        assert result.method == "ema"

    def test_ema_returns_result(self, tmp_path):
        dirs = [
            _make_simple_safetensors(1.0, tmp_path / "ckpts", "00001"),
            _make_simple_safetensors(2.0, tmp_path / "ckpts", "00002"),
        ]
        averager = CheckpointAverager()
        out_path = str(tmp_path / "averaged")
        result = averager.exponential_moving_average(
            [str(d) for d in dirs], out_path, decay=0.9
        )
        assert result.method == "ema"
        assert result.num_checkpoints == 2

    def test_ema_high_decay_stays_near_first(self, tmp_path):
        """With decay very close to 1, EMA should remain near the first checkpoint."""
        # decay=0.999: ema = 0.999 * 0.0 + 0.001 * 100 = 0.1
        dirs = [
            _make_simple_safetensors(0.0, tmp_path / "ckpts", "00001"),
            _make_simple_safetensors(100.0, tmp_path / "ckpts", "00002"),
        ]
        averager = CheckpointAverager()
        out_path = str(tmp_path / "averaged")
        averager.exponential_moving_average(
            [str(d) for d in dirs], out_path, decay=0.999
        )
        averaged = load_file(str(Path(out_path) / "model.safetensors"))
        for key in averaged:
            # Should be close to 0.1, definitely much less than 50 (the midpoint)
            assert averaged[key].mean().item() < 1.0, (
                f"High-decay EMA should stay near first checkpoint, got {averaged[key].mean()}"
            )

    def test_ema_invalid_decay_raises(self, tmp_path):
        """decay outside (0, 1) must raise ValueError."""
        dirs = [_make_simple_safetensors(1.0, tmp_path / "ckpts", "00001")]
        averager = CheckpointAverager()
        out_path = str(tmp_path / "averaged")

        with pytest.raises(ValueError, match="decay"):
            averager.exponential_moving_average([str(dirs[0])], out_path, decay=0.0)

        with pytest.raises(ValueError, match="decay"):
            averager.exponential_moving_average([str(dirs[0])], out_path, decay=1.0)

        with pytest.raises(ValueError, match="decay"):
            averager.exponential_moving_average([str(dirs[0])], out_path, decay=1.5)


# ---------------------------------------------------------------------------
# 3. find_checkpoints sorts correctly
# ---------------------------------------------------------------------------

class TestFindCheckpoints:

    def test_sorts_by_step_number(self, tmp_path):
        """Checkpoints must be returned in ascending step order."""
        # Create dirs out of order to verify sorting is by number, not name
        for step_str in ["00020", "00005", "00010", "00015"]:
            d = tmp_path / f"step_{step_str}"
            d.mkdir()
            save_file({"tok_emb.weight": torch.zeros(4, 4)}, str(d / "model.safetensors"))

        found = CheckpointAverager.find_checkpoints(str(tmp_path))
        names = [Path(p).name for p in found]
        assert names == ["step_00005", "step_00010", "step_00015", "step_00020"]

    def test_excludes_dirs_without_safetensors(self, tmp_path):
        """Directories without model.safetensors must be excluded."""
        # Valid checkpoint
        d1 = tmp_path / "step_00001"
        d1.mkdir()
        save_file({"tok_emb.weight": torch.zeros(4, 4)}, str(d1 / "model.safetensors"))

        # Directory without model.safetensors (e.g. incomplete save)
        d2 = tmp_path / "step_00002"
        d2.mkdir()
        (d2 / "metadata.json").write_text('{"step": 2}')

        found = CheckpointAverager.find_checkpoints(str(tmp_path))
        assert len(found) == 1
        assert Path(found[0]).name == "step_00001"

    def test_empty_directory(self, tmp_path):
        """find_checkpoints on an empty directory returns []."""
        assert CheckpointAverager.find_checkpoints(str(tmp_path)) == []

    def test_nonexistent_directory(self, tmp_path):
        """find_checkpoints on a missing directory returns []."""
        assert CheckpointAverager.find_checkpoints(str(tmp_path / "does_not_exist")) == []


# ---------------------------------------------------------------------------
# 4. average_last_k picks correct checkpoints
# ---------------------------------------------------------------------------

class TestAverageLastK:

    def test_picks_last_k(self, tmp_path):
        """average_last_k must select the K most recent checkpoints."""
        # Create 5 checkpoints with value = step / 10000
        for step in [1, 2, 3, 4, 5]:
            d = tmp_path / f"step_0000{step}"
            d.mkdir()
            save_file(
                {"tok_emb.weight": torch.full((4, 4), float(step))},
                str(d / "model.safetensors"),
            )

        averager = CheckpointAverager()
        out_path = str(tmp_path / "averaged")
        result = averager.average_last_k(str(tmp_path), k=3, output_path=out_path)

        # Last 3 checkpoints have values 3, 4, 5 — mean = 4.0
        averaged = load_file(str(Path(out_path) / "model.safetensors"))
        for key in averaged:
            assert torch.allclose(
                averaged[key], torch.full_like(averaged[key], 4.0), atol=1e-5
            )
        assert result.num_checkpoints == 3

    def test_default_output_path(self, tmp_path):
        """When output_path is None, defaults to <checkpoint_dir>/averaged_last_<k>."""
        for step in [1, 2, 3]:
            d = tmp_path / f"step_0000{step}"
            d.mkdir()
            save_file({"tok_emb.weight": torch.zeros(4, 4)}, str(d / "model.safetensors"))

        averager = CheckpointAverager()
        result = averager.average_last_k(str(tmp_path), k=2, output_path=None)

        expected = str(tmp_path / "averaged_last_2")
        assert result.output_path == expected
        assert (Path(expected) / "model.safetensors").exists()

    def test_k_larger_than_available_raises(self, tmp_path):
        """Requesting k > available checkpoints must raise ValueError."""
        for step in [1, 2]:
            d = tmp_path / f"step_0000{step}"
            d.mkdir()
            save_file({"tok_emb.weight": torch.zeros(4, 4)}, str(d / "model.safetensors"))

        averager = CheckpointAverager()
        with pytest.raises(ValueError, match="only 2 found"):
            averager.average_last_k(str(tmp_path), k=5)

    def test_ema_method_dispatch(self, tmp_path):
        """average_last_k with method='ema' calls EMA averaging."""
        for step in [1, 2, 3]:
            d = tmp_path / f"step_0000{step}"
            d.mkdir()
            save_file({"tok_emb.weight": torch.zeros(4, 4)}, str(d / "model.safetensors"))

        averager = CheckpointAverager()
        out_path = str(tmp_path / "averaged_ema")
        result = averager.average_last_k(
            str(tmp_path), k=3, output_path=out_path, method="ema"
        )
        assert result.method == "ema"

    def test_unknown_method_raises(self, tmp_path):
        """average_last_k with an unknown method raises ValueError."""
        d = tmp_path / "step_00001"
        d.mkdir()
        save_file({"tok_emb.weight": torch.zeros(4, 4)}, str(d / "model.safetensors"))

        averager = CheckpointAverager()
        with pytest.raises(ValueError, match="Unknown method"):
            averager.average_last_k(str(tmp_path), k=1, method="bad_method")


# ---------------------------------------------------------------------------
# 5. Error cases
# ---------------------------------------------------------------------------

class TestErrorCases:

    def test_missing_checkpoint_raises(self, tmp_path):
        """Passing a path that doesn't exist raises FileNotFoundError."""
        averager = CheckpointAverager()
        with pytest.raises(FileNotFoundError):
            averager.uniform_average(
                [str(tmp_path / "does_not_exist")],
                str(tmp_path / "out"),
            )

    def test_checkpoint_missing_model_safetensors_raises(self, tmp_path):
        """A directory without model.safetensors raises FileNotFoundError."""
        bad_dir = tmp_path / "step_00001"
        bad_dir.mkdir()
        (bad_dir / "metadata.json").write_text('{"step": 1}')

        averager = CheckpointAverager()
        with pytest.raises(FileNotFoundError, match="model.safetensors"):
            averager.uniform_average([str(bad_dir)], str(tmp_path / "out"))

    def test_mismatched_keys_raises(self, tmp_path):
        """Checkpoints with different keys must raise ValueError."""
        d1 = tmp_path / "step_00001"
        d1.mkdir()
        save_file({"tok_emb.weight": torch.zeros(4, 4)}, str(d1 / "model.safetensors"))

        d2 = tmp_path / "step_00002"
        d2.mkdir()
        save_file(
            {"tok_emb.weight": torch.zeros(4, 4), "extra.weight": torch.zeros(4)},
            str(d2 / "model.safetensors"),
        )

        averager = CheckpointAverager()
        with pytest.raises(ValueError, match="key mismatch"):
            averager.uniform_average([str(d1), str(d2)], str(tmp_path / "out"))

    def test_empty_checkpoint_list_raises(self, tmp_path):
        """Passing an empty list raises ValueError."""
        averager = CheckpointAverager()
        with pytest.raises(ValueError, match="must not be empty"):
            averager.uniform_average([], str(tmp_path / "out"))


# ---------------------------------------------------------------------------
# 6. Integration: averaged checkpoint loads into a real model
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_averaged_checkpoint_loads_into_model(self, tmp_path):
        """Averaged checkpoint must load successfully into a Transformer model."""
        model1 = _make_model()
        model2 = _make_model()

        ckpt1 = _save_model_checkpoint(model1, step=100, output_dir=str(tmp_path / "run1"))
        ckpt2 = _save_model_checkpoint(model2, step=200, output_dir=str(tmp_path / "run2"))

        averager = CheckpointAverager(_tiny_config())
        out_path = str(tmp_path / "averaged")
        averager.uniform_average([ckpt1, ckpt2], out_path)

        # Load into a fresh model — must not raise
        from cola_coder.training.checkpoint import load_model_only
        fresh = _make_model()
        load_model_only(out_path, fresh, device="cpu")

        # Weight tying must still hold after load
        assert fresh.output.weight.data_ptr() == fresh.tok_emb.weight.data_ptr(), (
            "Weight tying broken after loading averaged checkpoint"
        )

    def test_averaged_model_forward_pass(self, tmp_path):
        """Averaged model must produce valid (non-NaN) logits."""
        from cola_coder.training.checkpoint import load_model_only

        model1 = _make_model()
        model2 = _make_model()

        ckpt1 = _save_model_checkpoint(model1, step=100, output_dir=str(tmp_path / "run1"))
        ckpt2 = _save_model_checkpoint(model2, step=200, output_dir=str(tmp_path / "run2"))

        averager = CheckpointAverager(_tiny_config())
        out_path = str(tmp_path / "averaged")
        averager.uniform_average([ckpt1, ckpt2], out_path)

        fresh = _make_model()
        load_model_only(out_path, fresh, device="cpu")

        token_ids = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            logits = fresh(token_ids)

        assert logits.shape == (1, 16, 256)
        assert not torch.isnan(logits).any(), "Averaged model produces NaN logits"
