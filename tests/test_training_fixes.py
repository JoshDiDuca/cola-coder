"""Regression tests for training-loop bugs found in the 2026-06-10 audit.

Each test class covers one verified bug:
- Quality-weighted loss was `mean_loss * weights.mean()` (batch-level rescale,
  no within-batch differentiation)
- Warmup LR was 0 at step 0 (first optimizer step wasted)
- detect_latest_checkpoint trusted stale `latest` pointers over step_* dirs
- load_state_dict(strict=False) silently ignored every key mismatch
- torch.compile was never exercised (trainer called .compute_loss, which
  bypasses OptimizedModule.__call__)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cola_coder.model.config import ModelConfig
from cola_coder.model.transformer import Transformer, language_modeling_loss


def _tiny_model() -> Transformer:
    torch.manual_seed(0)
    return Transformer(ModelConfig(
        vocab_size=64, dim=32, n_layers=1, n_heads=2, n_kv_heads=1,
        max_seq_len=32, dropout=0.0,
    ))


class TestWeightedLoss:
    def test_uniform_weights_match_unweighted(self):
        model = _tiny_model()
        x = torch.randint(0, 64, (4, 16))
        logits = model(x)
        plain = language_modeling_loss(logits, x)
        weighted = language_modeling_loss(logits, x, torch.ones(4))
        torch.testing.assert_close(plain, weighted)

    def test_weights_differentiate_within_batch(self):
        """Up-weighting one sample must pull the batch loss toward its loss."""
        model = _tiny_model()
        x = torch.randint(0, 64, (2, 16))
        logits = model(x)

        # Per-sample losses
        loss_0 = language_modeling_loss(logits[:1], x[:1])
        loss_1 = language_modeling_loss(logits[1:], x[1:])

        # Heavily favor sample 0 → batch loss approaches loss_0
        favored = language_modeling_loss(logits, x, torch.tensor([100.0, 0.001]))
        assert abs(favored.item() - loss_0.item()) < abs(favored.item() - loss_1.item())

        # The old mean*mean formula could never do this: it always returned
        # mean_loss * weights.mean(), identical ORDERING regardless of which
        # sample carries the weight.
        flipped = language_modeling_loss(logits, x, torch.tensor([0.001, 100.0]))
        assert not torch.allclose(favored, flipped)

    def test_compute_loss_accepts_weights(self):
        model = _tiny_model()
        x = torch.randint(0, 64, (2, 16))
        loss = model.compute_loss(x, sample_weights=torch.tensor([1.0, 2.0]))
        assert loss.shape == ()
        assert torch.isfinite(loss)


class TestWarmupSchedule:
    def test_first_step_lr_nonzero(self):
        from cola_coder.training.optimizer import create_optimizer, create_scheduler

        model = _tiny_model()
        opt = create_optimizer(model, learning_rate=1e-3)
        create_scheduler(opt, warmup_steps=100, max_steps=1000)
        # LambdaLR applies the factor for epoch 0 at construction
        assert opt.param_groups[0]["lr"] > 0, "First optimizer step would run at LR=0"

    def test_warmup_reaches_peak(self):
        from cola_coder.training.optimizer import create_optimizer, create_scheduler

        model = _tiny_model()
        opt = create_optimizer(model, learning_rate=1e-3)
        sched = create_scheduler(opt, warmup_steps=10, max_steps=100)
        for _ in range(10):
            opt.step()
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-3, rel=1e-6)


class TestDetectLatestCheckpointStalePointer:
    def _make_checkpoint(self, d: Path, step: int) -> Path:
        ckpt = d / f"step_{step:08d}"
        ckpt.mkdir(parents=True)
        (ckpt / "metadata.json").write_text(json.dumps({
            "step": step, "loss": 1.0,
            "config": {"model": {"dim": 32, "n_layers": 1, "n_heads": 2,
                                 "n_kv_heads": 1, "vocab_size": 64}},
        }))
        return ckpt

    def test_stale_pointer_ignored_when_step_dirs_exist(self, tmp_path: Path):
        from cola_coder.training.checkpoint import detect_latest_checkpoint

        size_dir = tmp_path / "tiny"
        real = self._make_checkpoint(size_dir, 5000)
        # Stale pointer to a directory that no longer exists
        (size_dir / "latest").write_text(str(size_dir / "step_99999999"))

        result = detect_latest_checkpoint(str(tmp_path))
        assert result is not None
        path, info = result
        assert path == str(real)
        assert info["step"] == 5000

    def test_pointer_used_when_no_step_dirs(self, tmp_path: Path):
        from cola_coder.training.checkpoint import detect_latest_checkpoint

        # Checkpoint lives OUTSIDE the size dir; only the pointer knows it
        external = self._make_checkpoint(tmp_path / "elsewhere", 700)
        size_dir = tmp_path / "ckpts" / "tiny"
        size_dir.mkdir(parents=True)
        (size_dir / "latest").write_text(str(external))

        result = detect_latest_checkpoint(str(tmp_path / "ckpts"))
        assert result is not None
        _, info = result
        assert info["step"] == 700


class TestStrictTiedLoad:
    def test_truncated_state_dict_raises(self):
        from cola_coder.training.checkpoint import _load_state_dict_tied

        model = _tiny_model()
        state = {k: v for k, v in model.state_dict().items()
                 if k != "output.weight"}
        # Remove a real weight — must NOT load silently
        first_block_key = next(k for k in state if k.startswith("blocks.0"))
        del state[first_block_key]
        with pytest.raises(RuntimeError, match="does not match"):
            _load_state_dict_tied(model, state)

    def test_unexpected_key_raises(self):
        from cola_coder.training.checkpoint import _load_state_dict_tied

        model = _tiny_model()
        state = {k: v for k, v in model.state_dict().items()
                 if k != "output.weight"}
        state["bogus.weight"] = torch.zeros(1)
        with pytest.raises(RuntimeError, match="does not match"):
            _load_state_dict_tied(model, state)

    def test_tied_output_weight_missing_is_ok(self):
        from cola_coder.training.checkpoint import _load_state_dict_tied

        model = _tiny_model()
        state = {k: v for k, v in model.state_dict().items()
                 if k != "output.weight"}
        _load_state_dict_tied(model, state)  # must not raise


class TestCurriculumShuffleGuard:
    """Regression: the trainer hardcoded shuffle=True, silently undoing
    curriculum ordering (easy→hard) produced by score_data.py --curriculum."""

    def test_shuffles_plain_data(self, tmp_path):
        from cola_coder.training.trainer import _should_shuffle

        data = tmp_path / "train_data.npy"
        data.write_bytes(b"")
        assert _should_shuffle(str(data)) is True

    def test_preserves_curriculum_order(self, tmp_path):
        from cola_coder.training.trainer import _should_shuffle

        data = tmp_path / "train_data.npy"
        data.write_bytes(b"")
        (tmp_path / "train_data.curriculum.json").write_text("{}")
        assert _should_shuffle(str(data)) is False


class TestCompiledPathUsed:
    def test_direct_call_compiles_compute_loss_does_not(self):
        """Documents WHY the trainer calls model(x) + language_modeling_loss:
        method calls on an OptimizedModule bypass the compiled graph."""
        compilations = 0

        def counting_backend(gm, example_inputs):
            nonlocal compilations
            compilations += 1
            return gm.forward

        model = _tiny_model()
        compiled = torch.compile(model, backend=counting_backend)
        x = torch.randint(0, 64, (1, 8))

        compiled.compute_loss(x)
        assert compilations == 0, "compute_loss unexpectedly went through dynamo"

        compiled(x)
        assert compilations >= 1, "direct call did not go through dynamo"
