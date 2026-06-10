"""Tests for the 2025-26 modernization package.

Covers: min-p sampling, QK-Norm, residual-scaled init, z-loss,
Muon optimizer (+ embedded AdamW), WSD schedule, and Dr. GRPO /
DAPO clip-higher advantage handling.
"""

from __future__ import annotations

import pytest
import torch

from cola_coder.model.config import ModelConfig
from cola_coder.model.transformer import Transformer, language_modeling_loss


def _tiny_config(**overrides) -> ModelConfig:
    defaults = dict(
        vocab_size=64, dim=32, n_layers=2, n_heads=2, n_kv_heads=1,
        max_seq_len=32, dropout=0.0,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


# ── Min-p sampling ──────────────────────────────────────────────────────────

class TestMinPSampling:
    def test_min_p_filters_low_prob_tokens(self):
        from cola_coder.inference.sampling import _min_p_filter

        # Token 0 dominates; token 3 is far below 10% of its probability
        logits = torch.tensor([10.0, 9.5, 9.0, 0.0])
        filtered = _min_p_filter(logits.clone(), min_p=0.1)
        assert filtered[3] == float("-inf")
        assert torch.isfinite(filtered[0])
        assert torch.isfinite(filtered[1])

    def test_min_p_keeps_diversity_when_uncertain(self):
        from cola_coder.inference.sampling import _min_p_filter

        # Near-uniform distribution: nothing falls below 10% of max prob
        logits = torch.tensor([1.0, 0.9, 1.1, 0.95])
        filtered = _min_p_filter(logits.clone(), min_p=0.1)
        assert torch.isfinite(filtered).all()

    def test_min_p_batch_per_row_thresholds(self):
        from cola_coder.inference.sampling import _min_p_filter_batch

        logits = torch.tensor([
            [10.0, 0.0, 0.0],   # confident row → prune tails
            [1.0, 1.0, 1.0],    # uniform row → keep all
        ])
        filtered = _min_p_filter_batch(logits.clone(), min_p=0.5)
        assert filtered[0, 1] == float("-inf")
        assert filtered[0, 2] == float("-inf")
        assert torch.isfinite(filtered[1]).all()

    def test_sample_next_token_accepts_min_p(self):
        from cola_coder.inference.sampling import sample_next_token

        torch.manual_seed(0)
        logits = torch.randn(64)
        token = sample_next_token(logits, temperature=1.0, min_p=0.1)
        assert 0 <= token < 64

    def test_min_p_zero_is_noop(self):
        from cola_coder.inference.sampling import _min_p_filter

        logits = torch.randn(16)
        # min_p=0 path is gated by callers; the filter itself with tiny
        # min_p must keep everything finite
        filtered = _min_p_filter(logits.clone(), min_p=1e-9)
        assert torch.isfinite(filtered).all()


# ── QK-Norm ─────────────────────────────────────────────────────────────────

class TestQKNorm:
    def test_disabled_by_default(self):
        model = Transformer(_tiny_config())
        assert model.blocks[0].attention.q_norm is None

    def test_enabled_adds_norms_and_forward_works(self):
        model = Transformer(_tiny_config(qk_norm=True))
        attn = model.blocks[0].attention
        assert attn.q_norm is not None and attn.k_norm is not None
        x = torch.randint(0, 64, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 64)

    def test_checkpoint_roundtrip_with_qk_norm(self, tmp_path):
        from cola_coder.training.checkpoint import load_model_only, save_checkpoint
        from cola_coder.training.optimizer import create_optimizer, create_scheduler

        config = _tiny_config(qk_norm=True)
        model = Transformer(config)
        opt = create_optimizer(model)
        sched = create_scheduler(opt, warmup_steps=10, max_steps=100)
        save_checkpoint(
            model=model, optimizer=opt, scheduler=sched, step=1, loss=1.0,
            config={"model": vars(config)}, output_dir=str(tmp_path),
        )
        restored = Transformer(_tiny_config(qk_norm=True))
        load_model_only(str(tmp_path / "step_00000001"), restored, device="cpu")
        x = torch.randint(0, 64, (1, 8))
        torch.testing.assert_close(model(x), restored(x))

    def test_kv_cache_inference_with_qk_norm(self):
        model = Transformer(_tiny_config(qk_norm=True))
        model.eval()
        x = torch.randint(0, 64, (1, 8))
        with torch.no_grad():
            full = model(x, use_cache=False)
            model.clear_caches()
            cached = model(x, start_pos=0, use_cache=True)
            model.clear_caches()
        torch.testing.assert_close(
            full[:, -1, :], cached[:, -1, :], rtol=1e-4, atol=1e-4,
        )


# ── Residual-scaled init ────────────────────────────────────────────────────

class TestResidualScaledInit:
    def test_residual_projections_have_smaller_std(self):
        torch.manual_seed(0)
        model = Transformer(_tiny_config(n_layers=8, dim=64, n_heads=4, n_kv_heads=2))
        out_proj_std = model.blocks[0].attention.out_proj.weight.std().item()
        q_proj_std = model.blocks[0].attention.q_proj.weight.std().item()
        # Residual writers: 0.02/sqrt(16) = 0.005; others stay at 0.02
        assert out_proj_std < 0.01, f"out_proj std {out_proj_std} not residual-scaled"
        assert q_proj_std > 0.015, f"q_proj std {q_proj_std} unexpectedly scaled"

    def test_ffn_down_proj_scaled(self):
        torch.manual_seed(0)
        model = Transformer(_tiny_config(n_layers=8, dim=64, n_heads=4, n_kv_heads=2))
        down_std = model.blocks[0].ffn.down_proj.weight.std().item()
        up_std = model.blocks[0].ffn.up_proj.weight.std().item()
        assert down_std < 0.01
        assert up_std > 0.015


# ── Z-loss ──────────────────────────────────────────────────────────────────

class TestZLoss:
    def test_z_loss_increases_total_loss(self):
        torch.manual_seed(0)
        model = Transformer(_tiny_config())
        x = torch.randint(0, 64, (2, 16))
        logits = model(x)
        plain = language_modeling_loss(logits, x)
        with_z = language_modeling_loss(logits, x, z_loss=1e-2)
        assert with_z > plain

    def test_z_loss_gradient_flows(self):
        model = Transformer(_tiny_config())
        x = torch.randint(0, 64, (2, 16))
        loss = language_modeling_loss(model(x), x, z_loss=1e-4)
        loss.backward()
        assert model.tok_emb.weight.grad is not None

    def test_z_loss_works_with_sample_weights(self):
        model = Transformer(_tiny_config())
        x = torch.randint(0, 64, (2, 16))
        loss = language_modeling_loss(
            model(x), x, sample_weights=torch.tensor([1.0, 2.0]), z_loss=1e-4,
        )
        assert torch.isfinite(loss)


# ── Muon optimizer ──────────────────────────────────────────────────────────

class TestMuon:
    def test_newton_schulz_orthogonalizes(self):
        from cola_coder.training.optimizer import _zeropower_via_newtonschulz5

        torch.manual_seed(0)
        G = torch.randn(16, 32)
        ortho = _zeropower_via_newtonschulz5(G, steps=5).float()
        # Singular values should be ~1 (flat spectrum)
        sv = torch.linalg.svdvals(ortho)
        assert sv.max() < 1.6 and sv.min() > 0.4, f"spectrum not flattened: {sv}"

    def test_muon_training_decreases_loss(self):
        from cola_coder.training.optimizer import create_optimizer

        torch.manual_seed(0)
        model = Transformer(_tiny_config())
        opt = create_optimizer(model, learning_rate=1e-3, optimizer="muon", muon_lr=0.01)
        x = torch.randint(0, 64, (4, 16))

        initial = model.compute_loss(x).item()
        for _ in range(15):
            opt.zero_grad()
            loss = model.compute_loss(x)
            loss.backward()
            opt.step()
        assert model.compute_loss(x).item() < initial

    def test_muon_param_split(self):
        from cola_coder.training.optimizer import Muon, create_optimizer

        model = Transformer(_tiny_config())
        opt = create_optimizer(model, optimizer="muon")
        assert isinstance(opt, Muon)
        muon_group = next(g for g in opt.param_groups if g["use_muon"])
        adamw_groups = [g for g in opt.param_groups if not g["use_muon"]]
        # Tied embedding must NOT be in the Muon group
        emb = model.tok_emb.weight
        assert all(p is not emb for p in muon_group["params"])
        assert any(p is emb for g in adamw_groups for p in g["params"])
        # All Muon params are 2D block weights
        assert all(p.dim() == 2 for p in muon_group["params"])

    def test_muon_state_dict_roundtrip(self):
        from cola_coder.training.optimizer import create_optimizer

        torch.manual_seed(0)
        model = Transformer(_tiny_config())
        opt = create_optimizer(model, optimizer="muon")
        x = torch.randint(0, 64, (2, 16))
        model.compute_loss(x).backward()
        opt.step()

        state = opt.state_dict()
        opt2 = create_optimizer(model, optimizer="muon")
        opt2.load_state_dict(state)  # must not raise

    def test_scheduler_scales_both_optimizer_sides(self):
        from cola_coder.training.optimizer import create_optimizer, create_scheduler

        model = Transformer(_tiny_config())
        opt = create_optimizer(model, learning_rate=1e-3, optimizer="muon", muon_lr=0.02)
        create_scheduler(opt, warmup_steps=100, max_steps=1000)
        # Warmup factor applies multiplicatively to every group's base lr
        lrs = [g["lr"] for g in opt.param_groups]
        assert lrs[0] == pytest.approx(0.02 / 100)
        assert lrs[1] == pytest.approx(1e-3 / 100)


# ── WSD schedule ────────────────────────────────────────────────────────────

class TestWSDSchedule:
    def _factors(self, schedule: str) -> list[float]:
        from cola_coder.training.optimizer import create_optimizer, create_scheduler

        model = Transformer(_tiny_config())
        opt = create_optimizer(model, learning_rate=1.0)
        sched = create_scheduler(
            opt, warmup_steps=10, max_steps=100, min_lr_ratio=0.1, schedule=schedule,
        )
        factors = []
        for _ in range(100):
            factors.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        return factors

    def test_wsd_has_stable_plateau(self):
        factors = self._factors("wsd")
        # After warmup (10) and before decay (80), LR holds at peak
        assert factors[20] == pytest.approx(1.0)
        assert factors[70] == pytest.approx(1.0)
        # Decays at the end
        assert factors[99] < 0.2

    def test_cosine_decays_through_middle(self):
        factors = self._factors("cosine")
        assert factors[50] < 0.95  # cosine is already decaying mid-run


# ── GRPO modernization ──────────────────────────────────────────────────────

class TestGroupAdvantages:
    def test_std_norm_default(self):
        from cola_coder.reasoning.grpo import compute_group_advantages

        rewards = torch.tensor([0.0, 1.0, 0.0, 1.0])
        adv = compute_group_advantages(rewards)
        assert adv.mean().abs() < 1e-6
        assert adv.std() == pytest.approx(1.0, rel=0.2)

    def test_dr_grpo_mean_only(self):
        from cola_coder.reasoning.grpo import compute_group_advantages

        rewards = torch.tensor([0.0, 1.0, 0.0, 1.0])
        adv = compute_group_advantages(rewards, norm="mean")
        torch.testing.assert_close(adv, rewards - 0.5)

    def test_dr_grpo_does_not_inflate_low_variance_groups(self):
        from cola_coder.reasoning.grpo import compute_group_advantages

        # One outlier in an otherwise-uniform group: std-norm inflates the
        # update; mean-norm keeps it proportional to the actual reward gap
        rewards = torch.tensor([0.9, 1.0, 1.0, 1.0])
        std_adv = compute_group_advantages(rewards, norm="std")
        mean_adv = compute_group_advantages(rewards, norm="mean")
        assert std_adv.abs().max() > mean_adv.abs().max()
