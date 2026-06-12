"""MODEL-007: GRPO must use a PER-TOKEN clipped surrogate, and the PPO clip must
actually be reachable.

Two prior problems:
1. The importance ratio was SEQUENCE-LEVEL: exp(Σ_t Δlogp) = the product of the
   per-token ratios. Over long completions this explodes/vanishes and saturates
   the clip on nearly every sample. Reference GRPO/Dr.GRPO/DAPO clip PER TOKEN.
2. The trainer took ONE gradient step per generated group, so the new policy
   equaled the old policy and the ratio was always exactly 1 → clip_epsilon /
   clip_epsilon_high (DAPO clip-higher) never engaged. ppo_epochs > 1 fixes that.

These tests lock the pure surrogate's math (grpo_clipped_surrogate) and the
per-token completion-logprob slice.
"""

import torch

from cola_coder.reasoning.grpo import _completion_logprobs, grpo_clipped_surrogate


class TestCompletionLogprobsVector:
    def test_returns_completion_slice(self):
        lp = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        # prompt_len=3 → completion scored by indices >= 2.
        out = _completion_logprobs(lp, prompt_len=3)
        assert torch.equal(out, torch.tensor([3.0, 4.0, 5.0]))

    def test_empty_when_no_completion(self):
        lp = torch.tensor([1.0, 2.0, 3.0])
        out = _completion_logprobs(lp, prompt_len=6)
        assert out.numel() == 0 and out.shape == (0,)

    def test_sum_matches_legacy_helper(self):
        from cola_coder.reasoning.grpo import _completion_logprob_sum
        lp = torch.tensor([0.5, -1.0, 2.0, -0.5, 3.0])
        assert _completion_logprobs(lp, 3).sum() == _completion_logprob_sum(lp, 3)


class TestGrpoClippedSurrogate:
    def test_ratio_one_returns_sum_of_advantage(self):
        # new == old → every per-token ratio is 1 → surrogate = Σ_t A = T·A.
        new = torch.tensor([-1.0, -2.0, -0.5])
        old = new.clone()
        surr = grpo_clipped_surrogate(new, old, advantage=2.0, clip_low=0.2, clip_high=0.28)
        assert surr.item() == 6.0  # 3 tokens * 2.0

    def test_empty_completion_is_zero(self):
        empty = torch.zeros((0,))
        surr = grpo_clipped_surrogate(empty, empty, advantage=5.0,
                                      clip_low=0.2, clip_high=0.28)
        assert surr.item() == 0.0

    def test_positive_advantage_clipped_at_upper_bound(self):
        # ratio = exp(0.5) ≈ 1.6487 > 1+clip_high=1.28 → clipped to 1.28.
        new = torch.tensor([0.5])
        old = torch.tensor([0.0])
        surr = grpo_clipped_surrogate(new, old, advantage=1.0, clip_low=0.2, clip_high=0.28)
        # min(1.6487*1, 1.28*1) = 1.28
        assert abs(surr.item() - 1.28) < 1e-5

    def test_clip_higher_is_asymmetric(self):
        # Same positive ratio, looser upper bound (DAPO) → larger surrogate than
        # symmetric clipping would give.
        new, old = torch.tensor([0.5]), torch.tensor([0.0])
        loose = grpo_clipped_surrogate(new, old, 1.0, clip_low=0.2, clip_high=0.28)
        tight = grpo_clipped_surrogate(new, old, 1.0, clip_low=0.2, clip_high=0.2)
        assert loose.item() > tight.item()  # 1.28 > 1.20

    def test_negative_advantage_takes_more_negative_branch(self):
        # ratio ≈ 1.6487; A = -1. unclipped = -1.6487, clipped = -1.28.
        # min(-1.6487, -1.28) = -1.6487 (PPO keeps the pessimistic branch).
        new, old = torch.tensor([0.5]), torch.tensor([0.0])
        surr = grpo_clipped_surrogate(new, old, advantage=-1.0, clip_low=0.2, clip_high=0.28)
        assert abs(surr.item() - (-torch.exp(torch.tensor(0.5)).item())) < 1e-5

    def test_lower_clip_engages_for_negative_advantage(self):
        # PPO asymmetry: the LOWER clip only bites for NEGATIVE advantage.
        # ratio = exp(-0.5) ≈ 0.6065 < 1-clip_low=0.8, A=-1:
        # unclipped=-0.6065, clipped=0.8*-1=-0.8, min → -0.8 (clip engaged).
        new, old = torch.tensor([-0.5]), torch.tensor([0.0])
        surr = grpo_clipped_surrogate(new, old, advantage=-1.0, clip_low=0.2, clip_high=0.28)
        assert abs(surr.item() - (-0.8)) < 1e-5

    def test_positive_advantage_dropped_ratio_is_unclipped(self):
        # ratio = exp(-0.5) ≈ 0.6065, A=+1: min(0.6065, clip→0.8) = 0.6065.
        # The lower clip does NOT bite on the upside-positive branch (correct PPO).
        new, old = torch.tensor([-0.5]), torch.tensor([0.0])
        surr = grpo_clipped_surrogate(new, old, advantage=1.0, clip_low=0.2, clip_high=0.28)
        assert abs(surr.item() - torch.exp(torch.tensor(-0.5)).item()) < 1e-5

    def test_per_token_not_sequence_level(self):
        # Two tokens with opposite deviations. Sequence-level would use ONE ratio
        # exp(0.5-0.5)=1 → surrogate 2.0. Per-token clips each leg independently:
        # token A ratio exp(0.5)→clip 1.28; token B ratio exp(-0.5)=0.6065 unclipped.
        new = torch.tensor([0.5, -0.5])
        old = torch.tensor([0.0, 0.0])
        surr = grpo_clipped_surrogate(new, old, advantage=1.0, clip_low=0.2, clip_high=0.28)
        expected = 1.28 + torch.exp(torch.tensor(-0.5)).item()  # ≈ 1.8865
        assert abs(surr.item() - expected) < 1e-5
        assert abs(surr.item() - 2.0) > 0.1  # genuinely NOT the sequence-level value


class TestLengthNorm:
    """MODEL-008: optional Dr. GRPO constant length normalization."""

    def test_none_is_plain_sum(self):
        new = torch.tensor([-1.0, -2.0, -0.5])
        old = new.clone()
        assert grpo_clipped_surrogate(new, old, 2.0, 0.2, 0.28, length_norm=None).item() == 6.0

    def test_constant_divides_the_sum(self):
        # Same input, divided by L=3 → mean per token = 2.0.
        new = torch.tensor([-1.0, -2.0, -0.5])
        old = new.clone()
        out = grpo_clipped_surrogate(new, old, 2.0, 0.2, 0.28, length_norm=3.0)
        assert abs(out.item() - 2.0) < 1e-6

    def test_zero_length_norm_falls_back_to_sum(self):
        # length_norm=0 is falsy → no division (avoids a div-by-zero footgun).
        new = torch.tensor([-1.0, -2.0])
        old = new.clone()
        assert grpo_clipped_surrogate(new, old, 1.0, 0.2, 0.28, length_norm=0).item() == 2.0

    def test_length_norm_is_uniform_scaling_of_sum(self):
        # Division is applied AFTER the per-token clip, so it's just a constant
        # rescale of the summed surrogate (gradient direction unchanged).
        new = torch.tensor([0.5, -0.5, 0.1])
        old = torch.tensor([0.0, 0.0, 0.0])
        s = grpo_clipped_surrogate(new, old, 1.0, 0.2, 0.28, length_norm=None)
        n = grpo_clipped_surrogate(new, old, 1.0, 0.2, 0.28, length_norm=4.0)
        assert abs(n.item() - s.item() / 4.0) < 1e-6


def _tiny_trainer(length_norm: str, max_new_tokens: int = 384):
    """Construct a real GRPOTrainer (no forward pass) to test divisor resolution."""
    from unittest.mock import MagicMock

    from cola_coder.model.config import ModelConfig
    from cola_coder.model.transformer import Transformer
    from cola_coder.reasoning.grpo import GRPOTrainer

    model = Transformer(ModelConfig(
        vocab_size=64, dim=32, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=32,
    ))
    return GRPOTrainer(
        model=model, tokenizer=MagicMock(), device="cpu",
        length_norm=length_norm, max_new_tokens=max_new_tokens,
    )


class TestTrainerLengthNormWiring:
    """The real constructor resolves length_norm → the surrogate divisor."""

    def test_sum_mode_divisor_is_none(self):
        assert _tiny_trainer("sum")._loss_length_divisor is None

    def test_constant_mode_divisor_is_max_new_tokens(self):
        assert _tiny_trainer("constant", max_new_tokens=384)._loss_length_divisor == 384.0
