"""BUG-112: MoE capacity dropping must not silence routed experts at inference.

`MoEFFN.forward` applied capacity-based token dropping unconditionally with
`capacity = int(capacity_factor * num_tokens / num_experts)`. During single-token
autoregressive decode num_tokens == 1, so capacity rounded to 0 and
`token_indices[:0]` dropped EVERY routed-expert contribution — the MoE collapsed
to just its shared expert(s) at generation time. The formula also omitted top_k,
so it over-dropped even in training.

Fix: capacity dropping is gated to training and uses the top_k-aware formula. At
inference (model.eval()) every token is processed.

These tests use num_shared_experts=0 so the output is PURELY the routed-expert
contribution — making the collapse directly observable (it would be exactly 0).
"""

import torch

from cola_coder.features.moe_layer import MoEFFN


def _moe(num_experts=8, top_k=2, num_shared_experts=0, capacity_factor=1.25):
    torch.manual_seed(0)
    return MoEFFN(
        dim=8, hidden_dim=16, num_experts=num_experts, top_k=top_k,
        num_shared_experts=num_shared_experts, capacity_factor=capacity_factor,
    )


class TestInferenceNoCollapse:
    def test_single_token_decode_routed_experts_contribute(self):
        # The core bug: 1 token at inference → old capacity==0 → all routed
        # contributions dropped → output exactly 0 (no shared experts here).
        m = _moe()
        m.eval()
        x = torch.randn(1, 1, 8)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (1, 1, 8)
        assert out.abs().sum().item() > 0.0, "routed experts produced no output (collapse)"

    def test_eval_drops_no_tokens(self):
        # Every token in a multi-token eval forward must receive a routed
        # contribution (no silent capacity dropping at inference).
        m = _moe()
        m.eval()
        x = torch.randn(1, 20, 8)
        with torch.no_grad():
            out = m(x).view(-1, 8)
        per_token = out.abs().sum(dim=-1)
        assert (per_token > 0).all(), "some tokens were dropped (zero output) at inference"


class TestTrainingCapacity:
    def test_training_does_not_overdrop_with_topk_formula(self):
        # With the top_k-aware formula + slack, a normal-sized training batch is
        # not dropped: capacity = 1.25 * N * top_k / num_experts exceeds the
        # expected per-expert load (N * top_k / num_experts). Every token row
        # should still get a routed contribution.
        m = _moe(num_experts=4, top_k=2)
        m.train()
        torch.manual_seed(1)
        x = torch.randn(1, 64, 8)
        out = m(x).view(-1, 8)
        per_token = out.abs().sum(dim=-1)
        assert (per_token > 0).all(), "training over-dropped tokens"

    def test_capacity_factor_zero_disables_dropping(self):
        m = _moe(capacity_factor=0.0)
        m.train()
        x = torch.randn(1, 32, 8)
        out = m(x).view(-1, 8)
        assert (out.abs().sum(dim=-1) > 0).all()


class TestSharedExpertStillRuns:
    def test_shared_expert_contributes_on_top_of_routed(self):
        # Sanity: with a shared expert, output = routed + shared (both nonzero),
        # and the routed part is no longer silenced at inference.
        m = _moe(num_shared_experts=1)
        m.eval()
        x = torch.randn(1, 1, 8)
        with torch.no_grad():
            out = m(x)
        assert out.abs().sum().item() > 0.0
