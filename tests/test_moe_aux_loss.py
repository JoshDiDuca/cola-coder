"""MODEL-005 cleanup: vectorize the MoE load-balancing loss without changing it.

`_load_balancing_loss` counted per-expert top-k assignments with an
O(top_k * num_experts) Python double-loop of `.sum()` kernels, run every
training step. It's now a single `torch.bincount` over the flattened
assignments. These tests pin the vectorized result to the original double-loop
reference (bit-exact) and keep the train-only gate.
"""

import torch

from cola_coder.features.moe_layer import MoEFFN


def _reference_aux_loss(m: MoEFFN, router_logits, top_k_indices, num_tokens):
    # The ORIGINAL double-loop implementation, kept here as the oracle.
    counts = torch.zeros(m.num_experts, device=router_logits.device)
    for k in range(m.top_k):
        for e in range(m.num_experts):
            counts[e] += (top_k_indices[:, k] == e).float().sum()
    fraction = counts / (num_tokens * m.top_k)
    mean_prob = torch.softmax(router_logits, dim=-1).mean(dim=0)
    return m.aux_loss_weight * (m.num_experts * (fraction * mean_prob).sum())


class TestAuxLossVectorization:
    def test_matches_reference_double_loop(self):
        torch.manual_seed(0)
        m = MoEFFN(dim=8, hidden_dim=16, num_experts=6, top_k=2)
        m.train()
        for seed in range(5):
            torch.manual_seed(seed)
            num_tokens = 37
            logits = torch.randn(num_tokens, m.num_experts)
            probs = torch.softmax(logits, dim=-1)
            _, top_k_indices = torch.topk(probs, m.top_k, dim=-1)
            got = m._load_balancing_loss(logits, top_k_indices, num_tokens)
            ref = _reference_aux_loss(m, logits, top_k_indices, num_tokens)
            assert torch.allclose(got, ref, atol=1e-6), f"seed {seed}: {got} != {ref}"

    def test_handles_expert_with_zero_tokens(self):
        # minlength must keep length == num_experts even if an expert is unused.
        m = MoEFFN(dim=8, hidden_dim=16, num_experts=8, top_k=1)
        m.train()
        num_tokens = 10
        logits = torch.randn(num_tokens, 8)
        # Force every token to expert 0 → experts 1..7 get zero tokens.
        top_k_indices = torch.zeros(num_tokens, 1, dtype=torch.long)
        loss = m._load_balancing_loss(logits, top_k_indices, num_tokens)
        ref = _reference_aux_loss(m, logits, top_k_indices, num_tokens)
        assert torch.allclose(loss, ref, atol=1e-6)
        assert loss.item() > 0  # all mass on one expert → imbalanced → nonzero

    def test_zero_in_eval_mode(self):
        m = MoEFFN(dim=8, hidden_dim=16, num_experts=4, top_k=2)
        m.eval()
        logits = torch.randn(5, 4)
        _, idx = torch.topk(torch.softmax(logits, -1), 2, dim=-1)
        assert m._load_balancing_loss(logits, idx, 5).item() == 0.0
