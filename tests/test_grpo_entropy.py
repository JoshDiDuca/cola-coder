"""RLVR entropy collapse is the dominant failure mode of GRPO/RLVR training
(arXiv:2509.26114, arXiv:2509.21882). `completion_entropy` measures the mean
per-token Shannon entropy (nats) of the policy over completion positions only,
so the clip_low / clip_high knobs that raise / lower entropy become actionable.

The function takes LOG-PROBS (a log-softmax tensor — what train_step feeds as
`log_probs[0, :-1]`), shape [seq-1, vocab]. `log_probs_2d[j]` scores token j+1,
and only completion positions (j >= prompt_len - 1) count, mirroring
_completion_logprobs.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from cola_coder.reasoning.grpo import completion_entropy


def _logprobs(logits: torch.Tensor) -> torch.Tensor:
    """[seq-1, vocab] log-softmax — the shape train_step passes (log_probs[0, :-1])."""
    return F.log_softmax(logits, dim=-1)


class TestCompletionEntropy:
    def test_uniform_distribution_is_log_vocab(self):
        # Uniform over V tokens -> entropy = ln(V) nats, at every position.
        vocab = 8
        lp = _logprobs(torch.zeros(5, vocab))  # equal logits -> uniform
        ent = completion_entropy(lp, prompt_len=1)  # all positions count
        assert ent.item() == pytest.approx(math.log(vocab), abs=1e-5)

    def test_near_deterministic_is_near_zero(self):
        # One token dominates -> entropy collapses toward 0 (the failure mode).
        vocab = 8
        logits = torch.full((4, vocab), -20.0)
        logits[:, 0] = 20.0
        ent = completion_entropy(_logprobs(logits), prompt_len=1)
        assert 0.0 <= ent.item() < 1e-3  # non-negative AND collapsed

    def test_prompt_positions_are_excluded(self):
        # Positions < prompt_len-1 must not affect the reading. Make prompt
        # positions deterministic (entropy ~0) and completion positions uniform;
        # the mean must reflect ONLY the completion (uniform) positions.
        vocab = 4
        seq_minus1 = 6
        logits = torch.full((seq_minus1, vocab), -20.0)
        logits[:, 0] = 20.0          # near-deterministic by default
        logits[3:, :] = 0.0          # completion positions (>=3) uniform
        ent = completion_entropy(_logprobs(logits), prompt_len=4)
        assert ent.item() == pytest.approx(math.log(vocab), abs=1e-5)

    def test_no_completion_tokens_returns_zero(self):
        # prompt_len beyond the sequence -> no completion positions -> 0.0.
        lp = _logprobs(torch.randn(3, 5))
        ent = completion_entropy(lp, prompt_len=10)
        assert ent.item() == 0.0
        assert ent.shape == torch.Size([])

    def test_matches_manual_softmax_entropy(self):
        torch.manual_seed(0)
        lp = _logprobs(torch.randn(5, 6))
        # Manual mean entropy over all positions (prompt_len=1 includes all).
        p = lp.exp()
        manual = -(p * lp).sum(dim=-1).mean()
        ent = completion_entropy(lp, prompt_len=1)
        assert ent.item() == pytest.approx(manual.item(), abs=1e-6)

    def test_entropy_is_non_negative_and_detachable(self):
        torch.manual_seed(1)
        lp = _logprobs(torch.randn(4, 5, requires_grad=True))
        ent = completion_entropy(lp, prompt_len=2)
        assert ent.item() >= -1e-7  # Shannon entropy is non-negative
        # Used as a diagnostic float in train_step — must convert cleanly.
        assert isinstance(float(ent.detach()), float)
