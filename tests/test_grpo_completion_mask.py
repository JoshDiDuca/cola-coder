"""MODEL-004: GRPO policy log-prob must sum over COMPLETION tokens only.

`_completion_logprob_sum` masks the prompt out of the per-sequence log-prob.
The prompt is fixed context, not a sampled action, so only completion tokens
count under the policy — matching reference GRPO. (Summing the prompt too was
harmless given mean-centered, shared-prompt advantages, but is non-standard and
fragile to future changes.)

`token_log_probs[j]` scores token j+1, so completion tokens (index >= prompt_len)
are scored by indices >= prompt_len - 1.
"""

import torch

from cola_coder.reasoning.grpo import _completion_logprob_sum


class TestCompletionLogprobSum:
    def test_masks_prompt_tokens(self):
        # 6-token sequence -> 5 log-probs scoring tokens 1..5.
        lp = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        # prompt_len=3 -> completion tokens are indices 3,4,5, scored by lp[2:].
        assert _completion_logprob_sum(lp, prompt_len=3).item() == 12.0  # 3+4+5

    def test_prompt_len_one_keeps_all(self):
        # prompt_len=1 (just BOS) -> start=0 -> sum everything.
        lp = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _completion_logprob_sum(lp, prompt_len=1).item() == 15.0

    def test_no_completion_tokens_is_zero(self):
        lp = torch.tensor([1.0, 2.0, 3.0])
        # start = prompt_len-1 = 5 >= len(lp)=3 -> no completion tokens.
        out = _completion_logprob_sum(lp, prompt_len=6)
        assert out.item() == 0.0
        assert out.shape == ()

    def test_masked_plus_prompt_equals_full(self):
        # Sanity: completion sum + prompt-portion sum == full sum.
        lp = torch.tensor([0.5, -1.0, 2.0, -0.5, 3.0])
        prompt_len = 3
        comp = _completion_logprob_sum(lp, prompt_len).item()
        prompt_portion = lp[: prompt_len - 1].sum().item()
        assert abs((comp + prompt_portion) - lp.sum().item()) < 1e-6

    def test_start_exactly_at_end_is_zero(self):
        lp = torch.tensor([1.0, 2.0, 3.0])
        # start = prompt_len-1 = 3 == len(lp) -> empty completion -> 0.
        assert _completion_logprob_sum(lp, prompt_len=4).item() == 0.0
