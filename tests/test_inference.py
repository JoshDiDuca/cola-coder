"""Tests for inference components: sampling, evaluation metrics."""

import torch

from cola_coder.inference.sampling import (
    sample_next_token, _top_k_filter, _top_p_filter, _apply_repetition_penalty,
    _banned_ngram_tokens,
)
from cola_coder.evaluation.metrics import pass_at_k, compute_pass_at_k, ProblemResult


class TestSampling:
    """Tests for token sampling strategies."""

    def test_greedy_sampling(self):
        """Temperature 0 should always pick the highest logit."""
        logits = torch.tensor([0.1, 0.5, 2.0, 0.3, 0.8])
        token = sample_next_token(logits, temperature=0)
        assert token == 2  # Index of max value

    def test_temperature_affects_randomness(self):
        """Higher temperature should produce more diverse outputs."""
        logits = torch.tensor([0.1, 0.5, 2.0, 0.3, 0.8])

        # Low temperature — should mostly pick token 2
        low_temp_samples = set()
        for _ in range(50):
            token = sample_next_token(logits.clone(), temperature=0.01, top_k=0, top_p=1.0)
            low_temp_samples.add(token)

        # High temperature — should pick various tokens
        high_temp_samples = set()
        for _ in range(50):
            token = sample_next_token(logits.clone(), temperature=2.0, top_k=0, top_p=1.0)
            high_temp_samples.add(token)

        assert len(high_temp_samples) >= len(low_temp_samples)

    def test_top_k_filter(self):
        """Top-k should zero out all but top k logits."""
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
        filtered = _top_k_filter(logits.clone(), k=3)
        # Top 3: indices 1 (5.0), 4 (4.0), 2 (3.0)
        assert filtered[0] == float("-inf")  # 1.0 — not in top 3
        assert filtered[3] == float("-inf")  # 2.0 — not in top 3
        assert filtered[1] == 5.0  # Kept

    def test_top_p_filter(self):
        """Top-p should keep tokens until cumulative prob reaches p."""
        logits = torch.tensor([0.1, 10.0, 0.1, 0.1, 0.1])
        filtered = _top_p_filter(logits.clone(), p=0.9)
        # Token 1 has very high prob, should be the main one kept
        assert filtered[1] > float("-inf")

    def test_repetition_penalty(self):
        """Repetition penalty should reduce scores of repeated tokens."""
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        original = logits.clone()
        _apply_repetition_penalty(logits, generated_ids=[2, 4], penalty=1.5)
        assert logits[2] < original[2]  # Penalized
        assert logits[4] < original[4]  # Penalized
        assert logits[0] == original[0]  # Not penalized
        assert logits[1] == original[1]  # Not penalized


class TestSamplingRobustness:
    """Locks the safety fallback + keep-at-least-one invariants in
    sample_next_token. These have zero prior coverage but sit on EVERY
    generation step — a refactor that drops them would crash inference
    (torch.multinomial on all-zero/NaN probs) or let top_p remove everything."""

    def test_top_p_keeps_at_least_one_token(self):
        # The top token alone exceeds p → exactly ONE token must survive (the
        # shift in _top_p_filter forces this), never an empty nucleus.
        logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
        filtered = _top_p_filter(logits.clone(), p=0.5)
        finite = (filtered != float("-inf")).sum().item()
        assert finite == 1
        assert filtered.argmax().item() == 0  # the most-probable token is kept

    def test_peaked_top_p_samples_the_top_token(self):
        logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
        for _ in range(20):
            tok = sample_next_token(logits.clone(), temperature=1.0, top_k=0, top_p=0.5)
            assert tok == 0

    def test_nan_logits_do_not_crash(self):
        # An unstable model can emit NaN logits in bf16 inference. The safety
        # fallback must return a valid token id instead of crashing multinomial.
        logits = torch.tensor([1.0, float("nan"), 2.0, 0.5])
        tok = sample_next_token(logits, temperature=1.0)
        assert isinstance(tok, int) and 0 <= tok < 4

    def test_min_p_above_one_degrades_gracefully(self):
        # min_p > 1 masks even the argmax (threshold = min_p*max > max) → all
        # -inf → softmax all-zero. The fallback returns the argmax, not a crash.
        logits = torch.tensor([0.1, 5.0, 0.2, 0.3])
        tok = sample_next_token(logits.clone(), temperature=1.0, top_k=0,
                                top_p=1.0, min_p=1.5)
        assert tok == 1  # greedy fallback on the original argmax

    def test_inf_logits_do_not_crash(self):
        logits = torch.tensor([float("inf"), 1.0, 2.0])
        tok = sample_next_token(logits, temperature=1.0)
        assert isinstance(tok, int) and 0 <= tok < 3


class TestPassAtK:
    """Tests for pass@k metric computation."""

    def test_all_correct(self):
        """All correct solutions should give pass@1 = 1.0."""
        assert pass_at_k(n=10, c=10, k=1) == 1.0

    def test_none_correct(self):
        """No correct solutions should give pass@1 = 0.0."""
        assert pass_at_k(n=10, c=0, k=1) == 0.0

    def test_half_correct(self):
        """Half correct should give pass@1 = 0.5."""
        result = pass_at_k(n=10, c=5, k=1)
        assert abs(result - 0.5) < 0.01

    def test_pass_at_k_increases_with_k(self):
        """pass@k should increase as k increases."""
        p1 = pass_at_k(n=10, c=3, k=1)
        p5 = pass_at_k(n=10, c=3, k=5)
        p10 = pass_at_k(n=10, c=3, k=10)
        assert p1 <= p5 <= p10

    def test_compute_pass_at_k(self):
        """compute_pass_at_k aggregates across problems."""
        results = [
            ProblemResult(task_id="a", num_samples=10, num_correct=5),
            ProblemResult(task_id="b", num_samples=10, num_correct=10),
            ProblemResult(task_id="c", num_samples=10, num_correct=0),
        ]
        metrics = compute_pass_at_k(results, k_values=[1])
        assert "pass@1" in metrics
        # Average: (0.5 + 1.0 + 0.0) / 3 = 0.5
        assert abs(metrics["pass@1"] - 0.5) < 0.01


class TestNoRepeatNgram:
    """no_repeat_ngram_size hard-blocks tokens that would repeat a seen n-gram —
    the fix for verbatim repetition loops in code generation."""

    def test_bans_token_completing_seen_bigram(self):
        # seq: a b a -> the bigram (a,?) was (a,b); last token is a, so b is banned.
        banned = _banned_ngram_tokens([1, 2, 1], ngram_size=2)
        assert banned == {2}

    def test_trigram_prefix_match(self):
        # seq: 1 2 3 1 2 -> trigram prefix (1,2) previously continued with 3 -> ban 3.
        banned = _banned_ngram_tokens([1, 2, 3, 1, 2], ngram_size=3)
        assert banned == {3}

    def test_no_ban_when_prefix_unseen(self):
        banned = _banned_ngram_tokens([1, 2, 3, 4], ngram_size=3)
        assert banned == set()  # last prefix (3,4) never occurred before

    def test_too_short_no_ban(self):
        assert _banned_ngram_tokens([1], ngram_size=3) == set()

    def test_disabled_returns_empty(self):
        assert _banned_ngram_tokens([1, 2, 1], ngram_size=0) == set()

    def test_size_one_bans_all_seen(self):
        assert _banned_ngram_tokens([5, 7, 5], ngram_size=1) == {5, 7}

    def test_sample_respects_ban_greedy(self):
        # Greedy would pick token 2 (highest logit), but the bigram ban forbids it.
        logits = torch.tensor([0.0, 1.0, 5.0, 0.5])
        # generated [1,2,1]: bigram prefix (1,) previously continued with 2 -> 2
        # banned. Highest non-banned logit is index 1 (1.0).
        tok = sample_next_token(logits.clone(), temperature=0, generated_ids=[1, 2, 1],
                   no_repeat_ngram_size=2)
        assert tok != 2
        assert tok == 1  # highest non-banned logit

    def test_off_by_default_allows_repeat(self):
        logits = torch.tensor([0.0, 1.0, 5.0, 0.5])
        tok = sample_next_token(logits.clone(), temperature=0, generated_ids=[1, 2, 1])
        assert tok == 2  # no ban -> greedy picks the max

    def test_greedy_all_banned_does_not_emit_garbage_token(self):
        # ngram_size=1 bans every already-seen token. With all 4 vocab tokens
        # seen, the ban would mask the ENTIRE vocab -> a naive argmax of an
        # all -inf tensor returns index 0 (garbage). The all-vocab guard must
        # skip the ban and fall back to the real greedy max (index 2).
        logits = torch.tensor([0.0, 1.0, 5.0, 0.5])
        tok = sample_next_token(
            logits.clone(), temperature=0,
            generated_ids=[0, 1, 2, 3], no_repeat_ngram_size=1,
        )
        assert tok == 2  # highest real logit, NOT 0

    def test_sampling_all_banned_returns_valid_token(self):
        # Same all-vocab-banned scenario on the stochastic path: must return a
        # real in-range token, never a masked/garbage one.
        logits = torch.tensor([0.0, 1.0, 5.0, 0.5])
        toks = {
            sample_next_token(
                logits.clone(), temperature=1.0, top_k=0, top_p=1.0,
                generated_ids=[0, 1, 2, 3], no_repeat_ngram_size=1,
            )
            for _ in range(50)
        }
        assert toks  # produced something
        assert all(0 <= t < 4 for t in toks)

    def test_partial_ban_still_applies(self):
        # Guard must NOT disable bans that leave at least one option: token 2
        # (the greedy max) is banned, so index 1 should win.
        logits = torch.tensor([0.0, 1.0, 5.0, 0.5])
        tok = sample_next_token(
            logits.clone(), temperature=0,
            generated_ids=[1, 2, 1], no_repeat_ngram_size=2,
        )
        assert tok == 1
