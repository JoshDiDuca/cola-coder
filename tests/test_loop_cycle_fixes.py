"""Regression tests for the autonomous-loop cycle fixes (2026-06-10).

- INFER-002: generate_group must clear the KV-cache even when decode raises.
- EVAL-001: pass@k must report None (not 0.0) when no problem has >= k samples.
- INFER-003: _apply_repetition_penalty returns the (mutated) logits.
"""

import pytest
import torch

from cola_coder.evaluation.metrics import (
    ProblemResult,
    compute_pass_at_k,
    format_results,
)
from cola_coder.inference.sampling import _apply_repetition_penalty


# ---------------------------------------------------------------------------
# EVAL-001 — pass@k not-estimable semantics
# ---------------------------------------------------------------------------


class TestPassAtKSemantics:
    def test_none_when_no_problem_has_k_samples(self):
        # 1 sample per problem, asking for pass@10 → unestimable → None, not 0.0
        results = [ProblemResult("p1", num_samples=1, num_correct=0),
                   ProblemResult("p2", num_samples=1, num_correct=1)]
        metrics = compute_pass_at_k(results, k_values=[10])
        assert metrics["pass@10"] is None

    def test_numeric_when_enough_samples(self):
        results = [ProblemResult("p1", num_samples=10, num_correct=5)]
        metrics = compute_pass_at_k(results, k_values=[1, 5, 10])
        assert all(v is not None for v in metrics.values())
        assert 0.0 <= metrics["pass@1"] <= 1.0

    def test_pass_at_1_all_correct_is_one(self):
        results = [ProblemResult("p1", num_samples=5, num_correct=5)]
        assert compute_pass_at_k(results, [1])["pass@1"] == 1.0

    def test_pass_at_1_all_wrong_is_zero(self):
        results = [ProblemResult("p1", num_samples=5, num_correct=0)]
        assert compute_pass_at_k(results, [1])["pass@1"] == 0.0

    def test_mixed_counts_only_averages_eligible(self):
        # p1 has 10 samples (eligible for pass@5), p2 has 2 (excluded)
        results = [ProblemResult("p1", num_samples=10, num_correct=10),
                   ProblemResult("p2", num_samples=2, num_correct=0)]
        # Only p1 counts → pass@5 == 1.0 (not dragged down by excluded p2)
        assert compute_pass_at_k(results, [5])["pass@5"] == 1.0

    def test_format_results_handles_none(self):
        # format_results must not crash on a None metric
        results = [ProblemResult("p1", num_samples=1, num_correct=0)]
        text = format_results(results, k_values=[1, 10])
        assert "pass@10" in text
        assert "n/a" in text


# ---------------------------------------------------------------------------
# INFER-003 — repetition penalty returns the tensor
# ---------------------------------------------------------------------------


class TestRepetitionPenaltyReturn:
    def test_returns_same_tensor_mutated(self):
        logits = torch.tensor([2.0, -2.0, 0.5, 1.0])
        out = _apply_repetition_penalty(logits, [0, 1], penalty=2.0)
        assert out is logits  # in-place, same object
        # positive logit divided, negative multiplied
        assert out[0].item() == 1.0   # 2.0 / 2
        assert out[1].item() == -4.0  # -2.0 * 2
        assert out[2].item() == 0.5   # untouched
        assert out[3].item() == 1.0   # untouched


# ---------------------------------------------------------------------------
# INFER-002 — generate_group clears cache on exception
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    eos_id = 0

    def encode(self, text, add_bos=False):
        return [1, 2, 3]

    def decode(self, ids):
        return "x"


class _RaisingModel:
    """Minimal model stand-in: records cache clears, raises during decode."""

    def __init__(self, raise_at_call: int):
        self.clear_calls = 0
        self.expand_calls = 0
        self._call = 0
        self._raise_at = raise_at_call

    def __call__(self, input_ids, start_pos=0, use_cache=False):
        self._call += 1
        if self._call >= self._raise_at:
            raise RuntimeError("boom during decode")
        # (batch, seq, vocab) — vocab=8
        batch = input_ids.shape[0]
        return torch.zeros(batch, input_ids.shape[1], 8)

    def clear_caches(self):
        self.clear_calls += 1

    def expand_caches(self, batch_size):
        self.expand_calls += 1

    def eval(self):
        return self


def test_generate_group_clears_cache_on_decode_exception():
    from cola_coder.inference.generator import CodeGenerator

    model = _RaisingModel(raise_at_call=2)  # prefill ok (call 1), decode raises
    gen = CodeGenerator(model=model, tokenizer=_FakeTokenizer(), device="cpu")

    with pytest.raises(RuntimeError, match="boom"):
        gen._generate_group_single_batch(
            prompt="hi", batch_size=4, max_new_tokens=8,
            temperature=0.8, top_k=50, top_p=0.9,
        )

    # The finally must have cleared the cache despite the exception
    assert model.clear_calls >= 2  # entry clear + finally clear
    assert model.expand_calls == 1  # cache WAS expanded → must be cleared
