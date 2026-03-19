"""Tests for inference components: sampling, evaluation metrics."""

import pytest
import torch

from cola_coder.inference.sampling import (
    sample_next_token, _top_k_filter, _top_p_filter, _apply_repetition_penalty
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
