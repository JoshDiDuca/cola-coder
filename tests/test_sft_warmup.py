"""Tests for the SFT Warmup module.

Verifies:
- SFTWarmup class initialises correctly
- generate_synthetic_examples() returns properly formatted data
- train() runs without error on CPU with a mock model
- Integration with existing CoT data from cot_data.py
- Feature flag (FEATURE_ENABLED / is_enabled) behaves correctly
- Cosine LR scheduler helper works correctly

All tests run on CPU and use mocked/tiny models — no GPU required, no hangs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Make sure the package is importable from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cola_coder.reasoning.sft_warmup import (
    FEATURE_ENABLED,
    SFTWarmup,
    _cosine_scheduler,
    is_enabled,
)
from cola_coder.reasoning.cot_data import (
    COT_EXAMPLES,
    get_cot_training_data,
)


# ---------------------------------------------------------------------------
# Helpers — tiny CPU model + fake tokenizer (no GPU needed)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 64
DIM = 16
SEQ_LEN = 32


class TinyModel(nn.Module):
    """Minimal decoder-only model sufficient for SFTWarmup tests."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, dim: int = DIM):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.linear = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.linear(self.emb(x))


class FakeTokenizer:
    """Minimal tokenizer that encodes strings as fixed-length integer sequences."""

    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size
        self._token_counter = 0

    def encode(self, text: str, add_bos: bool = False) -> list[int]:  # noqa: D401
        # Produce a deterministic sequence of ints based on text length
        n = min(len(text) + 2, SEQ_LEN)
        return [(i % (self.vocab_size - 1)) + 1 for i in range(n)]


def make_sft_warmup(**kwargs) -> SFTWarmup:
    """Create an SFTWarmup on CPU with a tiny model."""
    model = TinyModel()
    tokenizer = FakeTokenizer()
    defaults = dict(model=model, tokenizer=tokenizer, device="cpu", precision="fp32")
    defaults.update(kwargs)
    return SFTWarmup(**defaults)


# ---------------------------------------------------------------------------
# Feature flag tests
# ---------------------------------------------------------------------------


class TestFeatureFlag:
    def test_feature_enabled_is_true_by_default(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled_returns_true(self):
        assert is_enabled() is True

    def test_is_enabled_returns_bool(self):
        result = is_enabled()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# SFTWarmup initialisation tests
# ---------------------------------------------------------------------------


class TestSFTWarmupInit:
    def test_init_stores_device(self):
        sft = make_sft_warmup()
        assert sft.device == "cpu"

    def test_init_stores_learning_rate(self):
        sft = make_sft_warmup(learning_rate=1e-4)
        assert sft.learning_rate == 1e-4

    def test_init_creates_optimizer(self):
        sft = make_sft_warmup()
        assert isinstance(sft.optimizer, torch.optim.AdamW)

    def test_init_default_precision_fp32_on_cpu(self):
        sft = make_sft_warmup(precision="fp32")
        assert not sft.use_bf16
        assert not sft.use_fp16

    def test_init_max_seq_len(self):
        sft = make_sft_warmup(max_seq_len=512)
        assert sft.max_seq_len == 512

    def test_init_model_stored(self):
        model = TinyModel()
        sft = SFTWarmup(model=model, tokenizer=FakeTokenizer(), device="cpu", precision="fp32")
        assert sft.model is model


# ---------------------------------------------------------------------------
# generate_synthetic_examples() tests
# ---------------------------------------------------------------------------


class TestGenerateSyntheticExamples:
    def test_returns_list(self):
        sft = make_sft_warmup()
        result = sft.generate_synthetic_examples()
        assert isinstance(result, list)

    def test_returns_nonempty_by_default(self):
        sft = make_sft_warmup()
        result = sft.generate_synthetic_examples()
        assert len(result) > 0

    def test_each_element_has_text_key(self):
        sft = make_sft_warmup()
        result = sft.generate_synthetic_examples()
        for item in result:
            assert "text" in item, f"Missing 'text' key in {item}"

    def test_text_is_string(self):
        sft = make_sft_warmup()
        result = sft.generate_synthetic_examples()
        for item in result:
            assert isinstance(item["text"], str)

    def test_text_contains_think_token(self):
        sft = make_sft_warmup()
        result = sft.generate_synthetic_examples()
        for item in result:
            assert "<think>" in item["text"], "Expected <think> token in synthetic example"

    def test_empty_problems_returns_empty_list(self):
        sft = make_sft_warmup()
        result = sft.generate_synthetic_examples(problems=[])
        assert result == []

    def test_custom_problems_used(self):
        sft = make_sft_warmup()
        problems = [
            {
                "task_id": "add_two",
                "prompt": "def add(a, b):\n",
                "solution": "def add(a, b):\n    return a + b",
            }
        ]
        result = sft.generate_synthetic_examples(problems=problems)
        assert len(result) >= 1

    def test_num_per_problem_multiplies_output(self):
        sft = make_sft_warmup()
        problems = [
            {
                "task_id": "add_two",
                "prompt": "def add(a, b):\n",
                "solution": "def add(a, b):\n    return a + b",
            }
        ]
        result_1 = sft.generate_synthetic_examples(problems=problems, num_per_problem=1)
        result_3 = sft.generate_synthetic_examples(problems=problems, num_per_problem=3)
        assert len(result_3) == 3 * len(result_1)


# ---------------------------------------------------------------------------
# train() tests
# ---------------------------------------------------------------------------


class TestTrain:
    def test_train_returns_float(self):
        sft = make_sft_warmup()
        examples = [{"text": "def foo():\n    return 1"}]
        result = sft.train(examples=examples, num_epochs=1)
        assert isinstance(result, float)

    def test_train_uses_seed_data_when_none_passed(self):
        sft = make_sft_warmup()
        # Should not raise; uses built-in COT_EXAMPLES
        result = sft.train(examples=None, num_epochs=1)
        assert isinstance(result, float)

    def test_train_returns_zero_on_empty_examples(self):
        sft = make_sft_warmup()
        result = sft.train(examples=[], num_epochs=1)
        assert result == 0.0

    def test_train_loss_is_finite(self):
        sft = make_sft_warmup()
        examples = [{"text": "x = 1"}]
        loss = sft.train(examples=examples, num_epochs=2)
        assert not (loss != loss)  # not NaN
        assert loss >= 0.0

    def test_train_multiple_epochs(self):
        sft = make_sft_warmup()
        examples = [{"text": "def f():\n    pass"}]
        # Should complete without error for 3 epochs
        loss = sft.train(examples=examples, num_epochs=3)
        assert isinstance(loss, float)

    def test_train_noop_when_feature_disabled(self):
        import cola_coder.reasoning.sft_warmup as module

        original = module.FEATURE_ENABLED
        try:
            module.FEATURE_ENABLED = False
            sft = make_sft_warmup()
            loss = sft.train(examples=[{"text": "x = 1"}], num_epochs=5)
            assert loss == 0.0
        finally:
            module.FEATURE_ENABLED = original


# ---------------------------------------------------------------------------
# Integration with cot_data
# ---------------------------------------------------------------------------


class TestCotDataIntegration:
    def test_cot_examples_exist(self):
        assert len(COT_EXAMPLES) >= 5

    def test_get_cot_training_data_returns_list(self):
        data = get_cot_training_data()
        assert isinstance(data, list)

    def test_get_cot_training_data_has_text_keys(self):
        data = get_cot_training_data()
        for item in data:
            assert "text" in item

    def test_cot_data_contains_think_tokens(self):
        data = get_cot_training_data()
        for item in data:
            assert "<think>" in item["text"]
            assert "</think>" in item["text"]

    def test_sft_can_train_on_cot_data(self):
        sft = make_sft_warmup()
        data = get_cot_training_data()
        loss = sft.train(examples=data, num_epochs=1)
        assert isinstance(loss, float)
        assert loss >= 0.0


# ---------------------------------------------------------------------------
# Cosine scheduler tests
# ---------------------------------------------------------------------------


class TestCosineScheduler:
    def test_scheduler_is_lambdalr(self):
        model = TinyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = _cosine_scheduler(opt, total_steps=10)
        assert isinstance(sched, torch.optim.lr_scheduler.LambdaLR)

    def test_scheduler_starts_at_one(self):
        model = TinyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = _cosine_scheduler(opt, total_steps=100)
        lrs = sched.get_last_lr()
        # Initial LR should be close to the full LR
        assert lrs[0] == pytest.approx(1e-3, rel=0.05)

    def test_scheduler_decays_over_steps(self):
        model = TinyModel()
        lr = 1e-3
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        sched = _cosine_scheduler(opt, total_steps=10, min_lr_ratio=0.1)
        for _ in range(10):
            sched.step()
        final_lr = sched.get_last_lr()[0]
        # Final LR should be significantly lower than initial
        assert final_lr < lr * 0.5

    def test_scheduler_handles_total_steps_one(self):
        model = TinyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # Should not raise
        sched = _cosine_scheduler(opt, total_steps=1)
        sched.step()
