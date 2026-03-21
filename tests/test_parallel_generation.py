"""Tests for the Parallel/Batched Generation feature.

Coverage:
- sample_next_tokens_batch (sampling.py)
- expand_cache / expand_caches (attention.py / transformer.py)
- generate_group (generator.py)
- compute_batch_rewards_parallel (reward.py)
- GRPOTrainer batched-generation path (grpo.py)

All tests run on CPU and mock the model/tokenizer so no GPU is needed.
ProcessPoolExecutor calls are either tested with lightweight work or mocked
to avoid hangs on Windows.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from cola_coder.inference.sampling import (
    sample_next_tokens_batch,
    _top_k_filter_batch,
    _top_p_filter_batch,
)
from cola_coder.model.attention import GroupedQueryAttention
from cola_coder.model.config import ModelConfig
from cola_coder.model.transformer import Transformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_config() -> ModelConfig:
    """Minimal ModelConfig for fast CPU tests."""
    return ModelConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim_multiplier=1.0,
        max_seq_len=64,
        dropout=0.0,
        rope_theta=10000.0,
    )


def _make_transformer() -> Transformer:
    """Return a tiny CPU-only Transformer."""
    cfg = _tiny_config()
    model = Transformer(cfg)
    model.eval()
    return model


def _make_tokenizer(vocab_size: int = 256) -> MagicMock:
    """Return a mock tokenizer that encodes text as single-byte IDs."""
    tok = MagicMock()
    tok.eos_id = 0
    tok.encode.side_effect = lambda text, add_bos=False: (
        [1] + [ord(c) % vocab_size for c in text[:8]]
        if add_bos
        else [ord(c) % vocab_size for c in text[:8]]
    )
    tok.decode.side_effect = lambda ids: "".join(chr(max(32, i % 128)) for i in ids)
    return tok


def _fake_info(correct: bool = False) -> dict:
    return {
        "correct": correct,
        "has_thinking": False,
        "thinking_length": 0,
        "format_bonus": 0.0,
        "length_penalty": 0.0,
        "execution_output": "",
    }


# ---------------------------------------------------------------------------
# 1. sample_next_tokens_batch
# ---------------------------------------------------------------------------


class TestSampleNextTokensBatch:

    def test_output_shape(self):
        """Returns a 1-D tensor of length batch_size."""
        batch, vocab = 4, 100
        logits = torch.randn(batch, vocab)
        result = sample_next_tokens_batch(logits, temperature=1.0)
        assert result.shape == (batch,)

    def test_values_in_vocab_range(self):
        """Every sampled token ID is within [0, vocab_size)."""
        batch, vocab = 8, 50
        logits = torch.randn(batch, vocab)
        result = sample_next_tokens_batch(logits, temperature=0.8, top_k=10, top_p=0.9)
        assert result.min().item() >= 0
        assert result.max().item() < vocab

    def test_greedy_argmax(self):
        """Temperature=0 must return the argmax for every row."""
        logits = torch.zeros(3, 10)
        logits[0, 3] = 10.0
        logits[1, 7] = 10.0
        logits[2, 0] = 10.0
        result = sample_next_tokens_batch(logits, temperature=0)
        assert result[0].item() == 3
        assert result[1].item() == 7
        assert result[2].item() == 0

    def test_top_k_one_equals_greedy(self):
        """With top_k=1, always picks the argmax (equiv. to greedy)."""
        torch.manual_seed(42)
        logits = torch.randn(5, 20)
        argmaxes = logits.argmax(dim=-1)
        result = sample_next_tokens_batch(logits, temperature=1.0, top_k=1, top_p=1.0)
        assert torch.all(result == argmaxes)

    def test_top_p_very_tight(self):
        """With overwhelmingly dominant token, nucleus sampling picks it."""
        logits = torch.zeros(3, 10)
        logits[:, 5] = 100.0  # p ≈ 1.0 for token 5
        result = sample_next_tokens_batch(logits, temperature=1.0, top_k=0, top_p=0.01)
        assert torch.all(result == 5)

    def test_different_rows_can_differ(self):
        """Two independent calls should not always produce the same tokens."""
        torch.manual_seed(0)
        logits = torch.randn(16, 256)
        r1 = sample_next_tokens_batch(logits.clone(), temperature=1.5)
        r2 = sample_next_tokens_batch(logits.clone(), temperature=1.5)
        # Two independent samples should differ on at least one row
        assert not torch.all(r1 == r2)

    def test_batch_size_one(self):
        """Degenerate batch of 1 should work without error."""
        logits = torch.randn(1, 50)
        result = sample_next_tokens_batch(logits, temperature=0.8)
        assert result.shape == (1,)
        assert 0 <= result.item() < 50

    def test_deterministic_with_seed(self):
        """Same seed → same result."""
        logits = torch.randn(4, 64)
        torch.manual_seed(7)
        r1 = sample_next_tokens_batch(logits.clone(), temperature=0.9)
        torch.manual_seed(7)
        r2 = sample_next_tokens_batch(logits.clone(), temperature=0.9)
        assert torch.all(r1 == r2)


class TestTopKFilterBatch:

    def test_only_k_tokens_remain(self):
        """At most k logits should be finite per row."""
        logits = torch.randn(4, 20)
        filtered = _top_k_filter_batch(logits.clone(), k=3)
        finite_count = filtered.isfinite().sum(dim=-1)
        assert (finite_count <= 3).all()

    def test_k_ge_vocab_no_change(self):
        """When k >= vocab_size, nothing should be filtered."""
        logits = torch.randn(2, 10)
        filtered = _top_k_filter_batch(logits.clone(), k=100)
        assert torch.all(filtered.isfinite())


class TestTopPFilterBatch:

    def test_dominant_token_survives(self):
        """Token with near-1.0 probability must survive any nucleus threshold."""
        logits = torch.zeros(2, 50)
        logits[:, 7] = 100.0  # p ≈ 1.0 for token 7
        filtered = _top_p_filter_batch(logits.clone(), p=0.9)
        assert filtered[:, 7].isfinite().all()

    def test_output_shape_unchanged(self):
        logits = torch.randn(3, 30)
        filtered = _top_p_filter_batch(logits.clone(), p=0.95)
        assert filtered.shape == logits.shape


# ---------------------------------------------------------------------------
# 2. expand_cache / expand_caches
# ---------------------------------------------------------------------------


class TestExpandCache:

    def _make_attention(self) -> GroupedQueryAttention:
        return GroupedQueryAttention(
            dim=64, n_heads=4, n_kv_heads=2, max_seq_len=32
        )

    def test_expand_cache_batch_dimension(self):
        """After expand_cache(N), cache_k/v should have batch dim = N."""
        attn = self._make_attention()
        attn._init_cache(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        assert attn.cache_k.shape[0] == 1

        attn.expand_cache(batch_size=4)
        assert attn.cache_k.shape[0] == 4
        assert attn.cache_v.shape[0] == 4

    def test_expand_cache_preserves_values(self):
        """All expanded rows should equal the original row."""
        attn = self._make_attention()
        attn._init_cache(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        attn.cache_k[0, :5] = torch.arange(5, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)

        attn.expand_cache(batch_size=3)
        for i in range(3):
            assert torch.allclose(attn.cache_k[i, :5], attn.cache_k[0, :5])

    def test_expand_cache_no_cache_is_noop(self):
        """expand_cache when cache is None must not raise."""
        attn = self._make_attention()
        assert attn.cache_k is None
        attn.expand_cache(batch_size=4)  # Should not raise
        assert attn.cache_k is None  # Still None — nothing was initialised

    def test_expand_caches_all_layers(self):
        """Transformer.expand_caches should expand every layer's cache."""
        model = _make_transformer()
        ids = torch.randint(1, 100, (1, 4))
        with torch.no_grad():
            model(ids, start_pos=0, use_cache=True)

        model.expand_caches(batch_size=3)
        for block in model.blocks:
            assert block.attention.cache_k.shape[0] == 3
            assert block.attention.cache_v.shape[0] == 3


# ---------------------------------------------------------------------------
# 3. generate_group
# ---------------------------------------------------------------------------


class TestGenerateGroup:

    def _make_generator(self):
        from cola_coder.inference.generator import CodeGenerator

        model = _make_transformer()
        tokenizer = _make_tokenizer()
        return CodeGenerator(model=model, tokenizer=tokenizer, device="cpu")

    def test_returns_correct_count(self):
        """generate_group should return exactly num_completions strings."""
        gen = self._make_generator()
        results = gen.generate_group(
            prompt="hello", num_completions=4, max_new_tokens=5
        )
        assert len(results) == 4

    def test_all_results_are_strings(self):
        """Every completion must be a non-empty string."""
        gen = self._make_generator()
        results = gen.generate_group(
            prompt="x", num_completions=3, max_new_tokens=4
        )
        for r in results:
            assert isinstance(r, str)
            assert len(r) > 0

    def test_greedy_completions_are_identical(self):
        """Temperature=0 (greedy) must produce the same completion every time."""
        gen = self._make_generator()
        results = gen.generate_group(
            prompt="abc", num_completions=4, max_new_tokens=6, temperature=0
        )
        assert len(set(results)) == 1

    def test_fallback_on_oom(self):
        """generate_group falls back to serial generation on OOM."""
        gen = self._make_generator()
        serial_results = ["out1", "out2"]

        with patch.object(
            gen,
            "_generate_group_batched",
            side_effect=torch.cuda.OutOfMemoryError(),
        ):
            with patch.object(gen, "generate", side_effect=serial_results):
                results = gen.generate_group(
                    prompt="p", num_completions=2, max_new_tokens=4
                )
        assert results == serial_results

    def test_num_completions_one(self):
        """Degenerate case: num_completions=1 returns a list of length 1."""
        gen = self._make_generator()
        results = gen.generate_group(
            prompt="hi", num_completions=1, max_new_tokens=3
        )
        assert len(results) == 1

    def test_mini_batching_stitches_results(self):
        """When batch_size < num_completions, results from mini-batches are combined."""
        gen = self._make_generator()
        # Request 4 completions but only allow batch_size=2 at a time
        results = gen._generate_group_batched(
            prompt="z",
            num_completions=4,
            max_new_tokens=3,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            batch_size=2,
        )
        assert len(results) == 4


# ---------------------------------------------------------------------------
# 4. compute_batch_rewards_parallel
# ---------------------------------------------------------------------------


class TestComputeBatchRewardsParallel:

    def test_workers_one_uses_serial(self):
        """workers=1 should delegate to compute_batch_rewards (serial)."""
        from cola_coder.reasoning.reward import compute_batch_rewards_parallel

        with patch(
            "cola_coder.reasoning.reward.compute_batch_rewards",
            return_value=([0.5, 1.0], [_fake_info(False), _fake_info(True)]),
        ) as mock_serial:
            rewards, infos = compute_batch_rewards_parallel(
                ["a", "b"], "test", workers=1
            )

        mock_serial.assert_called_once()
        assert rewards == [0.5, 1.0]

    def test_returns_correct_count_via_mock_pool(self):
        """Parallel call returns one reward and one info per generation."""
        from cola_coder.reasoning.reward import compute_batch_rewards_parallel

        n = 3
        fake_result = (0.0, _fake_info(False))

        with patch("cola_coder.reasoning.reward.ProcessPoolExecutor") as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool_cls.return_value.__enter__ = MagicMock(return_value=mock_pool)
            mock_pool_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_futures = []
            for _ in range(n):
                f = MagicMock()
                f.result.return_value = fake_result
                mock_futures.append(f)
            mock_pool.submit.side_effect = mock_futures

            rewards, infos = compute_batch_rewards_parallel(
                ["g1", "g2", "g3"], "tests", workers=2
            )

        assert len(rewards) == n
        assert len(infos) == n

    def test_fallback_on_executor_error(self):
        """If ProcessPoolExecutor raises, fall back to serial."""
        from cola_coder.reasoning.reward import compute_batch_rewards_parallel

        with patch(
            "cola_coder.reasoning.reward.ProcessPoolExecutor",
            side_effect=RuntimeError("spawn failed"),
        ):
            with patch(
                "cola_coder.reasoning.reward.compute_batch_rewards",
                return_value=([0.0], [_fake_info(False)]),
            ) as mock_serial:
                rewards, infos = compute_batch_rewards_parallel(
                    ["a"], "test", workers=2
                )

        mock_serial.assert_called_once()
        assert rewards == [0.0]

    def test_timed_out_future_gives_zero_reward(self):
        """A future that times out contributes reward=0, correct=False."""
        from cola_coder.reasoning.reward import compute_batch_rewards_parallel
        from concurrent.futures import TimeoutError as FT

        with patch("cola_coder.reasoning.reward.ProcessPoolExecutor") as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool_cls.return_value.__enter__ = MagicMock(return_value=mock_pool)
            mock_pool_cls.return_value.__exit__ = MagicMock(return_value=False)
            timed_out_future = MagicMock()
            timed_out_future.result.side_effect = FT()
            mock_pool.submit.return_value = timed_out_future

            rewards, infos = compute_batch_rewards_parallel(
                ["bad_code"], "tests", workers=2, per_task_timeout=1.0
            )

        assert rewards[0] == 0.0
        assert infos[0]["correct"] is False
        assert infos[0]["execution_output"] == "timeout"

    def test_errored_future_gives_zero_reward(self):
        """A future that raises an unexpected exception gives reward=0."""
        from cola_coder.reasoning.reward import compute_batch_rewards_parallel

        with patch("cola_coder.reasoning.reward.ProcessPoolExecutor") as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool_cls.return_value.__enter__ = MagicMock(return_value=mock_pool)
            mock_pool_cls.return_value.__exit__ = MagicMock(return_value=False)
            bad_future = MagicMock()
            bad_future.result.side_effect = ValueError("crash")
            mock_pool.submit.return_value = bad_future

            rewards, infos = compute_batch_rewards_parallel(
                ["code"], "tests", workers=2
            )

        assert rewards[0] == 0.0
        assert "error" in infos[0]["execution_output"]


# ---------------------------------------------------------------------------
# 5. GRPOTrainer — batched-generation path
# ---------------------------------------------------------------------------


class TestGRPOTrainerParallelGeneration:
    """Verify that GRPOTrainer calls generate_group when parallel_generation=True."""

    def _make_trainer(
        self, parallel_generation: bool = False, parallel_rewards: bool = False
    ):
        from cola_coder.reasoning.grpo import GRPOTrainer

        model = _make_transformer()
        tokenizer = _make_tokenizer()

        return GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            group_size=2,
            max_new_tokens=4,
            device="cpu",
            parallel_generation=parallel_generation,
            parallel_rewards=parallel_rewards,
            reward_workers=1,
        )

    def _fake_rewards(self, n: int = 2) -> tuple[list[float], list[dict]]:
        return ([0.0] * n, [_fake_info(False) for _ in range(n)])

    def test_parallel_flag_stored(self):
        trainer = self._make_trainer(parallel_generation=True)
        assert trainer.parallel_generation is True

    def test_serial_flag_stored(self):
        trainer = self._make_trainer(parallel_generation=False)
        assert trainer.parallel_generation is False

    def test_reward_workers_stored(self):
        trainer = self._make_trainer(parallel_generation=True, parallel_rewards=True)
        assert trainer.reward_workers == 1

    def test_parallel_rewards_flag_stored(self):
        trainer = self._make_trainer(parallel_generation=False, parallel_rewards=True)
        assert trainer.parallel_rewards is True

    def test_generate_group_called_when_parallel(self):
        """generate_group() should be called once when parallel_generation=True."""
        trainer = self._make_trainer(parallel_generation=True)
        fake_gens = ["result_a", "result_b"]

        with patch.object(
            trainer.generator, "generate_group", return_value=fake_gens
        ) as mock_gg:
            # Stub the reward function to avoid subprocess calls
            trainer._reward_fn = MagicMock(return_value=self._fake_rewards(2))
            trainer.train_step(prompt="def foo():", test_code="assert True")

        mock_gg.assert_called_once()

    def test_serial_generate_called_when_not_parallel(self):
        """generator.generate() is called group_size times when serial."""
        trainer = self._make_trainer(parallel_generation=False)
        fake_gen = "def foo(): pass"

        with patch.object(
            trainer.generator, "generate", return_value=fake_gen
        ) as mock_g:
            trainer._reward_fn = MagicMock(return_value=self._fake_rewards(2))
            trainer.train_step(prompt="def foo():", test_code="assert True")

        assert mock_g.call_count == trainer.group_size

    def test_fallback_to_serial_on_generate_group_error(self):
        """If generate_group raises, trainer falls back to serial generate()."""
        trainer = self._make_trainer(parallel_generation=True)
        fake_gen = "fallback result"

        with patch.object(
            trainer.generator, "generate_group", side_effect=RuntimeError("boom")
        ):
            with patch.object(
                trainer.generator, "generate", return_value=fake_gen
            ) as mock_g:
                trainer._reward_fn = MagicMock(return_value=self._fake_rewards(2))
                trainer.train_step(prompt="def foo():", test_code="assert True")

        assert mock_g.call_count == trainer.group_size
        assert trainer.parallel_generation is False

    def test_train_step_returns_metrics(self):
        """train_step should return a dict with expected keys."""
        trainer = self._make_trainer(parallel_generation=False)
        fake_gen = "def foo(): return 1"

        with patch.object(trainer.generator, "generate", return_value=fake_gen):
            trainer._reward_fn = MagicMock(
                return_value=([1.0, 0.0], [_fake_info(True), _fake_info(False)])
            )
            metrics = trainer.train_step(
                prompt="def foo():", test_code="assert True"
            )

        for key in ("loss", "mean_reward", "num_correct", "group_size", "pass_rate"):
            assert key in metrics

    def test_train_step_pass_rate_correct(self):
        """pass_rate should equal num_correct / group_size."""
        trainer = self._make_trainer(parallel_generation=False)
        fake_gen = "def foo(): return 1"

        with patch.object(trainer.generator, "generate", return_value=fake_gen):
            trainer._reward_fn = MagicMock(
                return_value=([1.0, 0.0], [_fake_info(True), _fake_info(False)])
            )
            metrics = trainer.train_step(
                prompt="def foo():", test_code="assert True"
            )

        expected = metrics["num_correct"] / metrics["group_size"]
        assert abs(metrics["pass_rate"] - expected) < 1e-6

    def test_parallel_rewards_calls_parallel_function(self):
        """When parallel_rewards=True and reward_name is python_exec, call the parallel fn."""
        trainer = self._make_trainer(parallel_generation=False, parallel_rewards=True)
        # Ensure the trainer thinks it's using python_exec
        trainer._reward_name = "python_exec"
        fake_gen = "def foo(): pass"

        with patch.object(trainer.generator, "generate", return_value=fake_gen):
            with patch(
                "cola_coder.reasoning.grpo.compute_batch_rewards_parallel",
                return_value=self._fake_rewards(2),
            ) as mock_par:
                trainer.train_step(prompt="def foo():", test_code="assert True")

        mock_par.assert_called_once()
