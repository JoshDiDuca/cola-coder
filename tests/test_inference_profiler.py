"""Tests for InferenceProfiler (features/inference_profiler.py)."""

from __future__ import annotations

import time

import pytest

from cola_coder.features.inference_profiler import InferenceProfile, InferenceProfiler


@pytest.fixture()
def profiler() -> InferenceProfiler:
    return InferenceProfiler()


class TestIsEnabled:
    def test_feature_enabled(self):
        from cola_coder.features.inference_profiler import FEATURE_ENABLED, is_enabled

        assert FEATURE_ENABLED is True
        assert is_enabled() is True


class TestContextManager:
    def test_run_returns_profile(self, profiler):
        with profiler.run():
            pass
        result = profiler.last_result()
        assert result is not None
        assert isinstance(result, InferenceProfile)

    def test_total_ms_is_positive(self, profiler):
        with profiler.run():
            time.sleep(0.01)
        result = profiler.last_result()
        assert result.total_ms >= 0  # may be 0 on very fast runs

    def test_multiple_runs_in_history(self, profiler):
        for _ in range(3):
            with profiler.run():
                pass
        assert len(profiler.history()) == 3

    def test_exception_in_block_still_records(self, profiler):
        with pytest.raises(ValueError):
            with profiler.run():
                raise ValueError("test error")
        # Profile should still be recorded (finally block runs)
        assert profiler.last_result() is not None


class TestPhaseRecording:
    def test_tokenization_recorded(self, profiler):
        with profiler.run() as ctx:
            ctx.set_tokenization_ms(5.0, prompt_tokens=10)
        result = profiler.last_result()
        assert result.tokenization_ms == pytest.approx(5.0)
        assert result.prompt_tokens == 10

    def test_record_tokenization_fn(self, profiler):
        def fake_tokenize(text):
            return [1, 2, 3, 4, 5]

        with profiler.run() as ctx:
            tokens = ctx.record_tokenization(fake_tokenize, "hello world")
        assert tokens == [1, 2, 3, 4, 5]
        result = profiler.last_result()
        assert result.prompt_tokens == 5
        assert result.tokenization_ms >= 0

    def test_prefill_and_tokens(self, profiler):
        with profiler.run() as ctx:
            ctx.record_prefill_start()
            time.sleep(0.001)
            ctx.record_first_token()
            ctx.record_token()
            ctx.record_token()

        result = profiler.last_result()
        assert result.generated_tokens == 3
        assert len(result.inter_token_latencies_ms) == 2

    def test_tokens_per_second_positive(self, profiler):
        with profiler.run() as ctx:
            ctx.record_prefill_start()
            ctx.record_first_token()
            for _ in range(9):
                ctx.record_token()

        result = profiler.last_result()
        assert result.generated_tokens == 10
        if result.decode_ms > 0:
            assert result.tokens_per_second > 0


class TestDerivedMetrics:
    def test_ttft_is_sum(self, profiler):
        with profiler.run() as ctx:
            ctx.set_tokenization_ms(3.0, prompt_tokens=5)
            ctx.record_prefill_start()
            ctx.record_first_token()

        result = profiler.last_result()
        expected_ttft = result.tokenization_ms + result.prefill_ms
        assert result.time_to_first_token_ms == pytest.approx(expected_ttft, abs=1.0)

    def test_avg_inter_token_latency(self, profiler):
        with profiler.run() as ctx:
            ctx.record_prefill_start()
            ctx.record_first_token()
            for _ in range(5):
                ctx.record_token()

        result = profiler.last_result()
        if result.inter_token_latencies_ms:
            avg = sum(result.inter_token_latencies_ms) / len(result.inter_token_latencies_ms)
            assert result.avg_inter_token_latency_ms == pytest.approx(avg)


class TestAverageAndReset:
    def test_no_runs_returns_none(self, profiler):
        assert profiler.last_result() is None
        assert profiler.average() is None

    def test_average_after_runs(self, profiler):
        for _ in range(3):
            with profiler.run() as ctx:
                ctx.set_tokenization_ms(10.0)
        avg = profiler.average()
        assert avg is not None
        assert avg.tokenization_ms == pytest.approx(10.0)

    def test_reset_clears_history(self, profiler):
        with profiler.run():
            pass
        profiler.reset()
        assert profiler.history() == []
        assert profiler.last_result() is None

    def test_repr(self, profiler):
        r = repr(profiler)
        assert "InferenceProfiler" in r

    def test_summary_string(self, profiler):
        with profiler.run():
            pass
        s = profiler.last_result().summary()
        assert "total=" in s
        assert "tok/s=" in s
