"""Tests for latency_optimizer.py."""

from __future__ import annotations

import pytest

from cola_coder.features.latency_optimizer import (
    FEATURE_ENABLED,
    HardwareConfig,
    InferenceProfile,
    LatencyOptimizer,
    LatencyReport,
    ModelConfig,
    analyze_attention,
    analyze_batch_size,
    analyze_kv_cache,
    analyze_quantization,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_model() -> ModelConfig:
    return ModelConfig(num_layers=12, num_heads=8, head_dim=64, vocab_size=32000, max_seq_len=2048)


@pytest.fixture()
def hardware() -> HardwareConfig:
    return HardwareConfig(gpu_memory_gb=24.0, gpu_bandwidth_gbps=900.0, gpu_tflops_fp16=77.0)


@pytest.fixture()
def light_profile() -> InferenceProfile:
    return InferenceProfile(
        prefill_ms=10.0,
        decode_ms_per_token=2.0,
        kv_cache_mb=500.0,
        model_weights_mb=4000.0,
        tokens_per_second=500.0,
        batch_size=1,
        prompt_length=128,
        generated_length=64,
        attention_ms=1.5,
        ffn_ms=0.4,
        sampling_ms=0.1,
    )


@pytest.fixture()
def heavy_profile() -> InferenceProfile:
    """Scenario with high KV cache usage and large batch."""
    return InferenceProfile(
        prefill_ms=50.0,
        decode_ms_per_token=10.0,
        kv_cache_mb=15000.0,
        model_weights_mb=18000.0,
        tokens_per_second=50.0,
        batch_size=64,
        prompt_length=2048,
        generated_length=512,
        attention_ms=8.0,
        ffn_ms=1.0,
        sampling_ms=1.0,
    )


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------

class TestIsEnabled:
    def test_constant(self):
        assert FEATURE_ENABLED is True

    def test_function(self):
        assert is_enabled() is True


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class TestModelConfig:
    def test_hidden_dim_inferred(self, small_model):
        assert small_model.hidden_dim == 8 * 64

    def test_kv_cache_bytes_per_token_positive(self, small_model):
        assert small_model.kv_cache_bytes_per_token > 0

    def test_kv_cache_mb_per_token_positive(self, small_model):
        assert small_model.kv_cache_mb_per_token > 0.0


# ---------------------------------------------------------------------------
# Component analyzers
# ---------------------------------------------------------------------------

class TestAnalyzeKVCache:
    def test_returns_list(self, light_profile, small_model, hardware):
        suggestions = analyze_kv_cache(light_profile, small_model, hardware)
        assert isinstance(suggestions, list)

    def test_high_kv_usage_triggers_suggestion(self, heavy_profile, small_model, hardware):
        suggestions = analyze_kv_cache(heavy_profile, small_model, hardware)
        categories = {s.category for s in suggestions}
        assert "kv_cache" in categories

    def test_long_prompt_suggests_sliding_window(self, heavy_profile, small_model, hardware):
        suggestions = analyze_kv_cache(heavy_profile, small_model, hardware)
        titles = [s.title.lower() for s in suggestions]
        assert any("sliding" in t or "window" in t for t in titles)


class TestAnalyzeBatchSize:
    def test_batch_one_suggests_increase(self, light_profile, hardware):
        suggestions = analyze_batch_size(light_profile, hardware)
        categories = {s.category for s in suggestions}
        assert "batch_size" in categories

    def test_high_throughput_suggestion_exists(self, light_profile, hardware):
        suggestions = analyze_batch_size(light_profile, hardware)
        assert len(suggestions) >= 1

    def test_suggestions_have_speedup(self, light_profile, hardware):
        suggestions = analyze_batch_size(light_profile, hardware)
        for s in suggestions:
            assert s.estimated_speedup > 0


class TestAnalyzeQuantization:
    def test_high_memory_triggers_int4(self, heavy_profile, small_model, hardware):
        suggestions = analyze_quantization(heavy_profile, small_model, hardware)
        assert any("INT4" in s.title or "int4" in s.title.lower() for s in suggestions)

    def test_returns_list(self, light_profile, small_model, hardware):
        suggestions = analyze_quantization(light_profile, small_model, hardware)
        assert isinstance(suggestions, list)


class TestAnalyzeAttention:
    def test_high_attention_fraction_suggests_flash(self, small_model):
        profile = InferenceProfile(
            attention_ms=8.0, ffn_ms=1.0, sampling_ms=0.5, other_ms=0.5
        )
        suggestions = analyze_attention(profile, small_model)
        titles = [s.title.lower() for s in suggestions]
        assert any("flash" in t for t in titles)

    def test_long_prompt_suggests_chunking(self, small_model):
        profile = InferenceProfile(prompt_length=2000, attention_ms=5.0)
        suggestions = analyze_attention(profile, small_model)
        titles = [s.title.lower() for s in suggestions]
        assert any("chunk" in t or "prefill" in t for t in titles)


# ---------------------------------------------------------------------------
# Full optimizer
# ---------------------------------------------------------------------------

class TestLatencyOptimizer:
    def test_analyze_returns_report(self, small_model, hardware, light_profile):
        optimizer = LatencyOptimizer(small_model, hardware)
        report = optimizer.analyze(light_profile)
        assert isinstance(report, LatencyReport)

    def test_report_has_suggestions(self, small_model, hardware, light_profile):
        optimizer = LatencyOptimizer(small_model, hardware)
        report = optimizer.analyze(light_profile)
        assert len(report.suggestions) > 0

    def test_suggestions_sorted_by_priority(self, small_model, hardware, heavy_profile):
        optimizer = LatencyOptimizer(small_model, hardware)
        report = optimizer.analyze(heavy_profile)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        for a, b in zip(report.suggestions, report.suggestions[1:]):
            assert priority_order[a.priority] <= priority_order[b.priority]

    def test_estimate_kv_cache_capacity(self, small_model, hardware):
        optimizer = LatencyOptimizer(small_model, hardware)
        capacity = optimizer.estimate_kv_cache_capacity()
        assert capacity > 0

    def test_optimal_batch_size(self, small_model, hardware):
        optimizer = LatencyOptimizer(small_model, hardware)
        batch = optimizer.optimal_batch_size(1000.0, 128, 64)
        assert batch >= 1

    def test_summary_contains_bottleneck(self, small_model, hardware, light_profile):
        optimizer = LatencyOptimizer(small_model, hardware)
        report = optimizer.analyze(light_profile)
        s = report.summary()
        assert "Bottleneck" in s

    def test_high_priority_subset(self, small_model, hardware, heavy_profile):
        optimizer = LatencyOptimizer(small_model, hardware)
        report = optimizer.analyze(heavy_profile)
        hp = report.high_priority()
        assert all(s.priority == "high" for s in hp)
