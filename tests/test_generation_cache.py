"""Tests for GenerationCache (features/generation_cache.py)."""

from __future__ import annotations

import pytest

from cola_coder.features.generation_cache import GenerationCache


@pytest.fixture()
def cache() -> GenerationCache:
    return GenerationCache(max_size=10)


class TestIsEnabled:
    def test_feature_enabled(self):
        from cola_coder.features.generation_cache import FEATURE_ENABLED, is_enabled

        assert FEATURE_ENABLED is True
        assert is_enabled() is True


class TestGetSet:
    def test_miss_returns_none(self, cache):
        assert cache.get("unknown prompt") is None

    def test_set_then_get(self, cache):
        cache.set("hello", "world")
        assert cache.get("hello") == "world"

    def test_overwrite_existing(self, cache):
        cache.set("prompt", "v1")
        cache.set("prompt", "v2")
        assert cache.get("prompt") == "v2"

    def test_different_prompts_independent(self, cache):
        cache.set("a", "result_a")
        cache.set("b", "result_b")
        assert cache.get("a") == "result_a"
        assert cache.get("b") == "result_b"

    def test_contains(self, cache):
        cache.set("x", "y")
        assert "x" in cache
        assert "z" not in cache

    def test_len(self, cache):
        assert len(cache) == 0
        cache.set("a", "1")
        cache.set("b", "2")
        assert len(cache) == 2


class TestLRUEviction:
    def test_eviction_on_overflow(self):
        cache = GenerationCache(max_size=3)
        for i in range(4):
            cache.set(f"prompt_{i}", f"output_{i}")
        assert len(cache) == 3

    def test_lru_evicts_oldest_unused(self):
        cache = GenerationCache(max_size=3, key_hash=False)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        # Access 'a' to make it recently used
        cache.get("a")
        # Add new entry — 'b' should be evicted (LRU)
        cache.set("d", "4")
        assert cache.get("a") == "1"
        assert cache.get("d") == "4"
        assert cache.get("b") is None  # evicted

    def test_eviction_count_in_stats(self):
        cache = GenerationCache(max_size=2)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")  # evicts 'a'
        stats = cache.stats()
        assert stats.evictions == 1


class TestStats:
    def test_initial_stats(self, cache):
        stats = cache.stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0

    def test_hit_and_miss_counts(self, cache):
        cache.set("p", "r")
        cache.get("p")   # hit
        cache.get("p")   # hit
        cache.get("q")   # miss
        stats = cache.stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.hit_rate == pytest.approx(2 / 3)

    def test_hit_rate_zero_for_all_misses(self, cache):
        cache.get("a")
        cache.get("b")
        stats = cache.stats()
        assert stats.hit_rate == 0.0


class TestInvalidateAndClear:
    def test_invalidate_removes_entry(self, cache):
        cache.set("x", "y")
        result = cache.invalidate("x")
        assert result is True
        assert cache.get("x") is None

    def test_invalidate_missing_returns_false(self, cache):
        assert cache.invalidate("nonexistent") is False

    def test_clear_resets_everything(self, cache):
        cache.set("a", "1")
        cache.get("a")
        cache.clear()
        assert len(cache) == 0
        stats = cache.stats()
        assert stats.hits == 0
        assert stats.misses == 0

    def test_invalid_max_size(self):
        with pytest.raises(ValueError, match="max_size"):
            GenerationCache(max_size=0)


class TestKeyHashing:
    def test_hash_mode_default(self, cache):
        cache.set("prompt", "output")
        assert cache.get("prompt") == "output"

    def test_no_hash_mode(self):
        cache = GenerationCache(max_size=10, key_hash=False)
        cache.set("raw_prompt", "result")
        assert cache.get("raw_prompt") == "result"

    def test_repr(self, cache):
        r = repr(cache)
        assert "GenerationCache" in r

    def test_stats_repr(self, cache):
        s = repr(cache.stats())
        assert "CacheStats" in s
