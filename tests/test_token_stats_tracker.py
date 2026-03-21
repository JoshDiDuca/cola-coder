"""Tests for TokenStatsTracker (features/token_stats_tracker.py)."""

from __future__ import annotations

import pytest

from cola_coder.features.token_stats_tracker import (
    TokenStats,
    TokenStatsTracker,
)


@pytest.fixture()
def tracker() -> TokenStatsTracker:
    return TokenStatsTracker()


class TestIsEnabled:
    def test_feature_enabled(self):
        from cola_coder.features.token_stats_tracker import FEATURE_ENABLED, is_enabled

        assert FEATURE_ENABLED is True
        assert is_enabled() is True


class TestRecord:
    def test_record_returns_stats(self, tracker):
        stats = tracker.record([1, 2, 3, 4, 5])
        assert isinstance(stats, TokenStats)
        assert stats.total_tokens == 5
        assert stats.unique_tokens == 5

    def test_record_empty_list(self, tracker):
        stats = tracker.record([])
        assert stats.total_tokens == 0
        assert stats.unique_tokens == 0
        assert stats.type_token_ratio == 0.0

    def test_all_same_tokens(self, tracker):
        stats = tracker.record([42, 42, 42, 42])
        assert stats.unique_tokens == 1
        assert stats.type_token_ratio == pytest.approx(0.25)
        assert stats.repeated_fraction == 1.0

    def test_all_unique_tokens(self, tracker):
        tokens = list(range(100))
        stats = tracker.record(tokens)
        assert stats.type_token_ratio == pytest.approx(1.0)
        assert stats.repeated_fraction == 0.0

    def test_type_token_ratio_range(self, tracker):
        stats = tracker.record([1, 1, 2, 3])
        assert 0.0 <= stats.type_token_ratio <= 1.0

    def test_top_tokens_sorted(self, tracker):
        stats = tracker.record([1, 1, 1, 2, 2, 3])
        assert stats.top_tokens[0][0] == 1  # most frequent first
        assert stats.top_tokens[0][1] == 3


class TestSummary:
    def test_empty_summary(self, tracker):
        agg = tracker.summary()
        assert agg.num_sequences == 0
        assert agg.total_tokens == 0
        assert agg.global_unique_tokens == 0

    def test_summary_after_records(self, tracker):
        tracker.record([1, 2, 3])
        tracker.record([4, 5, 6, 7])
        agg = tracker.summary()
        assert agg.num_sequences == 2
        assert agg.total_tokens == 7
        assert agg.global_unique_tokens == 7

    def test_avg_tokens_per_sequence(self, tracker):
        tracker.record([1, 2, 3])  # 3 tokens
        tracker.record([4, 5])     # 2 tokens
        agg = tracker.summary()
        assert agg.avg_tokens_per_sequence == pytest.approx(2.5)

    def test_global_ttr(self, tracker):
        tracker.record([1, 1, 2, 2])  # 2 unique / 4 total
        agg = tracker.summary()
        assert 0.0 <= agg.global_type_token_ratio <= 1.0


class TestDiversityScore:
    def test_diversity_empty(self, tracker):
        assert tracker.diversity_score() == 0.0

    def test_high_diversity(self, tracker):
        # 100 unique tokens
        tracker.record(list(range(100)))
        score = tracker.diversity_score()
        assert score > 0.5

    def test_low_diversity(self, tracker):
        # All same token
        tracker.record([1] * 100)
        score = tracker.diversity_score()
        assert score < 0.5

    def test_diversity_score_range(self, tracker):
        tracker.record([1, 2, 1, 3, 2, 4])
        score = tracker.diversity_score()
        assert 0.0 <= score <= 1.0


class TestResetAndCount:
    def test_reset_clears_data(self, tracker):
        tracker.record([1, 2, 3])
        tracker.reset()
        agg = tracker.summary()
        assert agg.num_sequences == 0
        assert tracker.diversity_score() == 0.0

    def test_num_sequences(self, tracker):
        assert tracker.num_sequences() == 0
        tracker.record([1, 2])
        tracker.record([3, 4])
        assert tracker.num_sequences() == 2

    def test_repr(self, tracker):
        tracker.record([1, 2, 3])
        r = repr(tracker)
        assert "TokenStatsTracker" in r
        assert "sequences=1" in r
