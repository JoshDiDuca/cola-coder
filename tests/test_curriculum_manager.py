"""Tests for curriculum_manager.py (feature 48)."""

from __future__ import annotations


from cola_coder.features.curriculum_manager import (
    FEATURE_ENABLED,
    CurriculumManager,
    CurriculumStats,
    CurriculumStrategy,
    is_enabled,
    make_scored_samples,
)


def test_feature_enabled():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_samples(n: int) -> list:
    """Create n samples with evenly-spaced difficulties 0 → 1."""
    return make_scored_samples(
        data=list(range(n)),
        difficulties=[i / max(n - 1, 1) for i in range(n)],
    )


# ---------------------------------------------------------------------------
# make_scored_samples
# ---------------------------------------------------------------------------


def test_make_scored_samples():
    samples = make_scored_samples([1, 2, 3], [0.1, 0.5, 0.9])
    assert len(samples) == 3
    assert samples[0].data == 1
    assert samples[0].difficulty == 0.1
    assert samples[0].index == 0


# ---------------------------------------------------------------------------
# Easy-to-hard strategy
# ---------------------------------------------------------------------------


def test_easy_to_hard_starts_with_easy():
    # Use a large pool (200) so that even at low ceiling there are >= batch_size
    # easy samples (no padding from hard samples needed).
    samples = _make_samples(200)
    manager = CurriculumManager(
        samples,
        strategy=CurriculumStrategy.EASY_TO_HARD,
        warmup_steps=1000,
        total_steps=5000,
    )
    # Step 0: ceiling = 0.05 → samples with diff in [0, 0.05] ≈ 10 samples
    # Enough to fill batch without padding from hard samples
    batch = manager.next_batch(batch_size=5, step=0)
    max_diff = max(s.difficulty for s in batch)
    assert max_diff <= 0.06  # ceiling at step=0 is ~0.05


def test_easy_to_hard_full_access_after_warmup():
    samples = _make_samples(50)
    manager = CurriculumManager(
        samples,
        strategy=CurriculumStrategy.EASY_TO_HARD,
        warmup_steps=100,
        total_steps=1000,
    )
    # After warmup, ceiling = 1.0
    batch = manager.next_batch(batch_size=20, step=100)
    max_diff = max(s.difficulty for s in batch)
    assert max_diff > 0.5  # should include hard samples


# ---------------------------------------------------------------------------
# Hard-to-easy strategy
# ---------------------------------------------------------------------------


def test_hard_to_easy_starts_with_hard():
    # Large pool so hard-end has >= batch_size samples at step=0
    samples = _make_samples(200)
    manager = CurriculumManager(
        samples,
        strategy=CurriculumStrategy.HARD_TO_EASY,
        warmup_steps=1000,
        total_steps=5000,
    )
    # Step 0: floor = 1 - 0.05 = 0.95 → samples with diff >= 0.95
    batch = manager.next_batch(batch_size=5, step=0)
    min_diff = min(s.difficulty for s in batch)
    assert min_diff >= 0.94  # starts with hardest (floor ~0.95)


# ---------------------------------------------------------------------------
# Mixed strategy
# ---------------------------------------------------------------------------


def test_mixed_contains_both_easy_and_hard():
    samples = _make_samples(100)
    manager = CurriculumManager(
        samples,
        strategy=CurriculumStrategy.MIXED,
        total_steps=1000,
        mixed_easy_ratio=0.8,
    )
    # Collect many batches to see both easy and hard
    all_diffs = []
    for step in range(20):
        batch = manager.next_batch(batch_size=10, step=step)
        all_diffs.extend(s.difficulty for s in batch)
    assert min(all_diffs) < 0.3  # some easy
    assert max(all_diffs) > 0.7  # some hard


# ---------------------------------------------------------------------------
# Random strategy
# ---------------------------------------------------------------------------


def test_random_strategy_uses_full_pool():
    samples = _make_samples(100)
    manager = CurriculumManager(samples, strategy=CurriculumStrategy.RANDOM)
    all_diffs = set()
    for step in range(100):
        batch = manager.next_batch(batch_size=10, step=step)
        all_diffs.update(round(s.difficulty, 2) for s in batch)
    # Should have samples from both ends of the difficulty range
    assert min(all_diffs) < 0.1
    assert max(all_diffs) > 0.9


# ---------------------------------------------------------------------------
# Tracking & stats
# ---------------------------------------------------------------------------


def test_num_seen_increases():
    samples = _make_samples(20)
    manager = CurriculumManager(
        samples, strategy=CurriculumStrategy.RANDOM, warmup_steps=0
    )
    manager.next_batch(batch_size=5, step=0)
    assert manager.num_seen > 0


def test_reset_clears_seen():
    samples = _make_samples(20)
    manager = CurriculumManager(samples, strategy=CurriculumStrategy.RANDOM)
    manager.next_batch(batch_size=10, step=0)
    manager.reset()
    assert manager.num_seen == 0


def test_stats_returns_curriculum_stats():
    samples = _make_samples(30)
    manager = CurriculumManager(samples, strategy=CurriculumStrategy.EASY_TO_HARD)
    manager.next_batch(batch_size=5, step=10)
    stats = manager.stats(step=10)
    assert isinstance(stats, CurriculumStats)
    assert stats.total_samples == 30
    assert stats.strategy == "easy_to_hard"


def test_stats_summary_format():
    samples = _make_samples(10)
    manager = CurriculumManager(samples, strategy=CurriculumStrategy.MIXED)
    stats = manager.stats(step=5)
    s = stats.summary()
    assert "step=5" in s
    assert "strategy=mixed" in s


def test_band_coverage_keys():
    samples = _make_samples(50)
    manager = CurriculumManager(samples, num_bands=5)
    manager.next_batch(batch_size=10, step=50)
    stats = manager.stats()
    assert len(stats.band_coverage) == 5


def test_batch_size_larger_than_pool():
    """Should still return batch_size items (with repetition)."""
    samples = _make_samples(5)
    manager = CurriculumManager(
        samples,
        strategy=CurriculumStrategy.EASY_TO_HARD,
        warmup_steps=0,
    )
    batch = manager.next_batch(batch_size=20, step=1000)
    assert len(batch) == 20
