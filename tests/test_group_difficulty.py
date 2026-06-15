"""Tests for summarize_group_difficulty — DAPO dynamic-sampling skip rate as a
curriculum difficulty signal (pure, framework-free, MAIN-SAFE).
"""

from __future__ import annotations

import pytest

from cola_coder.reasoning.grpo import GroupDifficultyStats, summarize_group_difficulty


def test_empty_window_is_none() -> None:
    s = summarize_group_difficulty([], [])
    assert s == GroupDifficultyStats(0, 0.0, 0.0, "none")


def test_low_skip_is_informative() -> None:
    # Healthy: groups have reward variance (few skips), middling pass-rate.
    s = summarize_group_difficulty([0.5, 0.4, 0.6, 0.5], [False, False, False, True])
    assert s.signal == "informative"
    assert s.degenerate_fraction == pytest.approx(0.25)
    assert s.mean_pass_rate == pytest.approx(0.5)


def test_high_skip_high_pass_is_too_easy() -> None:
    # Most steps skipped (zero variance) AND nearly everything passes → too easy.
    s = summarize_group_difficulty([1.0, 1.0, 1.0, 0.9], [True, True, True, False])
    assert s.signal == "too_easy"
    assert s.degenerate_fraction == pytest.approx(0.75)


def test_high_skip_low_pass_is_too_hard() -> None:
    s = summarize_group_difficulty([0.0, 0.0, 0.1, 0.0], [True, True, False, True])
    assert s.signal == "too_hard"


def test_high_skip_middling_pass_is_mixed() -> None:
    s = summarize_group_difficulty([0.5, 0.5, 0.5, 0.5], [True, True, True, False])
    assert s.signal == "mixed"


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        summarize_group_difficulty([0.5, 0.5], [True])


def test_thresholds_are_tunable() -> None:
    # With a stricter high_skip, a 0.5 skip rate is no longer "high" → informative.
    pass_rates = [1.0, 1.0]
    skipped = [True, False]  # degenerate_fraction = 0.5
    assert summarize_group_difficulty(pass_rates, skipped, high_skip=0.9).signal == "informative"
    assert summarize_group_difficulty(pass_rates, skipped, high_skip=0.5).signal == "too_easy"
