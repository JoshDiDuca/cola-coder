"""Tests for DAPO soft overlong reward shaping (arXiv:2503.14476).

The penalty is a pure, deterministic function of (length, max_length,
soft_buffer): 0 below the buffer, a linear ramp 0 -> -1 across the buffer, and
-1 at/over max_length. These tests pin the exact shape (endpoints, midpoint,
quarter points), monotonicity, argument validation, and the GRPO wiring's
default-safe / opt-in behavior.
"""

import pytest

from cola_coder.reasoning.rewards.overlong import (
    apply_overlong_shaping,
    soft_overlong_penalty,
)
from cola_coder.reasoning.grpo import apply_overlong_shaping_rewards


# --------------------------------------------------------------------------- #
# soft_overlong_penalty — exact shape
# --------------------------------------------------------------------------- #
def test_zero_penalty_well_below_buffer():
    # max_length=200, soft_buffer=50 -> ramp starts at 150.
    assert soft_overlong_penalty(0, 200, 50) == 0.0
    assert soft_overlong_penalty(100, 200, 50) == 0.0


def test_zero_penalty_exactly_at_ramp_start():
    # length == max_length - soft_buffer is the last fully-unpenalised length.
    assert soft_overlong_penalty(150, 200, 50) == 0.0


def test_full_penalty_at_max_length():
    assert soft_overlong_penalty(200, 200, 50) == -1.0


def test_full_penalty_over_max_length():
    assert soft_overlong_penalty(250, 200, 50) == -1.0
    assert soft_overlong_penalty(10_000, 200, 50) == -1.0


def test_linear_ramp_at_midpoint():
    # Halfway through the buffer (length 175) -> -0.5.
    assert soft_overlong_penalty(175, 200, 50) == pytest.approx(-0.5)


def test_linear_ramp_at_quarter_points():
    # 25% into the buffer (length 162.5 -> use 163 region via exact ints):
    # at length 162 -> over=12, -12/50 = -0.24
    assert soft_overlong_penalty(162, 200, 50) == pytest.approx(-0.24)
    # 75% into the buffer: length 188 -> over=38, -38/50 = -0.76
    assert soft_overlong_penalty(188, 200, 50) == pytest.approx(-0.76)


def test_penalty_is_in_unit_interval():
    for length in range(0, 260, 7):
        p = soft_overlong_penalty(length, 200, 50)
        assert -1.0 <= p <= 0.0


def test_monotonic_non_increasing_in_length():
    prev = soft_overlong_penalty(0, 200, 50)
    for length in range(1, 260):
        cur = soft_overlong_penalty(length, 200, 50)
        assert cur <= prev + 1e-12, f"penalty rose at length={length}"
        prev = cur


def test_negative_length_treated_as_zero_penalty():
    assert soft_overlong_penalty(-5, 200, 50) == 0.0


def test_zero_soft_buffer_is_a_step_at_max():
    # soft_buffer == 0: no ramp, only the at/over-max full penalty.
    assert soft_overlong_penalty(199, 200, 0) == 0.0
    assert soft_overlong_penalty(200, 200, 0) == -1.0
    assert soft_overlong_penalty(201, 200, 0) == -1.0


def test_full_buffer_ramps_from_zero():
    # soft_buffer == max_length: ramp starts at length 0.
    assert soft_overlong_penalty(0, 100, 100) == 0.0
    assert soft_overlong_penalty(50, 100, 100) == pytest.approx(-0.5)
    assert soft_overlong_penalty(100, 100, 100) == -1.0


# --------------------------------------------------------------------------- #
# soft_overlong_penalty — argument validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_max", [0, -1, -100])
def test_raises_on_nonpositive_max_length(bad_max):
    with pytest.raises(ValueError):
        soft_overlong_penalty(10, bad_max, 0)


def test_raises_on_negative_soft_buffer():
    with pytest.raises(ValueError):
        soft_overlong_penalty(10, 200, -1)


def test_raises_when_soft_buffer_exceeds_max_length():
    with pytest.raises(ValueError):
        soft_overlong_penalty(10, 200, 201)


# --------------------------------------------------------------------------- #
# apply_overlong_shaping — reward arithmetic
# --------------------------------------------------------------------------- #
def test_apply_no_change_below_buffer():
    assert apply_overlong_shaping(1.0, 100, 200, 50) == 1.0


def test_apply_subtracts_exact_penalty_at_midpoint():
    # midpoint penalty -0.5 -> reward 1.0 becomes 0.5
    assert apply_overlong_shaping(1.0, 175, 200, 50) == pytest.approx(0.5)


def test_apply_full_penalty_at_max():
    assert apply_overlong_shaping(1.0, 200, 200, 50) == pytest.approx(0.0)
    assert apply_overlong_shaping(0.0, 200, 200, 50) == pytest.approx(-1.0)


def test_scale_scales_the_penalty():
    # scale=2 doubles the (negative) penalty at the midpoint: -0.5 -> -1.0
    assert apply_overlong_shaping(1.0, 175, 200, 50, scale=2.0) == pytest.approx(0.0)
    # scale=0.5 halves it: -0.5 -> -0.25
    assert apply_overlong_shaping(1.0, 175, 200, 50, scale=0.5) == pytest.approx(0.75)


def test_scale_zero_is_noop():
    for length in (160, 175, 200, 250):
        assert apply_overlong_shaping(1.0, length, 200, 50, scale=0.0) == 1.0


def test_zero_buffer_noop_for_in_budget_lengths():
    # soft_buffer == 0 only bites at/over max_length.
    assert apply_overlong_shaping(1.0, 199, 200, 0) == 1.0
    assert apply_overlong_shaping(1.0, 200, 200, 0) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# GRPO group helper — opt-in / default-safe wiring
# --------------------------------------------------------------------------- #
def test_group_shaping_disabled_when_scale_zero_is_noop():
    rewards = [1.0, 0.0, 1.0]
    lengths = [300, 300, 300]  # all overlong
    shaped, n = apply_overlong_shaping_rewards(
        rewards, lengths, max_length=200, soft_buffer=50, scale=0.0
    )
    assert shaped == rewards
    assert n == 0


def test_group_shaping_disabled_when_buffer_zero_is_noop_for_in_budget():
    rewards = [1.0, 1.0]
    lengths = [180, 190]  # in budget
    shaped, n = apply_overlong_shaping_rewards(
        rewards, lengths, max_length=200, soft_buffer=0, scale=1.0
    )
    assert shaped == rewards
    assert n == 0


def test_group_shaping_reduces_overlong_reward():
    rewards = [1.0, 1.0, 1.0]
    # in-budget, midpoint of ramp, at-max
    lengths = [100, 175, 200]
    shaped, n = apply_overlong_shaping_rewards(
        rewards, lengths, max_length=200, soft_buffer=50, scale=1.0
    )
    assert shaped[0] == pytest.approx(1.0)   # unchanged (in budget)
    assert shaped[1] == pytest.approx(0.5)   # midpoint -0.5
    assert shaped[2] == pytest.approx(0.0)   # full -1.0
    assert n == 2  # two solutions actually changed


def test_group_shaping_preserves_order_and_length():
    rewards = [0.3, 0.7, 0.9, 0.1]
    lengths = [10, 250, 175, 999]
    shaped, _ = apply_overlong_shaping_rewards(
        rewards, lengths, max_length=200, soft_buffer=50, scale=1.0
    )
    assert len(shaped) == len(rewards)
    assert shaped[0] == pytest.approx(0.3)  # in budget, untouched
