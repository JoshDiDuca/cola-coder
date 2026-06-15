"""DAPO soft overlong reward shaping (length-aware penalty for GRPO).

DAPO (Yu et al. 2025, *DAPO: An Open-Source LLM RL System at Scale*,
arXiv:2503.14476, §"Soft Overlong Punishment") observes that abruptly
truncated overlong generations inject NOISE into the reward signal: a long
response is not necessarily wrong, yet a hard length cut-off gives it a sharp,
arbitrary reward. The fix is a *soft* length-aware penalty that ramps the
reward down linearly inside a buffer zone just below the maximum length and
saturates at a full penalty at/above the maximum.

This module is a PURE, deterministic utility — no model, no GPU, no I/O. It
composes with any existing GRPO reward (python_exec / typescript / combined):
the caller computes the functional reward, then subtracts ``scale *
soft_overlong_penalty(length, max_length, soft_buffer)`` per solution before
group advantages are taken. It is strictly OPT-IN; with shaping disabled the
GRPO reward path is byte-for-byte unchanged.

Shape of the penalty (DAPO):

    length <= max_length - soft_buffer   ->  0.0        (no penalty)
    max_length - soft_buffer < length    ->  linear ramp 0.0 -> -1.0
        < max_length
    length >= max_length                 -> -1.0        (full penalty)
"""

__all__ = ["soft_overlong_penalty", "apply_overlong_shaping"]


def soft_overlong_penalty(length: int, max_length: int, soft_buffer: int) -> float:
    """DAPO soft overlong punishment for a single response length.

    The penalty is 0.0 while the response stays at least ``soft_buffer`` tokens
    short of ``max_length``, then ramps LINEARLY from 0.0 down to -1.0 across the
    buffer zone, and saturates at -1.0 once the response reaches/exceeds
    ``max_length`` (DAPO, arXiv:2503.14476, §"Soft Overlong Punishment").

    Args:
        length: Response length (e.g. number of generated tokens). Negative
            values are treated as 0 (no shorter-than-empty response exists).
        max_length: The generation length budget (must be > 0).
        soft_buffer: Width of the linear-ramp zone just below ``max_length``,
            in the same unit as ``length`` (must satisfy
            ``0 <= soft_buffer <= max_length``).

    Returns:
        A penalty in the closed interval [-1.0, 0.0]: 0.0 for an in-budget
        response, -1.0 for one at/over the budget, and a linear interpolation
        in between. Monotonic non-increasing in ``length``.

    Raises:
        ValueError: If ``max_length <= 0`` or ``soft_buffer`` is outside
            ``[0, max_length]``.

    Example:
        >>> soft_overlong_penalty(100, max_length=200, soft_buffer=50)
        0.0
        >>> soft_overlong_penalty(175, max_length=200, soft_buffer=50)
        -0.5
        >>> soft_overlong_penalty(200, max_length=200, soft_buffer=50)
        -1.0
    """
    if max_length <= 0:
        raise ValueError(f"max_length must be > 0, got {max_length}")
    if soft_buffer < 0 or soft_buffer > max_length:
        raise ValueError(
            f"soft_buffer must satisfy 0 <= soft_buffer <= max_length "
            f"({max_length}), got {soft_buffer}"
        )

    # Threshold below which there is no penalty at all.
    ramp_start = max_length - soft_buffer

    # Full-penalty check FIRST: when soft_buffer == 0, ramp_start == max_length,
    # and `length == max_length` must saturate to -1.0 (not slip through the
    # `length <= ramp_start` no-penalty branch). Ordering full-penalty before
    # no-penalty makes the at/over-max case win the boundary tie.
    if length >= max_length:
        return -1.0
    if length <= ramp_start:
        return 0.0

    # Only reached when soft_buffer > 0 (ramp_start < length < max_length), so
    # the division below never hits a zero divisor.
    over = length - ramp_start
    return -float(over) / float(soft_buffer)


def apply_overlong_shaping(
    reward: float,
    length: int,
    max_length: int,
    soft_buffer: int,
    scale: float = 1.0,
) -> float:
    """Apply DAPO soft overlong shaping to a single reward value.

    Returns ``reward + scale * soft_overlong_penalty(length, max_length,
    soft_buffer)``. Because the penalty is <= 0, this can only reduce (or leave
    unchanged) the reward — a longer response is smoothly down-weighted as it
    approaches the length budget.

    Pure and deterministic. With ``soft_buffer == 0`` it is a no-op for every
    in-budget length (penalty kicks in only at/over ``max_length``); with
    ``scale == 0.0`` it is always a no-op.

    Args:
        reward: The functional reward for this solution (any real value).
        length: Response length (e.g. number of generated tokens).
        max_length: The generation length budget (must be > 0).
        soft_buffer: Width of the linear-ramp zone (0 <= soft_buffer <= max_length).
        scale: Multiplier on the penalty magnitude (default 1.0). The penalty
            already lives in [-1.0, 0.0]; ``scale`` sets how many reward points a
            fully-overlong response loses.

    Returns:
        The shaped reward.

    Raises:
        ValueError: Propagated from :func:`soft_overlong_penalty` on bad args.
    """
    penalty = soft_overlong_penalty(length, max_length, soft_buffer)
    return reward + scale * penalty
