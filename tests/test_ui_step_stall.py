"""OPS-003: step-stall telemetry — seconds since the training step last advanced.

Pure cross-poll logic (the dashboard recomputes status every ~1s); surfaces the
hung-vs-slow signal the babysitting runbook uses, made visible. No GPU/training.
"""

from __future__ import annotations

from cola_coder.ui.status import _step_stall_seconds


def test_none_step_before_any_observation() -> None:
    cache: dict[str, float | None] = {"max_step": None, "since": None}
    assert _step_stall_seconds(None, now=100.0, cache=cache) is None


def test_first_step_is_zero_stall() -> None:
    cache: dict[str, float | None] = {"max_step": None, "since": None}
    assert _step_stall_seconds(50, now=100.0, cache=cache) == 0.0
    assert cache["max_step"] == 50.0
    assert cache["since"] == 100.0


def test_advance_resets_stall_to_zero() -> None:
    cache: dict[str, float | None] = {"max_step": None, "since": None}
    _step_stall_seconds(50, now=100.0, cache=cache)
    # Same step later → stall grows.
    assert _step_stall_seconds(50, now=130.0, cache=cache) == 30.0
    # Step advances → stall back to 0, timer reset.
    assert _step_stall_seconds(51, now=140.0, cache=cache) == 0.0
    assert _step_stall_seconds(51, now=155.0, cache=cache) == 15.0


def test_stall_accumulates_while_step_unchanged() -> None:
    cache: dict[str, float | None] = {"max_step": None, "since": None}
    _step_stall_seconds(200, now=0.0, cache=cache)
    assert _step_stall_seconds(200, now=600.0, cache=cache) == 600.0
    assert _step_stall_seconds(200, now=2700.0, cache=cache) == 2700.0


def test_restart_to_lower_step_counts_as_progress() -> None:
    cache: dict[str, float | None] = {"max_step": None, "since": None}
    _step_stall_seconds(14000, now=0.0, cache=cache)
    # Resume from an earlier checkpoint → lower step → treated as a fresh advance.
    assert _step_stall_seconds(13500, now=50.0, cache=cache) == 0.0
    assert cache["max_step"] == 13500.0


def test_none_step_after_observation_reports_from_last_advance() -> None:
    cache: dict[str, float | None] = {"max_step": None, "since": None}
    _step_stall_seconds(10, now=100.0, cache=cache)
    # A poll with no parseable step still reports the stall from the last advance.
    assert _step_stall_seconds(None, now=160.0, cache=cache) == 60.0
