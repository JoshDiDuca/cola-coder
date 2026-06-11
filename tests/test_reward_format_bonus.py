"""BUG-109: the GRPO format bonus must reward think-FIRST-then-code structure.

`compute_reward` (reasoning/reward.py — the default `python_exec` reward) grants
a +0.1 "proper <think> format" bonus. The old check (`think_end < len(text) - 10`)
only verified that 10+ characters — whitespace included — followed </think>; it
never confirmed thinking came BEFORE the code, despite the comment saying so. So
`code<think>...</think>   ` (code first, then trailing spaces) wrongly earned the
bonus and the model got no signal that reasoning must precede the answer.

These tests isolate the format-bonus logic by stubbing out execute_code (the
correctness path runs real subprocesses; the bonus is independent of pass/fail),
and lock the think-first + code-follows requirement.
"""

import pytest

import cola_coder.reasoning.reward as reward_mod
from cola_coder.reasoning.reward import compute_reward
from cola_coder.reasoning.thinking_tokens import THINK_OPEN, THINK_CLOSE


@pytest.fixture(autouse=True)
def _no_execution(monkeypatch):
    # The format bonus does not depend on whether the code runs; stub execution
    # so the test is fast, deterministic, and never runs code on the host.
    monkeypatch.setattr(reward_mod, "execute_code", lambda code, timeout=10.0: (False, ""))


def _bonus(text: str) -> float:
    _, info = compute_reward(text, test_code="assert True")
    return info["format_bonus"]


class TestFormatBonus:
    def test_proper_think_first_then_code_earns_bonus(self):
        text = f"{THINK_OPEN}reason about it{THINK_CLOSE}\ndef solution():\n    return 42\n"
        assert _bonus(text) == 0.1

    def test_leading_whitespace_before_think_still_counts(self):
        text = f"  \n{THINK_OPEN}reason{THINK_CLOSE}\nreturn 0\n"
        assert _bonus(text) == 0.1

    def test_code_before_thinking_is_not_rewarded(self):
        # The core bug: code first, thinking after, trailing spaces. Old check
        # granted the bonus; it must not.
        text = f"def f():\n    return 1\n{THINK_OPEN}post-hoc reasoning{THINK_CLOSE}   "
        assert _bonus(text) == 0.0

    def test_think_first_but_no_code_after_is_not_rewarded(self):
        # Only whitespace after </think> — there is no answer to reward.
        text = f"{THINK_OPEN}reason{THINK_CLOSE}   \n  "
        assert _bonus(text) == 0.0

    def test_no_thinking_tokens_no_bonus(self):
        assert _bonus("def f():\n    return 1\n") == 0.0

    def test_only_open_tag_no_bonus(self):
        # Malformed: open without a matching close.
        assert _bonus(f"{THINK_OPEN}reasoning with no close\ndef f(): return 1") == 0.0
