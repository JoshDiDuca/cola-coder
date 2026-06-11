"""BUG-110 / REWARD-001: TypeScript & combined GRPO rewards must be thinking-aware.

The `python_exec` reward strips `<think>...</think>` (via extract_thinking) before
executing the code. The registry's `typescript` and `combined` wrappers did NOT —
they passed the RAW generation to the scorer. So on the recommended TS reasoning
path (`--reward typescript`), tsc treated the thinking tags as TypeScript, flagged
a syntax error, and zeroed the reward on EVERY reasoning-formatted generation →
no reward variance → GRPO's collapse-guard skips the update → training stalls.

These tests stub the scorers (so no tsc/Node is needed) to capture the exact code
they receive and to verify the thinking-trace is stripped first, plus the shared
thinking-length penalty (REWARD-001) now applies on the TS/combined paths too.
"""

import pytest

import cola_coder.reasoning.rewards.type_check as tc_mod
import cola_coder.reasoning.rewards.combined as cmb_mod
from cola_coder.reasoning.reward import thinking_length_penalty
from cola_coder.reasoning.reward_registry import RewardRegistry


class TestThinkingLengthPenaltyHelper:
    def test_under_and_at_budget_no_penalty(self):
        assert thinking_length_penalty(5, 10) == 0.0
        assert thinking_length_penalty(10, 10) == 0.0

    def test_excess_is_penalized_linearly(self):
        assert thinking_length_penalty(20, 10) == pytest.approx(0.01)  # 10 * 0.001

    def test_penalty_is_capped(self):
        assert thinking_length_penalty(10_000, 10) == 0.1


class TestTypeScriptRewardThinkingAware:
    def test_strips_thinking_before_scoring(self, monkeypatch):
        captured = []
        monkeypatch.setattr(
            tc_mod.TypeCheckReward, "score",
            lambda self, code: (captured.append(code) or 1.0),
        )
        monkeypatch.setattr(tc_mod.TypeCheckReward, "detailed_score", lambda self, code: {})
        fn = RewardRegistry.get("typescript")
        gen = "<think>let me reason about the types here</think>const x: number = 1;"
        rewards, infos = fn([gen], "")
        # The scorer must NEVER see the thinking tags — only the answer code.
        assert "<think>" not in captured[0]
        assert "const x: number = 1;" in captured[0]
        assert rewards[0] == 1.0
        assert infos[0]["thinking_length"] > 0

    def test_no_thinking_is_unchanged(self, monkeypatch):
        captured = []
        monkeypatch.setattr(
            tc_mod.TypeCheckReward, "score",
            lambda self, code: (captured.append(code) or 1.0),
        )
        monkeypatch.setattr(tc_mod.TypeCheckReward, "detailed_score", lambda self, code: {})
        fn = RewardRegistry.get("typescript")
        rewards, infos = fn(["const x: number = 1;"], "")
        assert captured[0] == "const x: number = 1;"
        assert rewards[0] == 1.0
        assert infos[0]["thinking_length"] == 0
        assert infos[0]["length_penalty"] == 0.0

    def test_applies_thinking_length_penalty(self, monkeypatch):
        monkeypatch.setattr(tc_mod.TypeCheckReward, "score", lambda self, code: 1.0)
        monkeypatch.setattr(tc_mod.TypeCheckReward, "detailed_score", lambda self, code: {})
        fn = RewardRegistry.get("typescript")
        gen = f"<think>{'word ' * 50}</think>const x = 1;"
        rewards, infos = fn([gen], "", max_thinking_tokens=10)
        # 50 words, budget 10 → excess 40 → penalty 0.04 → 1.0 - 0.04
        assert rewards[0] == pytest.approx(0.96)
        assert infos[0]["length_penalty"] == pytest.approx(-0.04)
        assert infos[0]["thinking_length"] == 50

    def test_reward_stays_in_unit_range(self, monkeypatch):
        # A failing score (-0.5) clamped to 0, then penalty must not go negative.
        monkeypatch.setattr(tc_mod.TypeCheckReward, "score", lambda self, code: -0.5)
        monkeypatch.setattr(tc_mod.TypeCheckReward, "detailed_score", lambda self, code: {})
        fn = RewardRegistry.get("typescript")
        gen = f"<think>{'word ' * 50}</think>broken("
        rewards, _ = fn([gen], "", max_thinking_tokens=10)
        assert rewards[0] == 0.0


class TestCombinedRewardThinkingAware:
    def test_strips_thinking_before_scoring(self, monkeypatch):
        captured = []

        def fake_detailed(self, code, context=None):
            captured.append(code)
            return {"combined_score": 1.0}

        monkeypatch.setattr(cmb_mod.CombinedReward, "detailed_score", fake_detailed)
        fn = RewardRegistry.get("combined")
        gen = "<think>reasoning about the impl</think>function f(): void {}"
        rewards, infos = fn([gen], "")
        assert "<think>" not in captured[0]
        assert "function f(): void {}" in captured[0]
        assert rewards[0] == 1.0
        assert infos[0]["thinking_length"] > 0

    def test_applies_thinking_length_penalty(self, monkeypatch):
        monkeypatch.setattr(
            cmb_mod.CombinedReward, "detailed_score",
            lambda self, code, context=None: {"combined_score": 1.0},
        )
        fn = RewardRegistry.get("combined")
        gen = f"<think>{'tok ' * 30}</think>const y = 2;"
        rewards, infos = fn([gen], "", max_thinking_tokens=10)
        # 30 words, budget 10 → excess 20 → penalty 0.02
        assert rewards[0] == pytest.approx(0.98)
        assert infos[0]["length_penalty"] == pytest.approx(-0.02)
