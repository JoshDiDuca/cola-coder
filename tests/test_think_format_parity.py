"""MODEL-010: the SFT-warmup CoT format and the GRPO reward's format bonus must
agree on what 'correct reasoning format' is.

format_thinking_example() builds the SFT training text; the GRPO reward gives a
format bonus for think-FIRST-then-code output. If those two definitions drift,
the model is trained on one shape and rewarded for another (silent training
inconsistency — the format-parity class: INFER-011/BUG-110). Both now route
through the single is_think_first_format() predicate; these tests lock it and the
cross-module consistency.
"""

from cola_coder.reasoning.thinking_tokens import (
    THINK_OPEN,
    THINK_CLOSE,
    format_thinking_example,
    is_think_first_format,
)
from cola_coder.reasoning.cot_data import get_cot_training_data


class TestIsThinkFirstFormat:
    def test_canonical_format_accepted(self):
        assert is_think_first_format(format_thinking_example("reason", "code = 1")) is True

    def test_leading_whitespace_ok(self):
        assert is_think_first_format("\n  " + format_thinking_example("r", "c=1")) is True

    def test_code_before_think_rejected(self):
        # BUG-102: code BEFORE the thinking must NOT count as think-first.
        assert is_think_first_format("def f(): pass\n<think>after</think>\n  ") is False

    def test_no_code_after_close_rejected(self):
        assert is_think_first_format(f"{THINK_OPEN}reason{THINK_CLOSE}   ") is False

    def test_missing_tags_rejected(self):
        assert is_think_first_format("just some code, no thinking") is False

    def test_only_open_tag_rejected(self):
        assert is_think_first_format(f"{THINK_OPEN}unterminated reasoning") is False


class TestCotMatchesRewardFormat:
    def test_python_cot_examples_are_think_first(self):
        data = get_cot_training_data("python")
        assert data, "expected built-in python CoT examples"
        for ex in data:
            assert THINK_OPEN in ex["text"] and THINK_CLOSE in ex["text"]
            # The reasoning-bearing portion (the completion the model learns to
            # produce) must satisfy the reward's think-first predicate.
            think_chunk = ex["text"][ex["text"].index(THINK_OPEN):]
            assert is_think_first_format(think_chunk), ex.get("task_id", "?")

    def test_typescript_cot_examples_are_think_first(self):
        for ex in get_cot_training_data("typescript"):
            think_chunk = ex["text"][ex["text"].index(THINK_OPEN):]
            assert is_think_first_format(think_chunk)
