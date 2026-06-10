"""Tests for inference/text_utils.strip_prompt_prefix (INFER-001).

The non-streaming server endpoints used `result[len(prompt):] if
result.startswith(prompt) else result`, which leaks the WHOLE prompt when
BPE decode(encode(prompt)) isn't byte-identical to the prompt. The shared
helper strips the longest common prefix instead, so a mismatch costs at most
a few boundary characters — never the entire prompt.
"""

from cola_coder.inference.text_utils import strip_prompt_prefix


class TestStripPromptPrefix:
    def test_exact_prefix(self):
        assert strip_prompt_prefix("def f():\n    pass", "def f():\n") == "    pass"

    def test_empty_prompt_returns_all(self):
        assert strip_prompt_prefix("hello world", "") == "hello world"

    def test_bpe_mismatch_does_not_leak_whole_prompt(self):
        # decode drifted: trailing space lost. The naive startswith would
        # return the WHOLE text (prompt echo + completion); LCP strips the
        # common part so the prompt body is gone.
        prompt = "def add(a, b):  "          # two trailing spaces
        text = "def add(a, b): return a + b"  # one space — diverges at the 2nd
        out = strip_prompt_prefix(text, prompt)
        assert "def add(a, b):" not in out   # prompt body not leaked
        assert "return a + b" in out

    def test_bos_render_prefix_difference(self):
        # Simulated: decode prepends a space the raw prompt lacks → no exact
        # startswith, but LCP still removes the shared body.
        prompt = "import numpy"
        text = "import numpy as np"
        assert strip_prompt_prefix(text, prompt) == " as np"

    def test_no_common_prefix_returns_full_text(self):
        # Total divergence (e.g. decode produced something unrelated) → returns
        # the text unchanged rather than crashing.
        assert strip_prompt_prefix("xyz completion", "abc") == "xyz completion"

    def test_completion_only_when_text_is_prompt_plus_more(self):
        prompt = "function foo() {\n"
        text = prompt + "  return 1;\n}"
        assert strip_prompt_prefix(text, prompt) == "  return 1;\n}"
