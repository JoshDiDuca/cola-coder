"""Tests for inference/text_utils.strip_prompt_prefix (INFER-001).

The non-streaming server endpoints used `result[len(prompt):] if
result.startswith(prompt) else result`, which leaks the WHOLE prompt when
BPE decode(encode(prompt)) isn't byte-identical to the prompt. The shared
helper strips the longest common prefix instead, so a mismatch costs at most
a few boundary characters — never the entire prompt.
"""

from cola_coder.inference.text_utils import strip_prompt_prefix, trim_suffix_overlap


class TestStripPromptPrefix:
    def test_exact_prefix(self):
        assert strip_prompt_prefix("def f():\n    pass", "def f():\n") == "    pass"


class TestTrimSuffixOverlap:
    def test_trims_verbatim_suffix_duplication(self):
        # Model regenerated the closing lines already present in the suffix.
        suffix = "\n  return x;\n}"
        infill = "const y = 1;\n  return x;\n}"
        assert trim_suffix_overlap(infill, suffix) == "const y = 1;"

    def test_trims_longest_overlap_not_shortest(self):
        # Both k=1 (';') and the full tail match; the LONGEST must win.
        suffix = "; done();"
        infill = "doThing(); done();"
        assert trim_suffix_overlap(infill, suffix) == "doThing()"

    def test_no_overlap_unchanged(self):
        assert trim_suffix_overlap("const y = 1;", "\nfunction g() {}") == "const y = 1;"

    def test_tiny_coincidental_overlap_kept(self):
        # A lone ';' the completion legitimately ends on (and the suffix opens
        # with) is below min_overlap -> not trimmed.
        assert trim_suffix_overlap("a = b;", ";") == "a = b;"

    def test_full_infill_is_suffix(self):
        # Entire infill duplicates the suffix start -> trimmed to empty.
        assert trim_suffix_overlap("})\n", "})\n more") == ""

    def test_empty_inputs(self):
        assert trim_suffix_overlap("", "x") == ""
        assert trim_suffix_overlap("x", "") == "x"

    def test_min_overlap_threshold_respected(self):
        # Overlap "})" is length 2; default min_overlap=3 keeps it, =2 trims it.
        assert trim_suffix_overlap("f()})", "}) end") == "f()})"
        assert trim_suffix_overlap("f()})", "}) end", min_overlap=2) == "f()"

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
