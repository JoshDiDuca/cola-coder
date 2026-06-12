"""INFER-013: /v1/chat/completions must not echo the ChatML prompt into the reply.

The non-streaming chat endpoint string-diffed the result against the marker-form
ChatML prompt (`<|im_start|>…`). But generate() returns decode(prompt+completion)
and decode() STRIPS special tokens, so the result never starts with the marker
prompt → the longest-common-prefix helper kept (nearly) the whole prompt and
leaked it back as the assistant message. The FIM endpoint already decoded the
prompt first (BUG-111); chat now shares that logic via _completion_after_prompt.
"""

from cola_coder.inference.server import _completion_after_prompt
from cola_coder.inference.text_utils import strip_prompt_prefix


class _StubTok:
    """decode() drops special-token markers, exactly like CodeTokenizer."""

    SPECIALS = (
        "<|im_start|>", "<|im_end|>",
        "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>",
    )

    def encode(self, text, add_bos=False):
        return [text]  # carry the string through as the only "id"

    def decode(self, ids, skip_special=True):
        s = "".join(ids)
        if skip_special:
            for sp in self.SPECIALS:
                s = s.replace(sp, "")
        return s


def _result_for(prompt: str, completion: str, tok: _StubTok) -> str:
    """What generate() returns: decode(prompt_ids + completion_ids), specials gone."""
    return tok.decode([prompt + completion])


class TestCompletionAfterPrompt:
    def test_chatml_prompt_does_not_leak(self):
        tok = _StubTok()
        prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        completion = "def f():\n    return 1"
        result = _result_for(prompt, completion, tok)
        assert _completion_after_prompt(result, prompt, tok) == completion

    def test_naive_raw_diff_would_have_leaked(self):
        # Proves the bug the fix addresses: diffing the RAW marker prompt leaves
        # the decoded prompt body in the "completion".
        tok = _StubTok()
        prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        result = _result_for(prompt, "REPLY", tok)
        leaked = strip_prompt_prefix(result, prompt)  # raw (marker) prompt
        assert "hi" in leaked and leaked != "REPLY"  # prompt body leaked
        # The fix returns only the reply.
        assert _completion_after_prompt(result, prompt, tok) == "REPLY"

    def test_fim_prompt_infill_only(self):
        tok = _StubTok()
        fim_prompt = "<|fim_prefix|>def f():\n    <|fim_suffix|>\n    return x<|fim_middle|>"
        infill = "x = 1"
        result = _result_for(fim_prompt, infill, tok)
        assert _completion_after_prompt(result, fim_prompt, tok) == infill

    def test_base_mode_plaintext_prompt_unaffected(self):
        # No special tokens → decoded prompt == prompt → behaves like before.
        tok = _StubTok()
        prompt = "def add(a, b):\n"
        completion = "    return a + b"
        result = _result_for(prompt, completion, tok)
        assert _completion_after_prompt(result, prompt, tok) == completion

    def test_empty_completion(self):
        tok = _StubTok()
        prompt = "<|im_start|>assistant\n"
        result = _result_for(prompt, "", tok)
        assert _completion_after_prompt(result, prompt, tok) == ""
