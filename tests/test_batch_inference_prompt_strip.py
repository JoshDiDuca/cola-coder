"""INFER-009: BatchInference must not echo the prompt on BPE round-trip drift.

generator.generate() returns decode(prompt_tokens + new_tokens). The naive
`output[len(prompt):] if output.startswith(prompt) else output` returned the
WHOLE prompt echo whenever the decoded text didn't match the prompt
byte-for-byte (the INFER-001 leak). Fixed by using the canonical
strip_prompt_prefix (longest-common-prefix). This is the same bug INFER-001
fixed in the server, in a different, untested code path.
"""

from cola_coder.features.batch_inference import BatchInference
from cola_coder.inference.text_utils import strip_prompt_prefix


class _StubGen:
    """Returns prompt+completion, optionally with a one-char drift to simulate
    a BPE decode mismatch (so startswith(prompt) fails)."""

    def __init__(self, completion: str, drift: bool = False):
        self._completion = completion
        self._drift = drift
        self.last_raw = None

    def generate(self, prompt, **kwargs):
        if self._drift:
            raw = prompt[:-1] + " " + self._completion  # last char replaced
        else:
            raw = prompt + self._completion
        self.last_raw = raw
        return raw


class TestPromptStrip:
    def test_clean_prompt_stripped(self):
        gen = _StubGen(completion="BODY")
        bi = BatchInference(generator=gen)
        out = bi.run(["def f():"]).results[0].output
        assert out == "BODY"

    def test_drift_does_not_echo_full_prompt(self):
        prompt = "abcdef"
        gen = _StubGen(completion="BODY", drift=True)
        bi = BatchInference(generator=gen)
        out = bi.run([prompt]).results[0].output
        # The full prompt must NOT be echoed back...
        assert prompt not in out
        # ...and the result matches the canonical strip (longest common prefix).
        assert out == strip_prompt_prefix(gen.last_raw, prompt)

    def test_matches_canonical_helper_general(self):
        # The plugin output must always equal strip_prompt_prefix of the raw.
        gen = _StubGen(completion=" return 1\n")
        bi = BatchInference(generator=gen)
        out = bi.run(["function g() {"]).results[0].output
        assert out == strip_prompt_prefix(gen.last_raw, "function g() {")
