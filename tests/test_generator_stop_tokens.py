"""INFER-006: multi-token stop sequences must match the full string, not just
their first token.

The generator reduced every requested stop string to its FIRST token
(``stop_ids.add(encoded[0])``). So a stop like ``";\\n"`` (which tokenizes to
``[";", "\\n"]``) halted generation at the first ``";"`` — truncating code after
a single statement — and ``"\\n\\n"`` halted at the first single newline. This
is on the standard OpenAI ``stop`` path (server forwards ``request.stop``) and
the built-in prompt_templates ship such multi-char stops.

The fix keeps single-token stops (EOS + special tokens like ``<|im_end|>`` /
``<|fim_suffix|>``) at the exact token level and matches multi-token stops at
the STRING level on the decoded completion. These tests lock both paths.
"""

import torch

from cola_coder.inference.generator import CodeGenerator, _earliest_stop_index


class _StubTokenizer:
    """Char-level tokenizer: token id == ord(char). Specials are stripped on
    decode, so generated text renders 1:1 with the scripted ids."""

    eos_id = 0
    bos_id = 1

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [ord(c) for c in text]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special=True):
        specials = {self.eos_id, self.bos_id}
        return "".join(chr(i) for i in ids if not (skip_special and i in specials))


class _ScriptedModel:
    """Returns one-hot logits forcing a predetermined token each forward call.

    Forward call k emits ``scripted[k]`` (greedy argmax). After the script is
    exhausted it emits EOS so generation always terminates.
    """

    def __init__(self, scripted_ids, vocab=256):
        self.scripted = scripted_ids
        self.vocab = vocab
        self.calls = 0

    def eval(self):
        return self

    def clear_caches(self):
        pass

    def __call__(self, input_ids, start_pos=0, use_cache=True):
        batch, seq = input_ids.shape[0], input_ids.shape[1]
        idx = self.calls
        self.calls += 1
        tok = self.scripted[idx] if idx < len(self.scripted) else _StubTokenizer.eos_id
        logits = torch.full((batch, seq, self.vocab), -10.0)
        logits[:, -1, tok] = 10.0
        return logits


def _gen(completion: str):
    """A generator whose next-token script spells out `completion`."""
    scripted = [ord(c) for c in completion]
    model = _ScriptedModel(scripted)
    return CodeGenerator(model, _StubTokenizer(), device="cpu")


def _complete(gen, prompt, **kw):
    """Strip the prompt prefix from generate()'s prompt+completion return."""
    out = gen.generate(prompt, temperature=0, repetition_penalty=1.0, **kw)
    assert out.startswith(prompt)
    return out[len(prompt):]


class TestMultiTokenStopNotTruncatedEarly:
    def test_double_newline_keeps_single_newline(self):
        # "x\ny\n\nz": old code stopped at the FIRST '\n' (after x). Correct
        # behavior keeps "x\ny" and stops at the real "\n\n".
        gen = _gen("x\ny\n\nz")
        comp = _complete(gen, "P", max_new_tokens=20, stop_tokens=["\n\n"])
        assert comp == "x\ny"

    def test_semicolon_newline_keeps_statements(self):
        # ";\n" must not stop at the first ';'.
        gen = _gen("a=1; b=2;\nc=3")
        comp = _complete(gen, "P", max_new_tokens=30, stop_tokens=[";\n"])
        assert comp == "a=1; b=2"

    def test_word_stop_sequence(self):
        gen = _gen("foo\nbar\nclass X")
        comp = _complete(gen, "P", max_new_tokens=30, stop_tokens=["\nclass "])
        assert comp == "foo\nbar"


class TestSingleTokenStopStillExact:
    def test_single_char_stop_is_token_level(self):
        # "@" is one token → exact token-level stop (excluded from output).
        gen = _gen("ab@cd")
        comp = _complete(gen, "P", max_new_tokens=20, stop_tokens=["@"])
        assert comp == "ab"

    def test_eos_always_stops(self):
        gen = _gen("abc")  # script exhausts → EOS emitted
        comp = _complete(gen, "P", max_new_tokens=20)
        assert comp == "abc"


class TestStopOnlyInCompletion:
    def test_stop_string_in_prompt_is_ignored(self):
        # The prompt itself contains ";\n"; it must not truncate the output.
        gen = _gen("abc")
        comp = _complete(gen, "x=1;\nrun()", max_new_tokens=20, stop_tokens=[";\n"])
        assert comp == "abc"


class TestStreamingStops:
    def _stream(self, gen, prompt, **kw):
        return "".join(
            gen.generate_stream(prompt, temperature=0, repetition_penalty=1.0, **kw)
        )

    def test_stream_stops_at_multi_token_stop(self):
        gen = _gen("x\ny\n\nz")
        text = self._stream(gen, "P", max_new_tokens=20, stop_tokens=["\n\n"])
        assert text == "x\ny"

    def test_stream_single_token_stop(self):
        gen = _gen("ab@cd")
        text = self._stream(gen, "P", max_new_tokens=20, stop_tokens=["@"])
        assert text == "ab"

    def test_stream_flushes_held_back_tail_when_stop_never_hits(self):
        # String stop active but absent → the held-back tail must still flush.
        gen = _gen("hello world")
        text = self._stream(gen, "P", max_new_tokens=30, stop_tokens=["\n\n"])
        assert text == "hello world"

    def test_stream_without_stops_emits_everything(self):
        gen = _gen("abcdef")
        text = self._stream(gen, "P", max_new_tokens=30)
        assert text == "abcdef"


class TestPartitionAndHelper:
    def test_partition_splits_single_and_multi(self):
        gen = _gen("")
        single, strings = gen._partition_stops(["@", ";\n", "", "\n\n"])
        assert _StubTokenizer.eos_id in single
        assert ord("@") in single  # single-char → token-level
        assert ";\n" in strings and "\n\n" in strings  # multi-char → string-level

    def test_earliest_stop_index_respects_start(self):
        text = "a;\nb;\nc"
        # first ";\n" at idx 1, but searching from 3 finds the second at idx 4
        assert _earliest_stop_index(text, [";\n"], 0) == 1
        assert _earliest_stop_index(text, [";\n"], 3) == 4
        assert _earliest_stop_index(text, ["zz"], 0) is None
