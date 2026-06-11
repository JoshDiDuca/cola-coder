"""INFER-008: StreamingGenerator had two bugs already fixed elsewhere.

(1) Per-token decode([next_token]) mangles byte-level BPE — a single token can
be a partial multi-byte UTF-8 sequence, so it decodes to nothing/garbage in
isolation. (2) Multi-token stop sequences were reduced to their first token
(the INFER-006 bug), halting far too early. This module is menu-wired but had
ZERO tests. Fixed to use full-decode-diff + the shared partition_stops matcher.
"""

import torch

from cola_coder.features.streaming_generation import StreamingGenerator


class _ScriptedModel:
    """Emits a fixed token sequence via one-hot logits (greedy argmax)."""

    def __init__(self, scripted, vocab=256):
        self.scripted = scripted
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
        tok = self.scripted[idx] if idx < len(self.scripted) else 0  # eos after script
        logits = torch.full((batch, seq, self.vocab), -10.0)
        logits[:, -1, tok] = 10.0
        return logits


class _CharTokenizer:
    eos_id = 0
    bos_id = 1

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [ord(c) for c in text]
        return [self.bos_id] + ids if add_bos else ids

    def decode(self, ids, skip_special=True):
        return "".join(chr(i) for i in ids if i not in (self.eos_id, self.bos_id))


class _MergeTokenizer:
    """Simulates byte-level BPE: tokens 100/101 alone are partial bytes (decode
    to nothing), but [100, 101] together decode to 'Z'. Per-token decode loses
    the 'Z'; full-decode-diff recovers it."""

    eos_id = 0
    bos_id = 1

    def encode(self, text, add_bos=False, add_eos=False):
        return [self.bos_id] if add_bos else []

    def decode(self, ids, skip_special=True):
        ids = [x for x in ids if x not in (self.eos_id, self.bos_id)]
        out, i = [], 0
        while i < len(ids):
            if ids[i] == 100 and i + 1 < len(ids) and ids[i + 1] == 101:
                out.append("Z")
                i += 2
            elif ids[i] in (100, 101):
                i += 1  # partial byte alone → nothing
            else:
                out.append(chr(ids[i]))
                i += 1
        return "".join(out)


def _stream_text(model, tokenizer, **kw):
    gen = StreamingGenerator(model, tokenizer, device="cpu")
    return "".join(
        st.text for st in gen.stream("P", temperature=0, repetition_penalty=1.0, **kw)
    )


class TestMultiTokenStop:
    def test_double_newline_not_truncated_early(self):
        # "x\ny\n\nz" with stop "\n\n": keep "x\ny", stop at the real "\n\n".
        model = _ScriptedModel([ord(c) for c in "x\ny\n\nz"])
        text = _stream_text(model, _CharTokenizer(), max_new_tokens=20, stop_tokens=["\n\n"])
        assert text == "x\ny"


class TestSingleTokenStop:
    def test_eos_stops_and_emits_all(self):
        model = _ScriptedModel([ord(c) for c in "abc"])  # then EOS
        text = _stream_text(model, _CharTokenizer(), max_new_tokens=20)
        assert text == "abc"

    def test_no_stop_passthrough(self):
        model = _ScriptedModel([ord(c) for c in "hello"])
        text = _stream_text(model, _CharTokenizer(), max_new_tokens=20)
        assert text == "hello"


class TestFullDecodeDiff:
    def test_multibyte_char_recovered(self):
        # Per-token decode would yield "" + "" (lost). full-decode-diff → "Z".
        model = _ScriptedModel([100, 101])  # then EOS
        text = _stream_text(model, _MergeTokenizer(), max_new_tokens=10)
        assert text == "Z"


class TestStreamTokenMetadata:
    def test_token_ids_and_positions_present(self):
        model = _ScriptedModel([ord(c) for c in "ab"])
        gen = StreamingGenerator(model, _CharTokenizer(), device="cpu")
        toks = list(gen.stream("P", temperature=0, repetition_penalty=1.0, max_new_tokens=10))
        assert [t.token_id for t in toks] == [ord("a"), ord("b")]
        assert [t.position for t in toks] == [0, 1]
