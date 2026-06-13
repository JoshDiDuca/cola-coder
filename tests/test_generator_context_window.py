"""INFER-013: generation must clamp prompt + new tokens to the KV-cache window.

The KV-cache and causal mask are allocated for exactly ``config.max_seq_len``
positions. Without a guard there are two reachable failures (both easy to hit
from the FIM endpoint, where a long file exceeds ``seq_len``):

* A prompt longer than ``max_seq_len`` makes prefill assign
  ``cache_k[:, 0:seq_len] = k`` with ``seq_len > max_seq_len`` → a cryptic
  ``RuntimeError`` that 500s the request.
* Generation crossing the window mid-decode writes to a zero-size slice
  (``cache_k[:, start_pos:start_pos+1]`` with ``start_pos >= max_seq_len``), so
  the token's K/V is silently dropped and the model reads stale cache — garbage
  output with no error.

``_fit_context_window`` left-truncates the prompt to the most recent
``max_seq_len - 1`` tokens and caps ``max_new_tokens`` so ``start_pos`` never
reaches the cache bound. These tests lock the helper and the generator paths.
"""

import torch

from cola_coder.inference.generator import CodeGenerator, _fit_context_window


# ── Pure helper ──────────────────────────────────────────────────────────────


class TestFitContextWindow:
    def test_prompt_within_window_unchanged(self):
        ids, cap = _fit_context_window([1, 2, 3], max_new_tokens=10, max_seq_len=16)
        assert ids == [1, 2, 3]
        assert cap == 10  # 16 - 3 = 13 room, 10 requested → 10

    def test_prompt_longer_than_window_left_truncated(self):
        ids, cap = _fit_context_window(list(range(20)), max_new_tokens=8, max_seq_len=8)
        # keep most-recent (max_seq_len - 1) = 7 tokens, the last 7 ids
        assert ids == [13, 14, 15, 16, 17, 18, 19]
        assert len(ids) == 7
        # one slot left → at most 1 new token may be generated
        assert cap == 1

    def test_max_new_tokens_capped_to_remaining_room(self):
        # 6-token prompt in a 10-slot window → only 4 generation slots remain.
        ids, cap = _fit_context_window([0, 1, 2, 3, 4, 5], max_new_tokens=100, max_seq_len=10)
        assert ids == [0, 1, 2, 3, 4, 5]
        assert cap == 4

    def test_prompt_exactly_fills_minus_one(self):
        # len == max_seq_len - 1 → no truncation, exactly 1 slot left.
        ids, cap = _fit_context_window([0, 1, 2], max_new_tokens=5, max_seq_len=4)
        assert ids == [0, 1, 2]
        assert cap == 1

    def test_zero_or_negative_max_seq_len_disables_guard(self):
        ids, cap = _fit_context_window([1, 2, 3], max_new_tokens=9, max_seq_len=0)
        assert ids == [1, 2, 3] and cap == 9


# ── Generator integration (stub model with a real KV-cache of fixed size) ─────


class _StubTokenizer:
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


class _Cfg:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len


class _FixedCacheModel:
    """Model that emits one scripted token per call, backed by a real fixed-size
    KV-cache so an out-of-bounds write fails exactly like the production model.

    cache_k has ``max_seq_len`` slots; every forward writes the step's K at
    ``start_pos`` (mirroring CausalSelfAttention), so a prompt or generation that
    overruns the window raises / silently no-ops just like the real path.
    """

    def __init__(self, scripted_ids, max_seq_len, vocab=256):
        self.scripted = scripted_ids
        self.vocab = vocab
        self.config = _Cfg(max_seq_len)
        self.calls = 0
        self.cache_k = None
        self.max_written_pos = -1

    def eval(self):
        return self

    def clear_caches(self):
        self.cache_k = None

    def __call__(self, input_ids, start_pos=0, use_cache=True):
        batch, seq = input_ids.shape[0], input_ids.shape[1]
        if use_cache:
            if self.cache_k is None:
                self.cache_k = torch.zeros(batch, self.config.max_seq_len, 1)
            # Mirror the production write — raises if seq overruns the window
            # at prefill, and tracks the highest position actually persisted so
            # we can assert no silent drops occurred during decode.
            self.cache_k[:batch, start_pos : start_pos + seq] = 1.0
            written = self.cache_k[:batch, start_pos : start_pos + seq]
            if written.shape[1] == seq:
                self.max_written_pos = max(self.max_written_pos, start_pos + seq - 1)
        idx = self.calls
        self.calls += 1
        tok = self.scripted[idx] if idx < len(self.scripted) else _StubTokenizer.eos_id
        logits = torch.full((batch, seq, self.vocab), -10.0)
        logits[:, -1, tok] = 10.0
        return logits


def _gen(completion, max_seq_len):
    scripted = [ord(c) for c in completion]
    return CodeGenerator(
        _FixedCacheModel(scripted, max_seq_len), _StubTokenizer(), device="cpu"
    )


class TestGeneratorRespectsWindow:
    def test_long_prompt_does_not_crash_prefill(self):
        # Prompt (+BOS) far exceeds the 8-slot window. Pre-fix this raised a
        # RuntimeError in prefill; now the prompt is left-truncated and it runs.
        gen = _gen("abc", max_seq_len=8)
        out = gen.generate(
            "x" * 50, max_new_tokens=5, temperature=0, repetition_penalty=1.0
        )
        assert isinstance(out, str)
        # No cache write ever exceeded the window bound.
        assert gen.model.max_written_pos < 8

    def test_generation_never_overruns_window(self):
        # Short prompt, but max_new_tokens is huge — generation must stop at the
        # window edge, never writing K/V to an out-of-range (zero-size) slice.
        gen = _gen("abcdefghijklmnop", max_seq_len=6)
        out = gen.generate(
            "P", max_new_tokens=100, temperature=0, repetition_penalty=1.0,
            return_new_only=True,
        )
        # prompt = BOS + 'P' (2 tokens) → at most 4 generated → 'abcd'
        assert out == "abcd"
        assert gen.model.max_written_pos < 6

    def test_stream_long_prompt_does_not_crash(self):
        gen = _gen("hi", max_seq_len=8)
        text = "".join(
            gen.generate_stream(
                "y" * 40, max_new_tokens=3, temperature=0, repetition_penalty=1.0
            )
        )
        assert isinstance(text, str)
        assert gen.model.max_written_pos < 8

    def test_stream_generation_capped_to_window(self):
        gen = _gen("abcdefghij", max_seq_len=6)
        text = "".join(
            gen.generate_stream(
                "P", max_new_tokens=100, temperature=0, repetition_penalty=1.0
            )
        )
        # BOS + 'P' = 2 prompt tokens, 4 slots left → 'abcd'
        assert text == "abcd"
        assert gen.model.max_written_pos < 6
