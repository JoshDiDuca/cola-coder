"""Tests for tokenizer_debugger.py (feature 59)."""

import pytest

from cola_coder.features.tokenizer_debugger import (
    FEATURE_ENABLED,
    TokenizerDebugger,
    TokenizerLike,
    is_enabled,
)


# ---------------------------------------------------------------------------
# A minimal fake tokenizer for testing (character-level)
# ---------------------------------------------------------------------------

class CharTokenizer:
    """Character-level tokenizer: each character maps to its ord value."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


class WordTokenizer:
    """Word-level tokenizer that splits on spaces."""

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._rev: dict[int, str] = {}

    def _get_id(self, word: str) -> int:
        if word not in self._vocab:
            idx = len(self._vocab)
            self._vocab[word] = idx
            self._rev[idx] = word
        return self._vocab[word]

    def encode(self, text: str) -> list[int]:
        return [self._get_id(w) for w in text.split(" ")] if text else []

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._rev.get(i, "?") for i in ids)


@pytest.fixture
def char_dbg():
    return TokenizerDebugger(CharTokenizer())


@pytest.fixture
def word_dbg():
    return TokenizerDebugger(WordTokenizer())


def test_feature_flag():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_tokenizer_like_protocol():
    tok = CharTokenizer()
    assert isinstance(tok, TokenizerLike)


def test_analyze_token_count(char_dbg):
    view = char_dbg.analyze("hello")
    assert view.n_tokens == 5
    assert view.n_chars == 5


def test_analyze_boundaries_count(char_dbg):
    view = char_dbg.analyze("abc")
    assert len(view.boundaries) == 3


def test_analyze_chars_per_token(char_dbg):
    view = char_dbg.analyze("abcd")
    assert view.chars_per_token == pytest.approx(1.0)


def test_analyze_visual_contains_brackets(char_dbg):
    view = char_dbg.analyze("hi")
    assert "[" in view.visual
    assert "]" in view.visual


def test_analyze_custom_boundaries():
    dbg = TokenizerDebugger(CharTokenizer(), boundary_open="<", boundary_close=">")
    view = dbg.analyze("ab")
    assert view.visual.startswith("<")
    assert ">" in view.visual


def test_analyze_as_records(char_dbg):
    view = char_dbg.analyze("abc")
    records = view.as_records()
    assert len(records) == 3
    assert all("token_id" in r and "token_str" in r for r in records)


def test_analyze_empty_string(char_dbg):
    view = char_dbg.analyze("")
    assert view.n_tokens == 0
    assert view.chars_per_token == pytest.approx(0.0)


def test_compare_same_text(char_dbg):
    result = char_dbg.compare("hello", "hello")
    assert result.n_tokens_a == result.n_tokens_b
    assert result.efficiency_delta == pytest.approx(0.0)


def test_compare_different_lengths(char_dbg):
    result = char_dbg.compare("hi", "hello")
    assert result.n_tokens_a == 2
    assert result.n_tokens_b == 5


def test_compare_shared_ids(char_dbg):
    result = char_dbg.compare("abc", "bcd")
    # b, c are shared
    assert ord("b") in result.shared_ids
    assert ord("c") in result.shared_ids


def test_compare_summary(char_dbg):
    result = char_dbg.compare("hello", "world")
    s = result.summary()
    assert "tok" in s


def test_find_worst_cases(char_dbg):
    candidates = ["a", "ab", "abc", "abcd", "abcde"]
    worst = char_dbg.find_worst_cases(candidates, top_n=3)
    # Character-level: all have tokens_per_char=1, so just check we get 3 back
    assert len(worst) == 3


def test_find_worst_cases_empty(char_dbg):
    result = char_dbg.find_worst_cases([""])
    assert result == []


def test_vocab_statistics_basic(char_dbg):
    vocab = {"a": 1, "b": 2, "c": 3, " ": 4}
    stats = char_dbg.vocab_statistics(vocab)
    assert stats["vocab_size"] == 4
    assert stats["mean_token_len"] == pytest.approx(1.0)


def test_vocab_statistics_empty(char_dbg):
    stats = char_dbg.vocab_statistics({})
    assert stats["vocab_size"] == 0


def test_highlight_merges_returns_stages(char_dbg):
    stages = char_dbg.highlight_merges("aabba", n_steps=3)
    assert len(stages) >= 1
    # Each stage should be a non-empty string
    assert all(isinstance(s, str) for s in stages)


def test_highlight_merges_first_stage_char_level(char_dbg):
    stages = char_dbg.highlight_merges("abc", n_steps=2)
    # First stage should have individual characters as tokens
    assert "[a]" in stages[0]
    assert "[b]" in stages[0]


def test_id_to_token_function():
    vocab = {0: "hello", 1: "world"}
    dbg = TokenizerDebugger(
        CharTokenizer(),
        id_to_token=lambda tid: vocab.get(tid, f"<{tid}>"),
    )
    # With id_to_token, _tok_str uses it
    assert dbg._tok_str(0) == "hello"
    assert dbg._tok_str(99) == "<99>"
