"""Hermetic tests for the UI tokenizer playground (``ui/tokenize.py``).

Each test builds a tiny REAL tokenizer at runtime via the ``tokenizers`` library
(a WordLevel model with a small known vocab), writes it to a tmp_path
tokenizer.json, and asserts ``tokenize_text`` encodes known text into the
expected ids/tokens. No GPU, no checkpoint, no project tokenizer required.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tokenizers")

from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402

from cola_coder.ui.tokenize import tokenize_text  # noqa: E402

VOCAB = {"<unk>": 0, "foo": 1, "bar": 2, "baz": 3, "qux": 4}


def _build_tokenizer(path) -> str:
    """Write a tiny WordLevel tokenizer.json at ``path`` and return its str path."""
    tok = Tokenizer(WordLevel(vocab=dict(VOCAB), unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    str_path = str(path)
    tok.save(str_path)
    return str_path


@pytest.fixture()
def tokenizer_file(tmp_path):
    return _build_tokenizer(tmp_path / "tokenizer.json")


def test_encode_known_text(tokenizer_file):
    result = tokenize_text("foo bar baz", path=tokenizer_file)
    assert "error" not in result
    assert result["ids"] == [1, 2, 3]
    assert result["tokens"] == ["foo", "bar", "baz"]


def test_count_matches_len_ids(tokenizer_file):
    result = tokenize_text("foo bar baz qux", path=tokenizer_file)
    assert result["count"] == len(result["ids"]) == 4
    assert result["ids"] == [1, 2, 3, 4]


def test_unknown_token_maps_to_unk(tokenizer_file):
    result = tokenize_text("foo unknownword", path=tokenizer_file)
    assert result["ids"] == [1, 0]
    assert result["tokens"] == ["foo", "<unk>"]


def test_path_is_resolved_in_result(tokenizer_file):
    result = tokenize_text("foo", path=tokenizer_file)
    assert result["path"] == tokenizer_file


def test_path_as_containing_directory(tmp_path):
    _build_tokenizer(tmp_path / "tokenizer.json")
    result = tokenize_text("foo bar", path=str(tmp_path))
    assert "error" not in result
    assert result["ids"] == [1, 2]


def test_empty_text_returns_zero_tokens(tokenizer_file):
    result = tokenize_text("", path=tokenizer_file)
    assert "error" not in result
    assert result["count"] == 0
    assert result["ids"] == []
    assert result["tokens"] == []
    assert result["path"] == tokenizer_file


def test_whitespace_only_returns_zero_tokens(tokenizer_file):
    result = tokenize_text("   \n\t  ", path=tokenizer_file)
    assert "error" not in result
    assert result["count"] == 0
    assert result["ids"] == []
    assert result["tokens"] == []


def test_truncation_sets_flag(tokenizer_file):
    # max_chars=7 keeps "foo bar" (7 chars); the trailing " baz" is dropped.
    result = tokenize_text("foo bar baz", path=tokenizer_file, max_chars=7)
    assert result.get("truncated") is True
    assert result["ids"] == [1, 2]
    assert result["tokens"] == ["foo", "bar"]


def test_no_truncation_flag_when_under_limit(tokenizer_file):
    result = tokenize_text("foo bar", path=tokenizer_file, max_chars=20000)
    assert "truncated" not in result


def test_truncation_to_empty_returns_zero_tokens(tokenizer_file):
    result = tokenize_text("foo bar", path=tokenizer_file, max_chars=0)
    assert "error" not in result
    assert result.get("truncated") is True
    assert result["count"] == 0
    assert result["ids"] == []


def test_missing_path_returns_error(tmp_path):
    missing = str(tmp_path / "does_not_exist.json")
    result = tokenize_text("foo", path=missing)
    assert "error" in result
    assert "ids" not in result


def test_garbage_tokenizer_file_returns_error(tmp_path):
    garbage = tmp_path / "tokenizer.json"
    garbage.write_text("this is not valid tokenizer json {{{", encoding="utf-8")
    result = tokenize_text("foo", path=str(garbage))
    assert "error" in result
