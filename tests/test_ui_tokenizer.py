"""Tests for the read-only tokenizer inspection UI helper.

Hermetic: every test writes a fake tokenizer.json into ``tmp_path`` so they
never depend on a real tokenizer existing on disk. One optional test probes a
real tokenizer and skips if none is found.
"""

from __future__ import annotations

import json
from pathlib import Path

from cola_coder.ui.tokenizer_info import tokenizer_info


def _fake_tokenizer(*, with_digits: bool = True, with_fim: bool = True) -> dict:
    """Build a minimal but realistic byte-level BPE tokenizer.json structure."""
    added_tokens = [
        {"id": 0, "content": "<|pad|>", "special": True},
        {"id": 1, "content": "<|bos|>", "special": True},
        {"id": 2, "content": "<|eos|>", "special": True},
        {"id": 3, "content": "<|unk|>", "special": True},
    ]
    if with_fim:
        added_tokens += [
            {"id": 4, "content": "<|fim_prefix|>", "special": True},
            {"id": 5, "content": "<|fim_middle|>", "special": True},
            {"id": 6, "content": "<|fim_suffix|>", "special": True},
        ]

    pretokenizers = []
    if with_digits:
        pretokenizers.append({"type": "Digits", "individual_digits": True})
    pretokenizers.append({"type": "ByteLevel", "add_prefix_space": False})

    return {
        "version": "1.0",
        "added_tokens": added_tokens,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": pretokenizers},
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4},
            "merges": ["a b", "ab c"],
        },
    }


def _write(path: Path, obj: object) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_full_fields_on_file(tmp_path: Path):
    tok = _write(tmp_path / "tokenizer.json", _fake_tokenizer())
    info = tokenizer_info(str(tok))

    assert "error" not in info
    assert info["path"] == str(tok)
    assert info["vocab_size"] == 5
    assert info["n_merges"] == 2
    assert info["model_type"] == "BPE"
    assert info["has_fim_tokens"] is True
    assert info["digit_splitting"] is True


def test_special_tokens_content(tmp_path: Path):
    tok = _write(tmp_path / "tokenizer.json", _fake_tokenizer())
    info = tokenizer_info(str(tok))

    assert info["special_tokens"] == [
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
        "<|unk|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
    ]


def test_passing_directory(tmp_path: Path):
    """A directory containing tokenizer.json resolves the same as the file."""
    _write(tmp_path / "tokenizer.json", _fake_tokenizer())
    info = tokenizer_info(str(tmp_path))

    assert "error" not in info
    assert info["path"] == str(tmp_path / "tokenizer.json")
    assert info["vocab_size"] == 5


def test_directory_without_tokenizer(tmp_path: Path):
    info = tokenizer_info(str(tmp_path))
    assert "error" in info


def test_digit_splitting_false_when_absent(tmp_path: Path):
    tok = _write(tmp_path / "tokenizer.json", _fake_tokenizer(with_digits=False))
    info = tokenizer_info(str(tok))

    assert "error" not in info
    assert info["digit_splitting"] is False


def test_digit_splitting_false_when_not_individual(tmp_path: Path):
    """A Digits step with individual_digits false must NOT count as splitting."""
    obj = _fake_tokenizer()
    obj["pre_tokenizer"]["pretokenizers"][0]["individual_digits"] = False
    tok = _write(tmp_path / "tokenizer.json", obj)
    info = tokenizer_info(str(tok))

    assert info["digit_splitting"] is False


def test_digit_splitting_single_step_pretokenizer(tmp_path: Path):
    """pre_tokenizer can be a single step (not a Sequence)."""
    obj = _fake_tokenizer()
    obj["pre_tokenizer"] = {"type": "Digits", "individual_digits": True}
    tok = _write(tmp_path / "tokenizer.json", obj)
    info = tokenizer_info(str(tok))

    assert info["digit_splitting"] is True


def test_no_fim_tokens(tmp_path: Path):
    tok = _write(tmp_path / "tokenizer.json", _fake_tokenizer(with_fim=False))
    info = tokenizer_info(str(tok))

    assert info["has_fim_tokens"] is False
    assert "<|fim_prefix|>" not in info["special_tokens"]


def test_missing_merges_yields_zero(tmp_path: Path):
    obj = _fake_tokenizer()
    del obj["model"]["merges"]
    tok = _write(tmp_path / "tokenizer.json", obj)
    info = tokenizer_info(str(tok))

    assert info["n_merges"] == 0
    assert info["vocab_size"] == 5


def test_missing_path_returns_error(tmp_path: Path):
    info = tokenizer_info(str(tmp_path / "does_not_exist.json"))
    assert "error" in info
    assert "not found" in info["error"]


def test_garbage_json_returns_error(tmp_path: Path):
    bad = tmp_path / "tokenizer.json"
    bad.write_text("{not valid json", encoding="utf-8")
    info = tokenizer_info(str(bad))
    assert "error" in info


def test_non_object_json_returns_error(tmp_path: Path):
    bad = tmp_path / "tokenizer.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    info = tokenizer_info(str(bad))
    assert "error" in info


def test_default_discovery_runs_without_raising():
    """With no path, discovery probes default locations and never raises.

    The real tokenizer may or may not exist; either a populated dict or an
    {"error": ...} is acceptable — the contract is just that it never raises.
    """
    info = tokenizer_info()
    assert isinstance(info, dict)
    assert ("error" in info) or ("vocab_size" in info)


def test_real_tokenizer_optional():
    """OPTIONAL: inspect the real tokenizer if one is discoverable; else skip."""
    import pytest

    info = tokenizer_info()
    if "error" in info:
        pytest.skip("no real tokenizer.json found on disk")
    assert info["vocab_size"] > 0
    assert isinstance(info["model_type"], str)
