"""TOK-001: encode_fim / fim_prompt must fail loud when FIM tokens are absent.

The constructor only *requires* the core tokens (pad/bos/eos/unk); FIM tokens
are optional. Its own comment promised that ``encode_fim`` "checks separately",
but it did not — a tokenizer trained without ``<|fim_*|>`` left
``fim_prefix_id`` etc. as ``None``, and ``encode_fim`` returned
``[None] + prefix_ids + [None] + ...``. That ``None`` is not a valid token id:
it silently corrupts the sequence and only crashes (or produces garbage logits)
much later in the model, far from the real cause.

These tests build a tokenizer WITHOUT FIM tokens and assert that the FIM helpers
raise a clear ValueError up front, while ``has_fim_tokens()`` reports False.
A tokenizer that DOES have FIM tokens is unaffected (numerics unchanged).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from cola_coder.tokenizer import train_tokenizer as tt
from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer


# Core tokens only — deliberately NO <|fim_*|> tokens.
_CORE_ONLY = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]


def _train_no_fim_tokenizer(out_path: str) -> None:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=320,
        special_tokens=_CORE_ONLY,
        min_frequency=2,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(
        iter(["def f():\n  return 1\n", "const x = 1;\n"] * 200), trainer
    )
    tokenizer.save(out_path)


def _fim_tokenizer(tmp_path: Path) -> CodeTokenizer:
    out = str(tmp_path / "tok.json")
    tt.train_from_iterator(
        iter(["def f():\n  return 1\n", "const x = 1;\n"] * 200),
        vocab_size=320, output_path=out,
    )
    return CodeTokenizer(out)


class TestEncodeFimMissingTokens:
    def test_has_fim_tokens_false_without_fim(self, tmp_path):
        out = str(tmp_path / "nofim.json")
        _train_no_fim_tokenizer(out)
        tok = CodeTokenizer(out)
        # Core tokens still resolve (constructor would have raised otherwise).
        assert tok.has_fim_tokens() is False
        assert tok.fim_prefix_id is None

    def test_encode_fim_raises_without_fim(self, tmp_path):
        out = str(tmp_path / "nofim.json")
        _train_no_fim_tokenizer(out)
        tok = CodeTokenizer(out)
        with pytest.raises(ValueError, match="FIM special tokens"):
            tok.encode_fim("PRE", "SUF")

    def test_fim_prompt_raises_without_fim(self, tmp_path):
        out = str(tmp_path / "nofim.json")
        _train_no_fim_tokenizer(out)
        tok = CodeTokenizer(out)
        with pytest.raises(ValueError, match="FIM special tokens"):
            tok.fim_prompt("PRE", "SUF")

    def test_no_none_ids_leak(self, tmp_path):
        """Regression guard: the result must never contain a None id."""
        out = str(tmp_path / "nofim.json")
        _train_no_fim_tokenizer(out)
        tok = CodeTokenizer(out)
        try:
            ids = tok.encode_fim("PRE", "SUF")
        except ValueError:
            ids = []  # raising is the correct behaviour
        assert None not in ids


class TestEncodeFimWithTokensUnaffected:
    def test_has_fim_tokens_true(self, tmp_path):
        tok = _fim_tokenizer(tmp_path)
        assert tok.has_fim_tokens() is True

    def test_encode_fim_still_works(self, tmp_path):
        tok = _fim_tokenizer(tmp_path)
        ids = tok.encode_fim("PRE", "SUF")
        assert None not in ids
        assert ids[0] == tok.fim_prefix_id
        assert ids[-1] == tok.fim_middle_id

    def test_fim_prompt_still_works(self, tmp_path):
        tok = _fim_tokenizer(tmp_path)
        s = tok.fim_prompt("PRE", "SUF")
        assert "<|fim_prefix|>" in s and "<|fim_middle|>" in s
