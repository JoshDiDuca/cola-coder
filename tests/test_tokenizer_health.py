"""Tests for the tokenizer health checks (cola_coder.tokenizer.health).

TOOL-018: the "Special tokens" check used to look for un-piped names
(<pad>, <unk>, <bos>, <eos>), but cola-coder tokenizers use the piped
<|...|> forms — so the check reported all four as MISSING on a perfectly
valid tokenizer. These tests assert the check passes against a real tokenizer
trained with the canonical SPECIAL_TOKENS.

The checks were extracted from scripts/tokenizer_health.py into the shared
``cola_coder.tokenizer.health`` library (so the CLI and the web UI run identical
logic); these tests target that library directly.
"""

from __future__ import annotations

from pathlib import Path

from cola_coder.tokenizer import health
from cola_coder.tokenizer.health import _REQUIRED_SPECIAL_TOKENS, run_health_checks
from cola_coder.tokenizer.train_tokenizer import (
    SPECIAL_TOKENS,
    train_from_iterator,
)


def _build_tokenizer(tmp_path: Path):
    code_samples = [
        "def hello():\n    print('hi')\n",
        "const x = [1, 2, 3].map(n => n * 2);\n",
    ] * 100
    return train_from_iterator(
        iter(code_samples),
        vocab_size=512,
        output_path=str(tmp_path / "tok.json"),
    )


def test_lib_exposes_checks() -> None:
    assert hasattr(health, "_check_special_tokens")
    assert callable(run_health_checks)


def test_required_tokens_are_piped_form() -> None:
    """The required list must be the <|...|> names, not the un-piped ones."""
    assert "<|pad|>" in _REQUIRED_SPECIAL_TOKENS
    assert "<|bos|>" in _REQUIRED_SPECIAL_TOKENS
    assert "<|eos|>" in _REQUIRED_SPECIAL_TOKENS
    assert "<|unk|>" in _REQUIRED_SPECIAL_TOKENS
    # Regression guard: the old un-piped names must NOT be expected.
    assert "<pad>" not in _REQUIRED_SPECIAL_TOKENS
    # FIM tokens are optional, not required.
    assert "<|fim_prefix|>" not in _REQUIRED_SPECIAL_TOKENS


def test_special_token_check_passes_on_real_tokenizer(tmp_path) -> None:
    """A tokenizer trained with cola-coder's SPECIAL_TOKENS must PASS."""
    tok = _build_tokenizer(tmp_path)
    ok, msg = health._check_special_tokens(tok)
    assert ok, msg
    # FIM tokens are part of SPECIAL_TOKENS, so they must NOT be reported as
    # missing (the reasoning <think> tokens legitimately are, until reasoning
    # training adds them — that's an optional note, not a failure).
    assert "<|fim_prefix|>" not in msg


def test_full_battery_passes_on_real_tokenizer(tmp_path) -> None:
    """The whole battery runs and the special-tokens check passes on a real tokenizer."""
    tok = _build_tokenizer(tmp_path)
    results = {r.name: r for r in run_health_checks(tok)}
    assert results["Special tokens"].ok
    assert len(results) == 5


def test_required_tokens_match_canonical_core() -> None:
    """Required tokens are exactly the non-FIM canonical SPECIAL_TOKENS."""
    expected = [t for t in SPECIAL_TOKENS if "fim" not in t]
    assert _REQUIRED_SPECIAL_TOKENS == expected
