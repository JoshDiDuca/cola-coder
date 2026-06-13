"""Tests for scripts/tokenizer_health.py.

TOOL-018: the "Special tokens" check used to look for un-piped names
(<pad>, <unk>, <bos>, <eos>), but cola-coder tokenizers use the piped
<|...|> forms — so the check reported all four as MISSING on a perfectly
valid tokenizer. These tests load the script and assert the check now passes
against a real tokenizer trained with the canonical SPECIAL_TOKENS.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from cola_coder.tokenizer.train_tokenizer import (
    SPECIAL_TOKENS,
    train_from_iterator,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "tokenizer_health.py"
)


def _load_health_module():
    """Import scripts/tokenizer_health.py as a module (it's a CLI script)."""
    spec = importlib.util.spec_from_file_location("tokenizer_health", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_script_imports() -> None:
    module = _load_health_module()
    assert hasattr(module, "_check_special_tokens")


def test_required_tokens_are_piped_form() -> None:
    """The required list must be the <|...|> names, not the un-piped ones."""
    module = _load_health_module()
    assert "<|pad|>" in module._REQUIRED_SPECIAL_TOKENS
    assert "<|bos|>" in module._REQUIRED_SPECIAL_TOKENS
    assert "<|eos|>" in module._REQUIRED_SPECIAL_TOKENS
    assert "<|unk|>" in module._REQUIRED_SPECIAL_TOKENS
    # Regression guard: the old un-piped names must NOT be expected.
    assert "<pad>" not in module._REQUIRED_SPECIAL_TOKENS
    # FIM tokens are optional, not required.
    assert "<|fim_prefix|>" not in module._REQUIRED_SPECIAL_TOKENS


def test_special_token_check_passes_on_real_tokenizer(tmp_path) -> None:
    """A tokenizer trained with cola-coder's SPECIAL_TOKENS must PASS."""
    module = _load_health_module()
    tok = _build_tokenizer(tmp_path)
    ok, msg = module._check_special_tokens(tok)
    assert ok, msg
    # FIM tokens are part of SPECIAL_TOKENS, so they must NOT be reported as
    # missing (the reasoning <think> tokens legitimately are, until reasoning
    # training adds them — that's an optional note, not a failure).
    assert "<|fim_prefix|>" not in msg


def test_required_tokens_match_canonical_core() -> None:
    """Required tokens are exactly the non-FIM canonical SPECIAL_TOKENS."""
    module = _load_health_module()
    expected = [t for t in SPECIAL_TOKENS if "fim" not in t]
    assert module._REQUIRED_SPECIAL_TOKENS == expected
