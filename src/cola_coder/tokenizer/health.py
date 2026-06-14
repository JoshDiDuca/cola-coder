"""Tokenizer health checks — shared library.

The canonical battery of BPE-tokenizer health checks (vocab size, special-token
presence, encode/decode roundtrip fidelity, average token length, encode speed),
extracted so BOTH the CLI (``scripts/tokenizer_health.py``) and the web UI
(``cola_coder.ui.tokenizer_health_view``) run the exact same logic.

Each check takes an already-loaded ``tokenizers.Tokenizer`` and returns a
:class:`HealthCheckResult`. Loading is CPU-only and fast — no model/GPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from cola_coder.tokenizer.train_tokenizer import SPECIAL_TOKENS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tokenizers import Tokenizer


@dataclass(frozen=True)
class HealthCheckResult:
    """One tokenizer health check outcome."""

    name: str
    ok: bool
    detail: str


# ── Sample code snippets for token-length / roundtrip analysis ────────────────

_SAMPLE_SNIPPETS: tuple[str, ...] = (
    "def fibonacci(n: int) -> int:\n"
    "    if n <= 1:\n"
    "        return n\n"
    "    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    "interface User {\n"
    "  id: number;\n"
    "  name: string;\n"
    "  email: string;\n"
    "}\n\n"
    "function getUser(id: number): Promise<User> {\n"
    "  return fetch(`/api/users/${id}`).then(r => r.json());\n"
    "}\n",
    "const sum = (arr) => arr.reduce((acc, x) => acc + x, 0);\n"
    "const average = (arr) => sum(arr) / arr.length;\n",
)

# Required special tokens, derived from the canonical SPECIAL_TOKENS list so this
# check can never drift from what the tokenizer is actually trained with. FIM
# (<|fim_*|>) and reasoning (<think>/<\/think>) tokens are optional.
_REQUIRED_SPECIAL_TOKENS: list[str] = [t for t in SPECIAL_TOKENS if "fim" not in t]
_FIM_SPECIAL_TOKENS: list[str] = [t for t in SPECIAL_TOKENS if "fim" in t]
_THINKING_SPECIAL_TOKENS: list[str] = ["<think>", "</think>"]
_OPTIONAL_SPECIAL_TOKENS: list[str] = _FIM_SPECIAL_TOKENS + _THINKING_SPECIAL_TOKENS


def load_tokenizer(path: str | Path) -> "Tokenizer":
    """Load a ``tokenizer.json`` into a ``tokenizers.Tokenizer``.

    Raises ``ImportError`` if the ``tokenizers`` package is absent and
    ``FileNotFoundError`` if the path does not exist.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"tokenizer not found: {path}")
    from tokenizers import Tokenizer  # local import: heavy, optional dependency

    return Tokenizer.from_file(str(path))


def _check_vocab_size(tokenizer: "Tokenizer", expected: int | None) -> tuple[bool, str]:
    actual = tokenizer.get_vocab_size()
    if expected is not None:
        return actual == expected, f"vocab_size = {actual:,} (expected {expected:,})"
    return True, f"vocab_size = {actual:,}"


def _check_special_tokens(tokenizer: "Tokenizer") -> tuple[bool, str]:
    vocab = tokenizer.get_vocab()
    missing = [t for t in _REQUIRED_SPECIAL_TOKENS if t not in vocab]
    optional_missing = [t for t in _OPTIONAL_SPECIAL_TOKENS if t not in vocab]
    if missing:
        return False, f"Missing required special tokens: {missing}"
    note = f" (optional missing: {optional_missing})" if optional_missing else ""
    return True, f"All {len(_REQUIRED_SPECIAL_TOKENS)} required special tokens present{note}"


def _check_roundtrip(tokenizer: "Tokenizer") -> tuple[bool, str]:
    failures: list[str] = []
    for i, snippet in enumerate(_SAMPLE_SNIPPETS):
        decoded = tokenizer.decode(tokenizer.encode(snippet).ids)
        if decoded != snippet:
            failures.append(f"snippet[{i}]: {len(snippet)} -> {len(decoded)} chars")
    if failures:
        return False, "Roundtrip failures: " + "; ".join(failures)
    return True, f"Roundtrip OK on {len(_SAMPLE_SNIPPETS)} snippets"


def _check_avg_token_length(tokenizer: "Tokenizer") -> tuple[bool, str]:
    total_chars = sum(len(s) for s in _SAMPLE_SNIPPETS)
    total_tokens = sum(len(tokenizer.encode(s).ids) for s in _SAMPLE_SNIPPETS)
    if total_tokens == 0:
        return False, "No tokens produced from samples"
    avg = total_chars / total_tokens
    # Good BPE tokenizers produce ~3.5–5.5 chars/token on code.
    return 2.0 <= avg <= 8.0, f"avg chars/token = {avg:.2f} over {total_tokens} tokens"


def _check_encode_speed(tokenizer: "Tokenizer") -> tuple[bool, str]:
    text = "\n".join(_SAMPLE_SNIPPETS) * 20  # ~1000 lines
    t0 = time.perf_counter()
    ids = tokenizer.encode(text).ids
    elapsed = time.perf_counter() - t0
    tps = len(ids) / max(elapsed, 1e-9)
    return tps > 10_000, f"encode speed = {tps:,.0f} tok/s ({len(ids):,} tokens in {elapsed * 1000:.1f}ms)"


def run_health_checks(
    tokenizer: "Tokenizer", expected_vocab: int | None = None
) -> list[HealthCheckResult]:
    """Run the full health battery against a loaded tokenizer.

    Each check is isolated: an exception in one becomes a failed result rather
    than aborting the rest.
    """
    checks: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
        ("Vocab size", lambda: _check_vocab_size(tokenizer, expected_vocab)),
        ("Special tokens", lambda: _check_special_tokens(tokenizer)),
        ("Roundtrip encode/decode", lambda: _check_roundtrip(tokenizer)),
        ("Avg token length", lambda: _check_avg_token_length(tokenizer)),
        ("Encode speed", lambda: _check_encode_speed(tokenizer)),
    ]
    results: list[HealthCheckResult] = []
    for name, check_fn in checks:
        try:
            ok, detail = check_fn()
        except Exception as exc:  # a broken tokenizer must not crash the report
            ok, detail = False, f"Exception: {exc}"
        results.append(HealthCheckResult(name=name, ok=ok, detail=detail))
    return results
