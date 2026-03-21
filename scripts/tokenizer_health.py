"""Tokenizer health check script.

Loads a trained BPE tokenizer and runs a battery of health checks:
- Vocab size matches expected
- Special tokens are present (<pad>, <unk>, <bos>, <eos>, <think>, </think>)
- Encode/decode roundtrip fidelity on sample code
- Average token length on representative code snippets

Usage:
    python scripts/tokenizer_health.py --tokenizer tokenizer.json
    python scripts/tokenizer_health.py --tokenizer tokenizer.json --expected-vocab 32768
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from cola_coder.cli import cli  # noqa: E402


# ── Sample code snippets for token-length analysis ────────────────────────────

_SAMPLE_SNIPPETS = [
    # Python
    """\
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
""",
    # TypeScript
    """\
interface User {
  id: number;
  name: string;
  email: string;
}

function getUser(id: number): Promise<User> {
  return fetch(`/api/users/${id}`).then(r => r.json());
}
""",
    # JavaScript
    """\
const sum = (arr) => arr.reduce((acc, x) => acc + x, 0);
const average = (arr) => sum(arr) / arr.length;
""",
]

# Known special tokens for cola-coder
_EXPECTED_SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
]

# Thinking tokens (reasoning module — optional)
_OPTIONAL_SPECIAL_TOKENS = ["<think>", "</think>"]


# ── Health check functions ─────────────────────────────────────────────────────

def _check_vocab_size(tokenizer, expected: int | None) -> tuple[bool, str]:
    """Check tokenizer vocabulary size."""
    actual = tokenizer.get_vocab_size()
    if expected is not None:
        ok = actual == expected
        msg = f"vocab_size = {actual:,} (expected {expected:,})"
        return ok, msg
    return True, f"vocab_size = {actual:,}"


def _check_special_tokens(tokenizer) -> tuple[bool, str]:
    """Check that required special tokens are present."""
    vocab = tokenizer.get_vocab()
    missing = [t for t in _EXPECTED_SPECIAL_TOKENS if t not in vocab]
    optional_missing = [t for t in _OPTIONAL_SPECIAL_TOKENS if t not in vocab]
    if missing:
        return False, f"Missing required special tokens: {missing}"
    note = ""
    if optional_missing:
        note = f" (optional missing: {optional_missing})"
    return True, f"All {len(_EXPECTED_SPECIAL_TOKENS)} required special tokens present{note}"


def _check_roundtrip(tokenizer) -> tuple[bool, str]:
    """Test encode → decode roundtrip on each sample snippet."""
    failures = []
    for i, snippet in enumerate(_SAMPLE_SNIPPETS):
        ids = tokenizer.encode(snippet).ids
        decoded = tokenizer.decode(ids)
        if decoded != snippet:
            failures.append(
                f"snippet[{i}]: original={len(snippet)} chars, "
                f"decoded={len(decoded)} chars"
            )
    if failures:
        return False, "Roundtrip failures: " + "; ".join(failures)
    return True, f"Roundtrip OK on {len(_SAMPLE_SNIPPETS)} snippets"


def _check_avg_token_length(tokenizer) -> tuple[bool, str]:
    """Compute average characters-per-token on sample code."""
    total_chars = 0
    total_tokens = 0
    for snippet in _SAMPLE_SNIPPETS:
        ids = tokenizer.encode(snippet).ids
        total_chars += len(snippet)
        total_tokens += len(ids)
    if total_tokens == 0:
        return False, "No tokens produced from samples"
    avg = total_chars / total_tokens
    # Good BPE tokenizers produce ~3.5–5.5 chars/token on code
    ok = 2.0 <= avg <= 8.0
    return ok, f"avg chars/token = {avg:.2f} over {total_tokens} tokens"


def _check_encode_speed(tokenizer) -> tuple[bool, str]:
    """Quick encode throughput test."""
    text = "\n".join(_SAMPLE_SNIPPETS) * 20  # ~1000 lines
    t0 = time.perf_counter()
    ids = tokenizer.encode(text).ids
    elapsed = time.perf_counter() - t0
    tps = len(ids) / max(elapsed, 1e-9)
    ok = tps > 10_000  # 10k tokens/sec is a soft floor for sanity
    return ok, f"encode speed = {tps:,.0f} tok/s ({len(ids):,} tokens in {elapsed*1000:.1f}ms)"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a health check on a trained BPE tokenizer."
    )
    parser.add_argument(
        "--tokenizer",
        default="tokenizer.json",
        help="Path to tokenizer.json (default: tokenizer.json)",
    )
    parser.add_argument(
        "--expected-vocab",
        type=int,
        default=None,
        metavar="N",
        help="Expected vocabulary size (optional, for strict check)",
    )
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer)
    cli.header("Cola-Coder", "Tokenizer Health Check")

    if not tokenizer_path.exists():
        cli.error(f"Tokenizer not found: {tokenizer_path}")
        return 1

    # Load tokenizer
    try:
        from tokenizers import Tokenizer  # type: ignore

        t0 = time.perf_counter()
        tok = Tokenizer.from_file(str(tokenizer_path))
        load_ms = (time.perf_counter() - t0) * 1000
        cli.info("Loaded", f"{tokenizer_path} in {load_ms:.1f}ms")
    except ImportError:
        cli.error("tokenizers package not installed", hint="pip install tokenizers")
        return 1
    except Exception as exc:
        cli.error(f"Failed to load tokenizer: {exc}")
        return 1

    # Run checks
    checks = [
        ("Vocab size", lambda: _check_vocab_size(tok, args.expected_vocab)),
        ("Special tokens", lambda: _check_special_tokens(tok)),
        ("Roundtrip encode/decode", lambda: _check_roundtrip(tok)),
        ("Avg token length", lambda: _check_avg_token_length(tok)),
        ("Encode speed", lambda: _check_encode_speed(tok)),
    ]

    passed = 0
    failed = 0

    for name, check_fn in checks:
        try:
            ok, msg = check_fn()
        except Exception as exc:
            ok, msg = False, f"Exception: {exc}"

        if ok:
            cli.print(f"  [green]PASS[/green]  {name}: {msg}")
            passed += 1
        else:
            cli.print(f"  [red]FAIL[/red]  {name}: {msg}")
            failed += 1

    cli.print("")
    if failed == 0:
        cli.success(f"All {passed} checks passed")
        return 0
    else:
        cli.error(f"{failed} check(s) failed, {passed} passed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
