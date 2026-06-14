"""Tokenizer health check script.

Loads a trained BPE tokenizer and runs a battery of health checks:
- Vocab size matches expected
- Special tokens are present (<|pad|>, <|unk|>, <|bos|>, <|eos|>, plus optional
  FIM and reasoning tokens)
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
from cola_coder.tokenizer.health import run_health_checks  # noqa: E402


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

    # Run checks (shared library — same battery the web UI runs).
    results = run_health_checks(tok, args.expected_vocab)

    passed = 0
    failed = 0

    for result in results:
        if result.ok:
            cli.print(f"  [green]PASS[/green]  {result.name}: {result.detail}")
            passed += 1
        else:
            cli.print(f"  [red]FAIL[/red]  {result.name}: {result.detail}")
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
