"""Tokenizer-health endpoint helper for the local UI.

Loads the project tokenizer and runs the shared health battery
(:mod:`cola_coder.tokenizer.health`) — the same checks the CLI
``tokenizer_health.py`` runs. Reuses ``tokenizer_info``'s path resolution so the
UI discovers the tokenizer the same way everywhere. Robust to a missing tokenizer
or absent ``tokenizers`` package: returns an ``{"error": ...}`` dict, never raises.
"""

from __future__ import annotations

from cola_coder.tokenizer.health import load_tokenizer, run_health_checks

from .tokenizer_info import _resolve_tokenizer_file


def tokenizer_health(path: str | None = None) -> dict:
    """Run the health battery on the resolved tokenizer.

    ``path`` may point at a ``tokenizer.json`` file OR its containing dir; when
    None the default locations are probed (storage.yaml, ``data/<dataset>/`` …).
    """
    resolved = _resolve_tokenizer_file(path)
    if resolved is None:
        target = path if path is not None else "<default locations>"
        return {"error": f"tokenizer.json not found: {target}"}

    try:
        tokenizer = load_tokenizer(resolved)
    except ImportError:
        return {"error": "tokenizers package not installed (pip install tokenizers)"}
    except Exception as exc:  # malformed tokenizer file, etc.
        return {"error": str(exc)}

    results = run_health_checks(tokenizer)
    failed = sum(1 for r in results if not r.ok)
    return {
        "path": str(resolved),
        "vocab_size": tokenizer.get_vocab_size(),
        "checks": [{"name": r.name, "ok": r.ok, "detail": r.detail} for r in results],
        "passed": len(results) - failed,
        "failed": failed,
        "ok": failed == 0,
    }
