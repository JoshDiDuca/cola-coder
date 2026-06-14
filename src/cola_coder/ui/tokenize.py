"""Tokenizer playground helpers for the local UI/dashboard.

A GPU-free tokenizer playground: encode arbitrary text into token ids and token
piece strings so users can SEE how the BPE tokenizer (with digit-splitting)
segments code. Unlike ``tokenizer_info`` (which parses tokenizer.json directly),
this needs the REAL tokenizer to encode, so it loads the HuggingFace
``tokenizers`` library. The import is guarded — if the library is missing, an
``{"error": ...}`` dict is returned rather than crashing. All functions are
robust to missing or malformed inputs and never raise.
"""

from __future__ import annotations

from cola_coder.ui.tokenizer_info import _resolve_tokenizer_file


def tokenize_text(text: str, path: str | None = None, max_chars: int = 20000) -> dict:
    """Encode ``text`` with the project tokenizer. If ``path`` is None, DISCOVER the default
    tokenizer.json (same discovery as tokenizer_info). Returns:
      {"path": str,            # resolved tokenizer.json
       "count": int,           # number of tokens
       "ids": list[int],       # token ids
       "tokens": list[str]}    # token piece strings (decoded per-id or from encoding.tokens)
    Truncate ``text`` to max_chars first (set a "truncated": bool when you do). On any failure
    (tokenizer not found, tokenizers lib missing, encode error) return {"error": "..."}. Never raise.
    """
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return {"error": "tokenizers library not installed"}

    resolved = _resolve_tokenizer_file(path)
    if resolved is None:
        target = path if path is not None else "<default locations>"
        return {"error": f"tokenizer.json not found: {target}"}

    try:
        tokenizer = Tokenizer.from_file(str(resolved))
    except Exception as exc:  # noqa: BLE001 - never raise; surface as error dict
        return {"error": str(exc)}

    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    truncated = len(text) > max_chars
    clipped = text[:max_chars]

    # Empty/whitespace text -> zero tokens (not an error).
    if not clipped.strip():
        result = {
            "path": str(resolved),
            "count": 0,
            "ids": [],
            "tokens": [],
        }
        if truncated:
            result["truncated"] = True
        return result

    try:
        encoding = tokenizer.encode(clipped)
    except Exception as exc:  # noqa: BLE001 - never raise; surface as error dict
        return {"error": str(exc)}

    ids = list(encoding.ids)
    tokens = list(encoding.tokens)

    result = {
        "path": str(resolved),
        "count": len(ids),
        "ids": ids,
        "tokens": tokens,
    }
    if truncated:
        result["truncated"] = True
    return result
