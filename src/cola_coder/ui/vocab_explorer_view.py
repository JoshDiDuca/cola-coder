"""Tokenizer vocab-explorer endpoint helper for the local UI.

Read-only search over the trained tokenizer's vocabulary. Loads the project
tokenizer (HuggingFace ``tokenizers.Tokenizer``) via the same path resolution as
:mod:`cola_coder.ui.tokenizer_info`, then filters the vocab dict (piece -> id)
by a substring query and returns the matching pieces plus the full list of
special/added tokens.

Substring matching is **case-sensitive** — byte-level BPE pieces are
case-significant (and carry the ``Ġ`` space marker), so a case-insensitive match
would be misleading. Robust to a missing tokenizer or an absent ``tokenizers``
package: returns an ``{"error": ...}`` dict, never raises.
"""

from __future__ import annotations

import logging

from .tokenizer_info import _resolve_tokenizer_file

logger = logging.getLogger(__name__)


def vocab_search(query: str = "", path: str | None = None, limit: int = 200) -> dict:
    """Search the resolved tokenizer's vocabulary for pieces containing ``query``.

    ``path`` may point at a ``tokenizer.json`` file OR its containing dir; when
    None the default locations are probed (storage.yaml, ``data/<dataset>/`` …).

    Behaviour:
      * Empty ``query`` → return the first ``limit`` tokens by id (so the panel
        shows something on load); ``total_matches`` is the full vocab size.
      * Non-empty ``query`` → case-sensitive substring filter over the pieces,
        sorted by id and capped at ``limit``; ``truncated`` / ``total_matches``
        reflect the pre-cap count.
      * ``special_tokens`` (added/special tokens) are always returned in full;
        derived from ``get_added_tokens_decoder()`` when available, else ``[]``.

    Returns a dict matching :class:`cola_coder.ui.schemas.VocabSearchResult`, or
    ``{"error": ...}`` on a missing tokenizer / absent ``tokenizers`` package.
    Never raises.
    """
    resolved = _resolve_tokenizer_file(path)
    if resolved is None:
        target = path if path is not None else "<default locations>"
        return {"error": f"tokenizer.json not found: {target}"}

    try:
        from tokenizers import Tokenizer  # local import: heavy, optional dependency
    except ImportError:
        return {"error": "tokenizers package not installed (pip install tokenizers)"}

    try:
        tokenizer = Tokenizer.from_file(str(resolved))
    except Exception as exc:  # malformed tokenizer file, etc.
        logger.warning("failed to load tokenizer %s: %s", resolved, exc)
        return {"error": str(exc)}

    vocab: dict[str, int] = tokenizer.get_vocab()
    vocab_size: int = tokenizer.get_vocab_size()

    # Special / added tokens — ids of the tokens the tokenizer reports as added.
    special_ids: set[int] = set()
    special_tokens: list[dict] = []
    decoder = getattr(tokenizer, "get_added_tokens_decoder", None)
    try:
        added = decoder() if callable(decoder) else {}
    except Exception as exc:  # older tokenizers / unexpected accessor failure
        logger.debug("get_added_tokens_decoder failed: %s", exc)
        added = {}
    for token_id, added_token in sorted(added.items()):
        piece = getattr(added_token, "content", None)
        if not isinstance(piece, str):
            piece = str(added_token)
        special_ids.add(int(token_id))
        special_tokens.append({"id": int(token_id), "piece": piece, "is_special": True})

    # Filter the vocab by substring (or take the head when the query is empty).
    if query == "":
        matches = sorted(vocab.items(), key=lambda kv: kv[1])
        total_matches = vocab_size
    else:
        matches = sorted(
            ((piece, tid) for piece, tid in vocab.items() if query in piece),
            key=lambda kv: kv[1],
        )
        total_matches = len(matches)

    capped = matches[:limit]
    tokens: list[dict] = [
        {"id": tid, "piece": piece, "is_special": tid in special_ids}
        for piece, tid in capped
    ]

    return {
        "query": query,
        "vocab_size": vocab_size,
        "total_matches": total_matches,
        "truncated": total_matches > len(capped),
        "tokens": tokens,
        "special_tokens": special_tokens,
    }
