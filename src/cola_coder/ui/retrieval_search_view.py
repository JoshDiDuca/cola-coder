"""Lexical search over the persisted retrieval index, for the local UI.

The CLI's "Semantic Search" embeds the query and does cosine search; that needs
the exact embedder the index was built with (model identity + a GPU load), so it
is deferred. This endpoint does the MAIN-SAFE half: a pure lexical rank over the
``texts`` already stored in the index sidecar (``data/vector_index.json`` — see
``retrieval_stats_view``). No model, no GPU, no network — so it never contends
with the live trainer and works the instant an index exists.

Ranking: case-insensitive query-token overlap (fraction of distinct query tokens
present in the chunk) plus a small bonus when the full query appears as a
substring. Returns the top-k chunks with a snippet + source metadata. A missing
index yields a valid empty result (``exists=False``), never an error.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from . import retrieval_stats_view as rsv

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_SNIPPET_CHARS = 240
_SOURCE_KEYS = ("file_path", "path", "repo_name", "source", "id")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def _source_of(meta: object) -> str | None:
    if not isinstance(meta, dict):
        return None
    for key in _SOURCE_KEYS:
        value = meta.get(key)
        if value:
            return str(value)
    return None


def _snippet(text: str, query: str) -> str:
    """A short window of ``text`` centered on the first query-substring hit."""
    lowered = text.lower()
    pos = lowered.find(query.lower())
    if pos < 0:
        return text[:_SNIPPET_CHARS].strip()
    start = max(0, pos - _SNIPPET_CHARS // 3)
    return text[start : start + _SNIPPET_CHARS].strip()


def search_index(query: str, top_k: int = 10, root: str | None = None) -> dict:
    """Lexically rank indexed chunks against ``query``. Never raises.

    Returns a ``RetrievalSearchResult`` dict: query, exists, total_indexed, hits.
    """
    query = query.strip()
    base_dir = (Path(root) if root else rsv._project_root()) / rsv._INDEX_REL_DIR
    store_base = rsv._find_store_base(base_dir)
    if store_base is None:
        return {"query": query, "exists": False, "total_indexed": 0, "hits": []}

    sidecar = store_base.with_suffix(".json")
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"error": f"could not read index: {exc}"}

    texts = data.get("texts") if isinstance(data, dict) else None
    if not isinstance(texts, list):
        return {"query": query, "exists": True, "total_indexed": 0, "hits": []}
    ids = data.get("ids") if isinstance(data.get("ids"), list) else []
    metadata = data.get("metadata") if isinstance(data.get("metadata"), list) else []

    total = len(texts)
    if not query:
        return {"query": query, "exists": True, "total_indexed": total, "hits": []}

    q_tokens = _tokens(query)
    q_lower = query.lower()
    scored: list[tuple[float, int]] = []
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            continue
        doc_tokens = _tokens(text)
        if not q_tokens:
            overlap = 0.0
        else:
            overlap = len(q_tokens & doc_tokens) / len(q_tokens)
        substring_bonus = 0.5 if q_lower in text.lower() else 0.0
        score = overlap + substring_bonus
        if score > 0:
            scored.append((score, i))

    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    hits = []
    for score, i in scored[: max(0, top_k)]:
        meta = metadata[i] if i < len(metadata) else {}
        hits.append(
            {
                "id": str(ids[i]) if i < len(ids) else str(i),
                "score": round(min(1.0, score), 4),
                "snippet": _snippet(texts[i], query),
                "source": _source_of(meta),
            }
        )

    return {"query": query, "exists": True, "total_indexed": total, "hits": hits}
