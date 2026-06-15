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
import math
import re
from collections import Counter
from pathlib import Path

from . import retrieval_stats_view as rsv

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_SNIPPET_CHARS = 240
_SOURCE_KEYS = ("file_path", "path", "repo_name", "source", "id")

# BM25 (Okapi) parameters — the standard lexical-ranking baseline. k1 controls
# term-frequency saturation, b controls document-length normalization.
_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def _token_list(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _bm25_scores(query_tokens: set[str], doc_tokens: list[list[str]]) -> list[float]:
    """Okapi BM25 score per document for the query terms.

    Rewards rare query terms (IDF) and saturates repeated terms, with
    document-length normalization — strictly better ranking than raw token
    overlap, which weights every term equally and ignores length. Pure/no model.
    """
    n_docs = len(doc_tokens)
    if n_docs == 0:
        return []
    doc_freq: dict[str, int] = {}
    for toks in doc_tokens:
        for term in set(toks) & query_tokens:
            doc_freq[term] = doc_freq.get(term, 0) + 1
    avgdl = sum(len(toks) for toks in doc_tokens) / n_docs
    idf = {
        term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        for term, df in doc_freq.items()
    }

    scores: list[float] = []
    for toks in doc_tokens:
        tf = Counter(toks)
        dl = len(toks)
        norm = _BM25_K1 * (1 - _BM25_B + _BM25_B * (dl / avgdl if avgdl > 0 else 1.0))
        score = 0.0
        for term in query_tokens:
            freq = tf.get(term, 0)
            if freq and term in idf:
                score += idf[term] * (freq * (_BM25_K1 + 1)) / (freq + norm)
        scores.append(score)
    return scores


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
    # BM25 over the indexed chunks (non-str entries become empty docs → score 0).
    doc_tokens = [_token_list(t) if isinstance(t, str) else [] for t in texts]
    bm25 = _bm25_scores(q_tokens, doc_tokens)
    # Small bonus when the full query appears verbatim (phrase match beats bag-of-words).
    raw_scores: list[float] = []
    for i, text in enumerate(texts):
        score = bm25[i]
        if isinstance(text, str) and q_lower and q_lower in text.lower():
            score += 0.5
        raw_scores.append(score)

    scored = [(s, i) for i, s in enumerate(raw_scores) if s > 0]
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    # Normalize the displayed 0-1 score relative to the top hit (BM25 is unbounded).
    top_score = scored[0][0] if scored else 1.0
    hits = []
    for score, i in scored[: max(0, top_k)]:
        meta = metadata[i] if i < len(metadata) else {}
        hits.append(
            {
                "id": str(ids[i]) if i < len(ids) else str(i),
                "score": round(score / top_score if top_score > 0 else 0.0, 4),
                "snippet": _snippet(texts[i], query),
                "source": _source_of(meta),
            }
        )

    return {"query": query, "exists": True, "total_indexed": total, "hits": hits}
