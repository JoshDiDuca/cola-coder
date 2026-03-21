"""Semantic Code Search: TF-IDF based search index over code snippets.

Builds a lightweight in-memory search index from code strings (functions,
classes, modules).  Uses TF-IDF weighting with token-overlap similarity to
find semantically related code without needing embeddings or a GPU.

For a TS dev: think of this as a local Lunr.js or Fuse.js index, but
specialised for code — it tokenises identifiers, keywords, and strings
rather than natural language words.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"[^a-zA-Z0-9_]+")
_CAMEL_RE = re.compile(r"([a-z])([A-Z])")


def _tokenise_code(text: str) -> list[str]:
    """Split code into lowercase token terms."""
    # Split camelCase: myVariable → my Variable
    text = _CAMEL_RE.sub(r"\1 \2", text)
    # Split on non-alphanumeric
    raw_tokens = _SPLIT_RE.split(text.lower())
    # Filter out very short tokens and pure digits
    return [t for t in raw_tokens if len(t) >= 2 and not t.isdigit()]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CodeDocument:
    """A single indexable code snippet."""

    doc_id: str  # unique identifier (e.g. "module.ClassName.method_name")
    source: str  # raw source code
    kind: str = "unknown"  # "function" | "class" | "module" | "unknown"
    tokens: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = _tokenise_code(self.source)


@dataclass
class SearchResult:
    """A single search result."""

    doc_id: str
    score: float
    kind: str
    snippet: str  # first 120 chars of source


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class SemanticSearchIndex:
    """TF-IDF search index over :class:`CodeDocument` objects."""

    def __init__(self) -> None:
        self._documents: dict[str, CodeDocument] = {}
        # IDF cache: recomputed when documents are added
        self._idf: dict[str, float] = {}
        self._dirty: bool = False

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def add(self, doc: CodeDocument) -> None:
        """Add a document to the index."""
        self._documents[doc.doc_id] = doc
        self._dirty = True

    def add_source(
        self,
        source: str,
        doc_id: str | None = None,
        kind: str = "unknown",
    ) -> list[CodeDocument]:
        """Parse *source* and add each top-level function/class as a document.

        Returns the list of documents added.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        docs: list[CodeDocument] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                snippet = ast.get_source_segment(source, node) or ""
                did = f"{doc_id or 'module'}.{name}" if doc_id else name
                doc = CodeDocument(doc_id=did, source=snippet or name, kind="function")
                self.add(doc)
                docs.append(doc)
            elif isinstance(node, ast.ClassDef):
                name = node.name
                snippet = ast.get_source_segment(source, node) or ""
                did = f"{doc_id or 'module'}.{name}" if doc_id else name
                doc = CodeDocument(doc_id=did, source=snippet or name, kind="class")
                self.add(doc)
                docs.append(doc)
        return docs

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Find the *top_k* most relevant documents for *query*."""
        if not self._documents:
            return []

        if self._dirty:
            self._recompute_idf()

        query_tokens = set(_tokenise_code(query))
        if not query_tokens:
            return []

        scores: list[tuple[str, float]] = []
        for doc_id, doc in self._documents.items():
            score = self._tfidf_score(query_tokens, doc)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for doc_id, score in scores[:top_k]:
            doc = self._documents[doc_id]
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    score=score,
                    kind=doc.kind,
                    snippet=doc.source[:120],
                )
            )
        return results

    def size(self) -> int:
        return len(self._documents)

    # ------------------------------------------------------------------
    # TF-IDF computation
    # ------------------------------------------------------------------

    def _recompute_idf(self) -> None:
        n = len(self._documents)
        if n == 0:
            self._idf = {}
            return

        # document frequency per term
        df: dict[str, int] = {}
        for doc in self._documents.values():
            for term in set(doc.tokens):
                df[term] = df.get(term, 0) + 1

        self._idf = {
            term: math.log((n + 1) / (freq + 1)) + 1.0
            for term, freq in df.items()
        }
        self._dirty = False

    def _tfidf_score(self, query_tokens: set[str], doc: CodeDocument) -> float:
        """Cosine-like TF-IDF score between query and document."""
        total_terms = len(doc.tokens)
        if total_terms == 0:
            return 0.0

        # Term frequency in document
        tf: dict[str, float] = {}
        for term in doc.tokens:
            tf[term] = tf.get(term, 0) + 1
        for term in tf:
            tf[term] = tf[term] / total_terms

        score = 0.0
        for term in query_tokens:
            if term in tf:
                idf = self._idf.get(term, 1.0)
                score += tf[term] * idf

        return score
