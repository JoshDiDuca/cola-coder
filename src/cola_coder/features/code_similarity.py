"""Code Similarity Scorer: compare two code snippets using AST-based analysis.

Uses structural AST comparison rather than raw string matching to give a
semantically meaningful similarity score.  Falls back to token-level Jaccard
when AST parsing fails (e.g. TypeScript, incomplete snippets).

For a TS dev: like a structural diff beyond text — it understands that renaming
a variable doesn't make two functions fundamentally different.

Returns 0.0 (completely different) to 1.0 (structurally identical).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the code similarity scorer feature is active."""
    return FEATURE_ENABLED


@dataclass
class SimilarityResult:
    """Result of comparing two code snippets."""

    score: float  # 0.0 = different, 1.0 = identical
    method: str  # "ast", "token", or "combined"
    ast_score: float | None = None
    token_score: float | None = None
    details: str = ""

    def is_similar(self, threshold: float = 0.7) -> bool:
        """Return True if the score is at or above *threshold*."""
        return self.score >= threshold


class CodeSimilarity:
    """Compare code snippets for semantic/structural similarity.

    Tries AST comparison first (Python only), falls back to token Jaccard for
    TypeScript/JavaScript or unparseable code.

    Usage::

        sim = CodeSimilarity()
        result = sim.compare(snippet_a, snippet_b)
        print(f"Similarity: {result.score:.3f} via {result.method}")
    """

    def __init__(
        self,
        ast_weight: float = 0.6,
        token_weight: float = 0.4,
        token_ngram: int = 3,
    ) -> None:
        self.ast_weight = ast_weight
        self.token_weight = token_weight
        self.token_ngram = token_ngram

    def compare(
        self,
        code_a: str,
        code_b: str,
        language: str = "auto",
    ) -> SimilarityResult:
        """Compare two code snippets.

        Args:
            code_a: First code snippet.
            code_b: Second code snippet.
            language: ``"python"``, ``"typescript"``, ``"javascript"``, or
                ``"auto"`` (detect from content).

        Returns:
            SimilarityResult with score and method used.
        """
        if language == "auto":
            language = self._detect_language(code_a, code_b)

        ast_score: float | None = None
        if language == "python":
            ast_score = self._ast_similarity(code_a, code_b)

        token_score = self._token_similarity(code_a, code_b)

        if ast_score is not None:
            score = self.ast_weight * ast_score + self.token_weight * token_score
            method = "combined"
        else:
            score = token_score
            method = "token"

        return SimilarityResult(
            score=score,
            method=method,
            ast_score=ast_score,
            token_score=token_score,
            details=f"lang={language}",
        )

    # ------------------------------------------------------------------
    # AST-based comparison (Python)
    # ------------------------------------------------------------------

    def _ast_similarity(self, code_a: str, code_b: str) -> float | None:
        """Return structural similarity via AST node-type sequences."""
        try:
            tree_a = ast.parse(code_a)
            tree_b = ast.parse(code_b)
        except SyntaxError:
            return None

        seq_a = self._ast_sequence(tree_a)
        seq_b = self._ast_sequence(tree_b)

        if not seq_a and not seq_b:
            return 1.0
        if not seq_a or not seq_b:
            return 0.0

        # Jaccard on node-type n-grams
        ngrams_a = self._ngrams(seq_a, 2)
        ngrams_b = self._ngrams(seq_b, 2)
        return self._jaccard(ngrams_a, ngrams_b)

    @staticmethod
    def _ast_sequence(tree: ast.AST) -> list[str]:
        """Produce an ordered sequence of AST node type names."""
        return [type(node).__name__ for node in ast.walk(tree)]

    # ------------------------------------------------------------------
    # Token-level comparison
    # ------------------------------------------------------------------

    def _token_similarity(self, code_a: str, code_b: str) -> float:
        """Return token n-gram Jaccard similarity."""
        toks_a = self._tokenize(code_a)
        toks_b = self._tokenize(code_b)

        if not toks_a and not toks_b:
            return 1.0
        if not toks_a or not toks_b:
            return 0.0

        ng_a = self._ngrams(toks_a, self.token_ngram)
        ng_b = self._ngrams(toks_b, self.token_ngram)
        return self._jaccard(ng_a, ng_b)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(code: str) -> list[str]:
        """Split code into meaningful tokens (identifiers + operators)."""
        return re.findall(r"\w+|[^\w\s]", code)

    @staticmethod
    def _ngrams(seq: list[Any], n: int) -> frozenset[tuple[Any, ...]]:
        """Return n-gram frozenset from sequence."""
        if len(seq) < n:
            return frozenset([tuple(seq)]) if seq else frozenset()
        return frozenset(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))

    @staticmethod
    def _jaccard(a: frozenset, b: frozenset) -> float:  # type: ignore[type-arg]
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _detect_language(code_a: str, code_b: str) -> str:
        """Heuristically detect language from code content."""
        combined = code_a + code_b
        ts_indicators = [
            r"\bconst\b", r"\blet\b", r"\bvar\b", r"\binterface\b",
            r"\btype\b\s+\w+\s*=", r"=>", r"\bimport\b.*from",
        ]
        py_indicators = [
            r"^\s*def\s+\w+", r"^\s*class\s+\w+.*:", r"\bself\b",
            r"^\s*import\s+\w+", r"^\s*from\s+\w+\s+import",
        ]
        ts_count = sum(
            1 for p in ts_indicators if re.search(p, combined, re.MULTILINE)
        )
        py_count = sum(
            1 for p in py_indicators if re.search(p, combined, re.MULTILINE)
        )
        return "python" if py_count >= ts_count else "typescript"
