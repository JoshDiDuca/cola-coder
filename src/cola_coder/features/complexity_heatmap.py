"""Code Complexity Heatmap: generate per-line complexity scores for code.

Maps AST node types to complexity weights and produces a heatmap data
structure that can be used for difficulty-aware training (e.g. weighting
loss by line complexity, or highlighting hard regions for curriculum learning).

Works for Python source code using the built-in `ast` module.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the complexity heatmap feature is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# AST node → complexity weight mapping
# ---------------------------------------------------------------------------

# Higher weight = more complex.  These are heuristic scores.
NODE_WEIGHTS: dict[str, float] = {
    # Control flow
    "If": 2.0,
    "For": 2.0,
    "While": 2.5,
    "Try": 3.0,
    "ExceptHandler": 2.0,
    "With": 1.5,
    "Match": 3.0,
    "MatchCase": 1.5,
    # Comprehensions / generators
    "ListComp": 1.5,
    "SetComp": 1.5,
    "DictComp": 2.0,
    "GeneratorExp": 1.5,
    # Functions & classes
    "FunctionDef": 1.0,
    "AsyncFunctionDef": 1.5,
    "ClassDef": 1.0,
    "Lambda": 2.0,
    # Expressions
    "BoolOp": 1.0,
    "BinOp": 0.5,
    "UnaryOp": 0.5,
    "Compare": 0.5,
    "Call": 0.5,
    "Yield": 1.5,
    "YieldFrom": 1.5,
    "Await": 1.5,
    # Assignments
    "AugAssign": 0.5,
    "AnnAssign": 0.3,
    "Assign": 0.2,
    "NamedExpr": 1.0,
    # Imports
    "Import": 0.2,
    "ImportFrom": 0.2,
    # Default (not listed)
    "__default__": 0.1,
}


@dataclass
class LineComplexity:
    """Complexity data for a single source line."""

    line_number: int  # 1-based
    source: str  # Original source text
    score: float  # Complexity score (higher = harder)
    node_types: list[str] = field(default_factory=list)  # AST nodes on this line


@dataclass
class ComplexityHeatmap:
    """Heatmap of complexity scores across all lines in a code snippet."""

    lines: list[LineComplexity] = field(default_factory=list)
    total_score: float = 0.0
    max_score: float = 0.0
    mean_score: float = 0.0
    hottest_line: Optional[int] = None  # 1-based line number

    def normalized_scores(self) -> list[float]:
        """Return scores in [0, 1] relative to the maximum."""
        if self.max_score == 0:
            return [0.0] * len(self.lines)
        return [lc.score / self.max_score for lc in self.lines]

    def as_records(self) -> list[dict]:
        return [
            {
                "line": lc.line_number,
                "source": lc.source,
                "score": lc.score,
                "node_types": lc.node_types,
            }
            for lc in self.lines
        ]

    def top_n_lines(self, n: int = 5) -> list[LineComplexity]:
        """Return the n most complex lines."""
        return sorted(self.lines, key=lambda lc: lc.score, reverse=True)[:n]


class ComplexityHeatmapGenerator:
    """Generate per-line complexity heatmaps from Python source code."""

    def __init__(self, node_weights: Optional[dict[str, float]] = None) -> None:
        self.weights = node_weights if node_weights is not None else dict(NODE_WEIGHTS)

    def _weight(self, node_type: str) -> float:
        return self.weights.get(node_type, self.weights.get("__default__", 0.1))

    def generate(self, code: str) -> ComplexityHeatmap:
        """Parse code and produce a per-line complexity heatmap.

        Parameters
        ----------
        code:
            Python source code as a string.

        Returns
        -------
        ComplexityHeatmap with per-line scores.
        """
        source_lines = code.splitlines()
        n_lines = len(source_lines)

        # Initialize per-line accumulators
        line_scores: list[float] = [0.0] * (n_lines + 1)  # 1-indexed
        line_nodes: list[list[str]] = [[] for _ in range(n_lines + 1)]

        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Return a flat heatmap if the code doesn't parse
            heatmap_lines = [
                LineComplexity(line_number=i + 1, source=source_lines[i], score=0.0)
                for i in range(n_lines)
            ]
            return ComplexityHeatmap(
                lines=heatmap_lines,
                total_score=0.0,
                max_score=0.0,
                mean_score=0.0,
                hottest_line=None,
            )

        # Walk AST and accumulate weights per line
        for node in ast.walk(tree):
            node_type = type(node).__name__
            weight = self._weight(node_type)
            if hasattr(node, "lineno"):
                lineno = node.lineno
                if 1 <= lineno <= n_lines:
                    line_scores[lineno] += weight
                    line_nodes[lineno].append(node_type)

        # Build LineComplexity objects
        heatmap_lines = []
        for i, src in enumerate(source_lines):
            lineno = i + 1
            heatmap_lines.append(
                LineComplexity(
                    line_number=lineno,
                    source=src,
                    score=line_scores[lineno],
                    node_types=line_nodes[lineno],
                )
            )

        total = sum(lc.score for lc in heatmap_lines)
        max_score = max((lc.score for lc in heatmap_lines), default=0.0)
        mean_score = total / n_lines if n_lines > 0 else 0.0
        hottest = None
        if heatmap_lines:
            hottest = max(heatmap_lines, key=lambda lc: lc.score).line_number

        return ComplexityHeatmap(
            lines=heatmap_lines,
            total_score=total,
            max_score=max_score,
            mean_score=mean_score,
            hottest_line=hottest,
        )

    def compare(self, code_a: str, code_b: str) -> dict:
        """Compare heatmaps for two code snippets.

        Returns a dict with keys: a_total, b_total, diff, more_complex.
        """
        hm_a = self.generate(code_a)
        hm_b = self.generate(code_b)
        diff = hm_b.total_score - hm_a.total_score
        return {
            "a_total": hm_a.total_score,
            "b_total": hm_b.total_score,
            "diff": diff,
            "more_complex": "b" if diff > 0 else ("a" if diff < 0 else "equal"),
        }

    def bucket_lines(
        self,
        heatmap: ComplexityHeatmap,
        n_buckets: int = 5,
    ) -> list[list[LineComplexity]]:
        """Distribute lines into n equal-width score buckets.

        Returns a list of n lists (bucket 0 = simplest).
        """
        if not heatmap.lines or heatmap.max_score == 0:
            return [list(heatmap.lines)] + [[] for _ in range(n_buckets - 1)]

        buckets: list[list[LineComplexity]] = [[] for _ in range(n_buckets)]
        for lc in heatmap.lines:
            # Which bucket?
            frac = lc.score / heatmap.max_score
            idx = min(int(frac * n_buckets), n_buckets - 1)
            buckets[idx].append(lc)
        return buckets
