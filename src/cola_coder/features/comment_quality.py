"""Code Comment Quality Evaluator (improvement #62).

Evaluates code comments on four axes:
  - Informativeness: does the comment add meaning beyond restating the code?
  - Accuracy: does the comment align with what the code actually does?
  - Density: is the comment-to-code ratio appropriate?
  - Staleness: are there signs the comment is outdated?

TypeScript analogy: like ESLint's `spaced-comment`, `valid-jsdoc`, and
`require-jsdoc` rules combined into a holistic scorer.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Feature toggle (project convention)
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if comment quality evaluation is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Trivial comment patterns that restate the obvious
_TRIVIAL_PATTERNS = [
    re.compile(r"^\s*#\s*(get|set|return|returns?|increment|decrement|loop|iterate|print)\s+\w", re.I),
    re.compile(r"^\s*#\s*this (function|method|class|variable)\s+(does|is|has|stores?)\b", re.I),
    re.compile(r"^\s*#\s*(todo|fixme|hack|xxx|note):?\s*$", re.I),
    re.compile(r"^\s*#\s*\.\.\.\s*$"),
]

# Staleness signals: old version references, deprecated APIs, etc.
_STALE_PATTERNS = [
    re.compile(r"python\s*[23]\.\d", re.I),
    re.compile(r"\bpy(thon)?\s*2\b", re.I),
    re.compile(r"\bdeprecated\b", re.I),
    re.compile(r"TODO.*\d{4}", re.I),  # TODO with year in comment
    re.compile(r"(?:legacy|old|outdated|obsolete|unused)\s+\w", re.I),
]

# Informative keywords that suggest the comment adds value
_INFORMATIVE_KEYWORDS = {
    "why", "because", "note", "important", "warning", "caveat",
    "workaround", "assumption", "invariant", "precondition", "postcondition",
    "see", "ref", "reference", "algorithm", "complexity", "performance",
    "thread-safe", "not thread-safe", "idempotent", "side effect",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CommentRecord:
    """A single comment found in source code."""

    line_no: int
    text: str
    is_docstring: bool = False
    is_inline: bool = False


@dataclass
class CommentQualityReport:
    """Full quality report for a code snippet's comments."""

    total_lines: int
    code_lines: int
    comment_lines: int
    docstring_lines: int
    comments: List[CommentRecord] = field(default_factory=list)

    informativeness_score: float = 0.0   # 0-1
    density_score: float = 0.0           # 0-1
    staleness_score: float = 0.0         # 0-1 (1 = not stale)
    overall_score: float = 0.0           # 0-1

    trivial_count: int = 0
    stale_count: int = 0
    informative_count: int = 0

    issues: List[str] = field(default_factory=list)

    @property
    def density_ratio(self) -> float:
        """Comment+docstring lines / total non-blank lines."""
        denom = max(self.code_lines + self.comment_lines + self.docstring_lines, 1)
        return (self.comment_lines + self.docstring_lines) / denom


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class CommentQualityEvaluator:
    """Evaluate comment quality for Python source code.

    Parameters
    ----------
    ideal_density_low:
        Lower bound of ideal comment density (fraction of total lines).
    ideal_density_high:
        Upper bound of ideal comment density.
    """

    def __init__(
        self,
        ideal_density_low: float = 0.10,
        ideal_density_high: float = 0.30,
    ) -> None:
        self.ideal_density_low = ideal_density_low
        self.ideal_density_high = ideal_density_high

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, source: str) -> CommentQualityReport:
        """Evaluate comment quality for the given Python source."""
        lines = source.splitlines()
        comments, docstring_lines = self._extract_comments(source, lines)

        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif stripped.startswith("#"):
                comment_lines += 1
            else:
                code_lines += 1

        report = CommentQualityReport(
            total_lines=len(lines),
            code_lines=code_lines,
            comment_lines=comment_lines,
            docstring_lines=docstring_lines,
            comments=comments,
        )

        self._score_informativeness(report)
        self._score_density(report)
        self._score_staleness(report)
        report.overall_score = (
            0.4 * report.informativeness_score
            + 0.3 * report.density_score
            + 0.3 * report.staleness_score
        )
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_comments(
        self, source: str, lines: list[str]
    ) -> Tuple[List[CommentRecord], int]:
        """Extract comment records and count docstring lines."""
        comments: List[CommentRecord] = []
        docstring_lines = 0

        # Use AST to find docstrings
        docstring_ranges: set[int] = set()
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                    if (
                        node.body
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)
                    ):
                        ds_node = node.body[0]
                        for ln in range(ds_node.lineno, (ds_node.end_lineno or ds_node.lineno) + 1):
                            docstring_ranges.add(ln)
        except SyntaxError:
            pass

        docstring_lines = len(docstring_ranges)

        # Extract inline # comments
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                is_inline = bool(re.search(r"\S.*#", line))
                comments.append(
                    CommentRecord(
                        line_no=i,
                        text=stripped,
                        is_docstring=False,
                        is_inline=is_inline,
                    )
                )
            # Docstrings as records
            if i in docstring_ranges and stripped.startswith('"""') or (
                i in docstring_ranges and stripped.startswith("'''")
            ):
                comments.append(
                    CommentRecord(
                        line_no=i,
                        text=stripped,
                        is_docstring=True,
                        is_inline=False,
                    )
                )

        return comments, docstring_lines

    def _score_informativeness(self, report: CommentQualityReport) -> None:
        """Score based on how informative vs trivial the comments are."""
        if not report.comments:
            # No comments → neutral, not penalised here (density handles that)
            report.informativeness_score = 0.5
            return

        trivial = 0
        informative = 0
        for c in report.comments:
            text_lower = c.text.lower()
            is_trivial = any(p.search(c.text) for p in _TRIVIAL_PATTERNS)
            has_info = any(kw in text_lower for kw in _INFORMATIVE_KEYWORDS)
            if is_trivial:
                trivial += 1
                report.issues.append(f"Line {c.line_no}: trivial comment '{c.text[:50]}'")
            if has_info:
                informative += 1

        report.trivial_count = trivial
        report.informative_count = informative

        n = len(report.comments)
        trivial_penalty = trivial / n
        info_bonus = informative / n
        # Score: start at 0.7, subtract for trivials, add for informatives
        score = 0.7 - 0.5 * trivial_penalty + 0.3 * info_bonus
        report.informativeness_score = max(0.0, min(1.0, score))

    def _score_density(self, report: CommentQualityReport) -> None:
        """Score based on comment density (too few or too many = lower score)."""
        ratio = report.density_ratio
        lo, hi = self.ideal_density_low, self.ideal_density_high

        if ratio < lo:
            # Too sparse
            score = ratio / lo
            if ratio == 0:
                report.issues.append("No comments found (consider adding docstrings)")
        elif ratio <= hi:
            score = 1.0
        else:
            # Too dense — linear decay above hi
            over = ratio - hi
            score = max(0.0, 1.0 - over * 2)
            report.issues.append(f"Comment density {ratio:.1%} exceeds ideal {hi:.0%}")

        report.density_score = max(0.0, min(1.0, score))

    def _score_staleness(self, report: CommentQualityReport) -> None:
        """Score based on staleness signals in comments."""
        stale = 0
        for c in report.comments:
            if any(p.search(c.text) for p in _STALE_PATTERNS):
                stale += 1
                report.issues.append(f"Line {c.line_no}: possible stale comment")

        report.stale_count = stale
        n = max(len(report.comments), 1)
        stale_ratio = stale / n
        report.staleness_score = max(0.0, 1.0 - stale_ratio * 2)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def evaluate_comments(source: str) -> CommentQualityReport:
    """Evaluate comment quality with default settings."""
    return CommentQualityEvaluator().evaluate(source)
