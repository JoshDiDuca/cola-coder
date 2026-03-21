"""Code Readability Scorer — feature 50.

Scores code readability using five dimensions:

1. **Naming quality**: Short/cryptic identifier names reduce readability.
   Measures average identifier length; penalises very short names (< 3 chars).

2. **Indentation consistency**: Mixed tabs/spaces or irregular indent sizes
   are a readability red flag.

3. **Comment density**: Ratio of comment lines to code lines.  Very low
   density (no documentation) and very high density (over-commented) both
   reduce scores.

4. **Function length**: Long functions are harder to read.  Penalises
   functions over ``max_function_lines``.

5. **Cyclomatic complexity per function**: Counts branching decisions (if,
   for, while, and, or, etc.).  High complexity → hard to follow.

Feature toggle pattern (project convention):
    FEATURE_ENABLED = False → scorer returns a neutral ReadabilityReport.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if readability scoring is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FunctionReadability:
    """Readability metrics for a single function."""

    name: str
    line_count: int
    cyclomatic_complexity: int
    avg_identifier_length: float
    has_docstring: bool

    def is_readable(
        self,
        max_lines: int = 50,
        max_complexity: int = 10,
    ) -> bool:
        return self.line_count <= max_lines and self.cyclomatic_complexity <= max_complexity


@dataclass
class ReadabilityReport:
    """Aggregate readability metrics for a code snippet."""

    naming_score: float = 0.0
    """0.0–1.0: quality of identifier names."""
    indentation_score: float = 0.0
    """0.0–1.0: consistency of indentation."""
    comment_density_score: float = 0.0
    """0.0–1.0: appropriate comment density."""
    function_length_score: float = 0.0
    """0.0–1.0: function length appropriateness."""
    complexity_score: float = 0.0
    """0.0–1.0: inverse cyclomatic complexity."""

    overall_score: float = 0.0
    """Weighted composite of all sub-scores."""

    function_reports: List[FunctionReadability] = field(default_factory=list)
    total_lines: int = 0
    comment_lines: int = 0
    code_lines: int = 0
    avg_identifier_length: float = 0.0
    indentation_style: str = "unknown"  # "spaces", "tabs", "mixed", "none"
    parse_error: Optional[str] = None

    def summary(self) -> str:
        if self.parse_error:
            return f"ReadabilityReport: parse_error={self.parse_error}"
        return (
            f"ReadabilityReport: overall={self.overall_score:.3f} "
            f"naming={self.naming_score:.2f} "
            f"indent={self.indentation_score:.2f} "
            f"comments={self.comment_density_score:.2f} "
            f"fn_len={self.function_length_score:.2f} "
            f"complexity={self.complexity_score:.2f}"
        )

    def grade(self) -> str:
        """Letter grade based on overall score."""
        s = self.overall_score
        if s >= 0.85:
            return "A"
        if s >= 0.70:
            return "B"
        if s >= 0.55:
            return "C"
        if s >= 0.40:
            return "D"
        return "F"


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class ReadabilityScorer:
    """Scores Python code readability across 5 dimensions.

    Only Python is supported for AST-based metrics (naming, complexity,
    function length).  Non-Python code gets line-based metrics only.
    """

    def __init__(
        self,
        max_function_lines: int = 50,
        max_cyclomatic: int = 10,
        ideal_comment_density: float = 0.15,
        min_identifier_length: int = 3,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Args:
            max_function_lines: Functions longer than this are penalised.
            max_cyclomatic: Complexity above this threshold is penalised.
            ideal_comment_density: Target comment/code ratio (default 15%).
            min_identifier_length: Names shorter than this count as "cryptic".
            weights: Override default sub-score weights.
        """
        self.max_function_lines = max_function_lines
        self.max_cyclomatic = max_cyclomatic
        self.ideal_comment_density = ideal_comment_density
        self.min_identifier_length = min_identifier_length
        self.weights = weights or {
            "naming": 0.25,
            "indentation": 0.20,
            "comment_density": 0.15,
            "function_length": 0.20,
            "complexity": 0.20,
        }

    def score(self, code: str) -> ReadabilityReport:
        """Score a Python code snippet.

        Args:
            code: Source code string.

        Returns:
            ReadabilityReport with sub-scores and overall score.
        """
        if not FEATURE_ENABLED:
            return ReadabilityReport(overall_score=0.5)

        report = ReadabilityReport()

        # Line-based analysis (works for any language)
        lines = code.splitlines()
        report.total_lines = len(lines)
        report.comment_lines, report.code_lines = _count_line_types(lines)
        report.indentation_style = _detect_indentation_style(lines)
        report.indentation_score = _score_indentation(lines)
        report.comment_density_score = _score_comment_density(
            report.comment_lines,
            report.code_lines,
            self.ideal_comment_density,
        )

        # AST-based analysis (Python only)
        try:
            tree = ast.parse(code)
            functions = _extract_functions(tree)
            report.function_reports = [
                _analyse_function(fn, code, self.min_identifier_length)
                for fn in functions
            ]
            identifiers = _collect_identifiers(tree)
            if identifiers:
                report.avg_identifier_length = sum(len(i) for i in identifiers) / len(
                    identifiers
                )
            else:
                report.avg_identifier_length = 0.0

            report.naming_score = _score_naming(
                identifiers, self.min_identifier_length
            )
            report.function_length_score = _score_function_lengths(
                report.function_reports, self.max_function_lines
            )
            report.complexity_score = _score_complexity(
                report.function_reports, self.max_cyclomatic
            )

        except SyntaxError as e:
            report.parse_error = str(e)
            report.naming_score = 0.5
            report.function_length_score = 0.5
            report.complexity_score = 0.5

        # Weighted composite
        w = self.weights
        report.overall_score = round(
            w["naming"] * report.naming_score
            + w["indentation"] * report.indentation_score
            + w["comment_density"] * report.comment_density_score
            + w["function_length"] * report.function_length_score
            + w["complexity"] * report.complexity_score,
            4,
        )
        return report


# ---------------------------------------------------------------------------
# Line-based metrics
# ---------------------------------------------------------------------------


def _count_line_types(lines: List[str]) -> Tuple[int, int]:
    """Return (comment_lines, code_lines)."""
    comment_lines = 0
    code_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            comment_lines += 1
        else:
            code_lines += 1
    return comment_lines, code_lines


def _detect_indentation_style(lines: List[str]) -> str:
    has_spaces = False
    has_tabs = False
    for line in lines:
        if line and line[0] == " ":
            has_spaces = True
        elif line and line[0] == "\t":
            has_tabs = True
    if has_spaces and has_tabs:
        return "mixed"
    if has_spaces:
        return "spaces"
    if has_tabs:
        return "tabs"
    return "none"


def _score_indentation(lines: List[str]) -> float:
    """Score indentation consistency."""
    style = _detect_indentation_style(lines)
    if style == "mixed":
        return 0.2  # heavily penalise mixed
    if style in ("spaces", "tabs", "none"):
        # Check for consistent indent size (if spaces)
        if style == "spaces":
            indent_sizes = set()
            prev_indent = 0
            for line in lines:
                if not line.strip():
                    continue
                indent = len(line) - len(line.lstrip(" "))
                if indent > prev_indent:
                    indent_sizes.add(indent - prev_indent)
                prev_indent = indent
            indent_sizes.discard(0)
            if len(indent_sizes) <= 1:
                return 1.0
            if len(indent_sizes) == 2:
                return 0.8
            return 0.5
        return 1.0
    return 0.5


def _score_comment_density(
    comment_lines: int,
    code_lines: int,
    ideal: float,
) -> float:
    """Score how close the comment density is to the ideal."""
    total = comment_lines + code_lines
    if total == 0:
        return 0.5
    density = comment_lines / total
    # Distance from ideal: 0 = perfect, 1 = worst
    distance = abs(density - ideal) / max(ideal, 1 - ideal)
    return max(0.0, 1.0 - distance)


# ---------------------------------------------------------------------------
# AST-based metrics
# ---------------------------------------------------------------------------


def _collect_identifiers(tree: ast.AST) -> List[str]:
    """Collect all user-defined identifier names from an AST."""
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
            for arg in node.args.args:
                names.append(arg.arg)
        elif isinstance(node, ast.ClassDef):
            names.append(node.name)
    return names


def _score_naming(identifiers: List[str], min_length: int) -> float:
    """Score naming quality: penalise short/cryptic names."""
    if not identifiers:
        return 0.5
    # Exclude common short names that are conventional
    allowed_short = {"i", "j", "k", "x", "y", "z", "n", "f", "e", "v", "_"}
    cryptic = sum(
        1 for name in identifiers
        if len(name) < min_length and name not in allowed_short
    )
    cryptic_ratio = cryptic / len(identifiers)
    return max(0.0, 1.0 - cryptic_ratio * 2)


def _extract_functions(tree: ast.AST) -> List[ast.FunctionDef]:
    """Return all function definitions (including async) from an AST."""
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node)
    return funcs


def _analyse_function(
    node: ast.FunctionDef,
    source: str,
    min_identifier_length: int,
) -> FunctionReadability:
    """Compute per-function readability metrics."""
    # Line count
    try:
        line_count = node.end_lineno - node.lineno + 1
    except AttributeError:
        # Python < 3.8 may not have end_lineno
        line_count = max(1, len(ast.dump(node)) // 80)

    # Cyclomatic complexity (count decision points + 1)
    complexity = 1
    for child in ast.walk(node):
        if isinstance(
            child,
            (
                ast.If,
                ast.For,
                ast.While,
                ast.ExceptHandler,
                ast.With,
                ast.comprehension,
            ),
        ):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1

    # Average identifier length in this function
    names = [
        arg.arg for arg in node.args.args
    ] + [node.name]
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.append(child.id)
    avg_len = sum(len(n) for n in names) / max(len(names), 1)

    # Docstring
    has_docstring = (
        bool(node.body)
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    )

    return FunctionReadability(
        name=node.name,
        line_count=line_count,
        cyclomatic_complexity=complexity,
        avg_identifier_length=avg_len,
        has_docstring=has_docstring,
    )


def _score_function_lengths(
    functions: List[FunctionReadability],
    max_lines: int,
) -> float:
    """Score: penalise long functions."""
    if not functions:
        return 1.0
    scores = []
    for fn in functions:
        if fn.line_count <= max_lines:
            scores.append(1.0)
        else:
            # Penalty increases with length
            ratio = max_lines / fn.line_count
            scores.append(max(0.0, ratio))
    return sum(scores) / len(scores)


def _score_complexity(
    functions: List[FunctionReadability],
    max_complexity: int,
) -> float:
    """Score: penalise high cyclomatic complexity."""
    if not functions:
        return 1.0
    scores = []
    for fn in functions:
        if fn.cyclomatic_complexity <= max_complexity:
            scores.append(1.0)
        else:
            ratio = max_complexity / fn.cyclomatic_complexity
            scores.append(max(0.0, ratio))
    return sum(scores) / len(scores)
