"""Type Annotation Scorer (improvement #63).

Scores Python type annotation coverage and quality.
Detects: missing return types, untyped parameters, use of Any,
incomplete generics (e.g. List without element type), and more.

TypeScript analogy: like tsc's --strict mode, but outputting a 0.0-1.0
coverage score rather than hard errors.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Feature toggle (project convention)
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if type annotation scoring is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AnnotationIssue:
    """A single annotation problem found in source."""

    line_no: int
    name: str
    kind: str   # "missing_return", "untyped_param", "any_usage", "incomplete_generic"
    message: str


@dataclass
class TypeAnnotationReport:
    """Full type annotation quality report for one Python file."""

    total_functions: int = 0
    annotated_functions: int = 0      # all params + return annotated
    partial_functions: int = 0        # some but not all annotated
    unannotated_functions: int = 0    # no annotations at all

    total_params: int = 0
    annotated_params: int = 0
    missing_return_count: int = 0
    any_usage_count: int = 0
    incomplete_generic_count: int = 0

    issues: List[AnnotationIssue] = field(default_factory=list)

    # Scores (0.0 - 1.0)
    coverage_score: float = 0.0       # what fraction of params/returns are annotated
    quality_score: float = 0.0        # penalise Any and incomplete generics
    overall_score: float = 0.0

    @property
    def param_coverage(self) -> float:
        if self.total_params == 0:
            return 1.0
        return self.annotated_params / self.total_params


# ---------------------------------------------------------------------------
# Incomplete generic detection
# ---------------------------------------------------------------------------

# Bare generic names that should carry a type parameter
_BARE_GENERICS = frozenset(
    {
        "List", "Dict", "Tuple", "Set", "FrozenSet",
        "Sequence", "Mapping", "Iterable", "Iterator",
        "Generator", "Optional",
    }
)


def _annotation_uses_bare_generic(annotation: ast.expr) -> bool:
    """Return True if the annotation is a bare generic (e.g. List not List[str])."""
    if isinstance(annotation, ast.Name):
        return annotation.id in _BARE_GENERICS
    if isinstance(annotation, ast.Attribute):
        return annotation.attr in _BARE_GENERICS
    return False


def _annotation_uses_any(annotation: ast.expr) -> bool:
    """Return True if the annotation references Any."""
    if isinstance(annotation, ast.Name):
        return annotation.id == "Any"
    if isinstance(annotation, ast.Attribute):
        return annotation.attr == "Any"
    if isinstance(annotation, ast.Subscript):
        return _annotation_uses_any(annotation.value) or _annotation_uses_any(annotation.slice)
    return False


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class TypeAnnotationScorer:
    """Score type annotation coverage and quality for Python source code."""

    def __init__(self, skip_self_cls: bool = True) -> None:
        """
        Parameters
        ----------
        skip_self_cls:
            If True, 'self' and 'cls' parameters are not required to have
            annotations (matching mypy behaviour).
        """
        self.skip_self_cls = skip_self_cls

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, source: str) -> TypeAnnotationReport:
        """Analyse the given Python source and return a report."""
        report = TypeAnnotationReport()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return report

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._analyse_function(node, report)

        self._compute_scores(report)
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _analyse_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        report: TypeAnnotationReport,
    ) -> None:
        report.total_functions += 1
        args = node.args
        all_args = list(args.args) + list(args.posonlyargs) + list(args.kwonlyargs)
        if args.vararg:
            all_args.append(args.vararg)
        if args.kwarg:
            all_args.append(args.kwarg)

        param_count = 0
        param_annotated = 0
        for arg in all_args:
            if self.skip_self_cls and arg.arg in ("self", "cls"):
                continue
            param_count += 1
            if arg.annotation is not None:
                param_annotated += 1
                if _annotation_uses_any(arg.annotation):
                    report.any_usage_count += 1
                    report.issues.append(
                        AnnotationIssue(
                            line_no=node.lineno,
                            name=f"{node.name}:{arg.arg}",
                            kind="any_usage",
                            message=f"Parameter '{arg.arg}' uses Any",
                        )
                    )
                if _annotation_uses_bare_generic(arg.annotation):
                    report.incomplete_generic_count += 1
                    report.issues.append(
                        AnnotationIssue(
                            line_no=node.lineno,
                            name=f"{node.name}:{arg.arg}",
                            kind="incomplete_generic",
                            message=f"Parameter '{arg.arg}' uses bare generic",
                        )
                    )
            else:
                report.issues.append(
                    AnnotationIssue(
                        line_no=node.lineno,
                        name=f"{node.name}:{arg.arg}",
                        kind="untyped_param",
                        message=f"Parameter '{arg.arg}' has no annotation",
                    )
                )

        report.total_params += param_count
        report.annotated_params += param_annotated

        # Return annotation
        has_return = node.returns is not None
        if not has_return:
            report.missing_return_count += 1
            report.issues.append(
                AnnotationIssue(
                    line_no=node.lineno,
                    name=node.name,
                    kind="missing_return",
                    message=f"Function '{node.name}' missing return annotation",
                )
            )
        else:
            if _annotation_uses_any(node.returns):
                report.any_usage_count += 1
                report.issues.append(
                    AnnotationIssue(
                        line_no=node.lineno,
                        name=node.name,
                        kind="any_usage",
                        message=f"Function '{node.name}' return uses Any",
                    )
                )

        # Classify function
        fully_annotated = param_annotated == param_count and has_return
        no_annotations = param_annotated == 0 and not has_return
        if fully_annotated:
            report.annotated_functions += 1
        elif no_annotations and param_count > 0:
            report.unannotated_functions += 1
        else:
            report.partial_functions += 1

    def _compute_scores(self, report: TypeAnnotationReport) -> None:
        if report.total_functions == 0:
            report.coverage_score = 1.0
            report.quality_score = 1.0
            report.overall_score = 1.0
            return

        # Coverage: fraction of params annotated + fraction of returns annotated
        total_annot_slots = report.total_params + report.total_functions
        filled_slots = report.annotated_params + (
            report.total_functions - report.missing_return_count
        )
        report.coverage_score = filled_slots / max(total_annot_slots, 1)

        # Quality: penalise Any usage and bare generics
        total_slots = max(total_annot_slots, 1)
        penalty = (report.any_usage_count + report.incomplete_generic_count) / total_slots
        report.quality_score = max(0.0, 1.0 - penalty * 2)

        report.overall_score = 0.6 * report.coverage_score + 0.4 * report.quality_score


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def score_annotations(source: str) -> TypeAnnotationReport:
    """Score type annotations with default settings."""
    return TypeAnnotationScorer().score(source)
