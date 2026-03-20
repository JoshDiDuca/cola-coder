"""Type Correctness Rate: heuristic type annotation quality checker.

Evaluates the quality of type annotations in generated code without
requiring an external type checker like mypy or pyright. Useful for
filtering and scoring generated code during training data prep and eval.

For a TS dev: think of this as a lightweight eslint rule that checks
whether type annotations exist, are consistent, and avoid common pitfalls —
without actually running the TypeScript compiler.
"""

import re
from dataclasses import dataclass, field

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TypeScore:
    """Aggregated type annotation quality score for a code snippet."""
    annotation_coverage: float  # 0-1 ratio of annotated functions
    consistency: float          # 0-1 score based on absence of inconsistencies
    overall: float              # weighted combination of above
    issues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Patterns (Python-focused, with basic TypeScript awareness)
# ---------------------------------------------------------------------------

# Matches Python function definitions: def name(...) or async def name(...)
_PY_FUNC_DEF = re.compile(
    r'^[ \t]*(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?:',
    re.MULTILINE,
)

# Matches a parameter that has a type annotation:  param: SomeType
_PY_PARAM_ANNOTATED = re.compile(r'\w+\s*:\s*\S')

# TypeScript/JavaScript function patterns (arrow functions + regular)
_TS_FUNC_DEF = re.compile(
    r'(?:function\s+\w+\s*\([^)]*\)\s*(?::\s*\S+)?|'
    r'(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*(?::\s*\S+)?\s*=>)',
    re.MULTILINE,
)

# Common Python type mistakes / anti-patterns
_COMMON_ERRORS: list[tuple[str, re.Pattern, str]] = [
    (
        "use `list[T]` instead of `List[T]` (Python 3.9+)",
        re.compile(r':\s*List\[', re.MULTILINE),
        "legacy typing.List used; prefer list[T]",
    ),
    (
        "use `dict[K, V]` instead of `Dict[K, V]` (Python 3.9+)",
        re.compile(r':\s*Dict\[', re.MULTILINE),
        "legacy typing.Dict used; prefer dict[K, V]",
    ),
    (
        "use `tuple[...]` instead of `Tuple[...]` (Python 3.9+)",
        re.compile(r':\s*Tuple\[', re.MULTILINE),
        "legacy typing.Tuple used; prefer tuple[...]",
    ),
    (
        "use `T | None` instead of `Optional[T]` (Python 3.10+)",
        re.compile(r':\s*Optional\[', re.MULTILINE),
        "legacy Optional[T] used; prefer T | None",
    ),
    (
        "use `T | U` instead of `Union[T, U]` (Python 3.10+)",
        re.compile(r':\s*Union\[', re.MULTILINE),
        "legacy Union[T, U] used; prefer T | U",
    ),
    (
        "avoid bare `-> None` on __init__ omission",
        re.compile(r'def\s+__init__\s*\([^)]*\)\s*:', re.MULTILINE),
        "__init__ missing return type annotation (-> None)",
    ),
    (
        "type: ignore comment suppresses type checking",
        re.compile(r'#\s*type:\s*ignore', re.MULTILINE),
        "# type: ignore suppresses type errors",
    ),
    (
        "Any annotation weakens type safety",
        re.compile(r':\s*Any\b', re.MULTILINE),
        "Any annotation used; consider a more specific type",
    ),
    (
        "object annotation is overly broad",
        re.compile(r'\)\s*->\s*object\b', re.MULTILINE),
        "return type annotated as `object`; consider a more specific type",
    ),
]

# Detects variable re-assignment with incompatible-looking literal types,
# e.g.  x: int = 0  then later  x = "hello"
_VAR_ANNOTATION = re.compile(
    r'^[ \t]*(\w+)\s*:\s*(\w+)\s*=\s*(.+)$', re.MULTILINE
)


# ---------------------------------------------------------------------------
# TypeChecker
# ---------------------------------------------------------------------------

class TypeChecker:
    """Heuristic type annotation checker — no external tools required."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_annotations_present(
        self, code: str, language: str = 'python'
    ) -> float:
        """Return 0-1 ratio of functions that have type annotations.

        A function is considered annotated if it has a return type annotation
        OR at least one annotated parameter (excluding `self` / `cls`).
        """
        if language in ('typescript', 'javascript', 'ts', 'js'):
            return self._ts_annotation_coverage(code)
        return self._py_annotation_coverage(code)

    def check_type_consistency(self, code: str) -> list[str]:
        """Find type inconsistencies in Python code.

        Looks for variables declared with a type annotation whose value
        literal is obviously incompatible (e.g. `x: int = "hello"`).
        Returns a list of human-readable issue descriptions.
        """
        issues: list[str] = []
        for m in _VAR_ANNOTATION.finditer(code):
            var, declared_type, value = m.group(1), m.group(2), m.group(3).strip()
            issue = self._check_literal_type_mismatch(var, declared_type, value)
            if issue:
                issues.append(issue)
        return issues

    def check_common_errors(self, code: str) -> list[str]:
        """Find common type annotation mistakes in Python code."""
        issues: list[str] = []
        for _name, pattern, message in _COMMON_ERRORS:
            if pattern.search(code):
                issues.append(message)
        return issues

    def score(self, code: str, language: str = 'python') -> TypeScore:
        """Compute an overall TypeScore for the given code snippet."""
        coverage = self.check_annotations_present(code, language)
        consistency_issues = self.check_type_consistency(code)
        common_error_issues = self.check_common_errors(code)

        all_issues = consistency_issues + common_error_issues

        # Consistency score: start at 1.0, deduct per issue (floor at 0)
        consistency = max(0.0, 1.0 - 0.1 * len(all_issues))

        # Overall: 70% coverage, 30% consistency
        overall = 0.7 * coverage + 0.3 * consistency

        return TypeScore(
            annotation_coverage=coverage,
            consistency=consistency,
            overall=overall,
            issues=all_issues,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _py_annotation_coverage(self, code: str) -> float:
        """Compute annotation coverage for Python code."""
        matches = list(_PY_FUNC_DEF.finditer(code))
        if not matches:
            return 0.0

        annotated = 0
        for m in matches:
            params_str = m.group(2) or ''
            return_type = m.group(3)

            has_return = bool(return_type and return_type.strip())
            has_param_annotation = self._has_annotated_param(params_str)

            if has_return or has_param_annotation:
                annotated += 1

        return annotated / len(matches)

    def _has_annotated_param(self, params_str: str) -> bool:
        """Return True if any parameter (excluding self/cls) has an annotation."""
        for param in params_str.split(','):
            param = param.strip()
            if not param or param in ('self', 'cls', '*', '/'):
                continue
            # Strip default value before checking
            bare = param.split('=')[0].strip()
            if bare in ('self', 'cls', '*args', '**kwargs'):
                continue
            if ':' in bare:
                return True
        return False

    def _ts_annotation_coverage(self, code: str) -> float:
        """Rough annotation coverage for TypeScript/JavaScript."""
        matches = list(_TS_FUNC_DEF.finditer(code))
        if not matches:
            return 0.0

        annotated = 0
        for m in matches:
            # Presence of `: Type` in the match text is a reasonable proxy
            text = m.group(0)
            if re.search(r':\s*\w', text):
                annotated += 1

        return annotated / len(matches)

    @staticmethod
    def _check_literal_type_mismatch(
        var: str, declared_type: str, value: str
    ) -> str | None:
        """Return an issue string if the literal value obviously mismatches declared_type."""
        # Map declared type -> set of obviously-wrong literal patterns
        mismatches: list[tuple[str, re.Pattern, str]] = [
            ('int',   re.compile(r'^".*"$|^\'.*\'$'),         'str literal'),
            ('int',   re.compile(r'^\['),                      'list literal'),
            ('str',   re.compile(r'^\d+$'),                    'int literal'),
            ('str',   re.compile(r'^\['),                      'list literal'),
            ('str',   re.compile(r'^\{'),                      'dict/set literal'),
            ('bool',  re.compile(r'^".*"$|^\'.*\'$'),          'str literal'),
            ('bool',  re.compile(r'^\d+[^.]'),                 'non-bool int literal'),
            ('float', re.compile(r'^".*"$|^\'.*\'$'),          'str literal'),
            ('list',  re.compile(r'^".*"$|^\'.*\'$'),          'str literal'),
            ('list',  re.compile(r'^\{[^}]*:[^}]*\}'),         'dict literal'),
            ('dict',  re.compile(r'^\['),                      'list literal'),
        ]

        dt = declared_type.lower()
        for type_name, wrong_pattern, wrong_desc in mismatches:
            if dt == type_name and wrong_pattern.match(value):
                return (
                    f"Variable `{var}` declared as `{declared_type}` "
                    f"but assigned a {wrong_desc}"
                )
        return None


# ---------------------------------------------------------------------------
# TypeCorrectnessTracker
# ---------------------------------------------------------------------------

class TypeCorrectnessTracker:
    """Records type scores across multiple code snippets and reports averages."""

    def __init__(self) -> None:
        self._checker = TypeChecker()
        self._scores: list[TypeScore] = []

    def record(self, code: str, language: str = 'python') -> None:
        """Score the given code and add the result to the tracker."""
        self._scores.append(self._checker.score(code, language))

    def average_score(self) -> float:
        """Return the mean overall score across all recorded snippets."""
        if not self._scores:
            return 0.0
        return sum(s.overall for s in self._scores) / len(self._scores)

    def summary(self) -> dict:
        """Return a summary dict with aggregated statistics."""
        if not self._scores:
            return {
                'count': 0,
                'average_overall': 0.0,
                'average_annotation_coverage': 0.0,
                'average_consistency': 0.0,
                'total_issues': 0,
            }

        n = len(self._scores)
        avg_overall = sum(s.overall for s in self._scores) / n
        avg_coverage = sum(s.annotation_coverage for s in self._scores) / n
        avg_consistency = sum(s.consistency for s in self._scores) / n
        total_issues = sum(len(s.issues) for s in self._scores)

        return {
            'count': n,
            'average_overall': round(avg_overall, 4),
            'average_annotation_coverage': round(avg_coverage, 4),
            'average_consistency': round(avg_consistency, 4),
            'total_issues': total_issues,
        }
