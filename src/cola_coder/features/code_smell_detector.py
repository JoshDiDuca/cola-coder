"""Code Smell Detector: identify common code quality issues in Python source.

Detects:
  - Long methods (too many statements)
  - Deep nesting (excessive block depth)
  - God classes (too many methods/attributes)
  - Feature envy (method calls many external objects)
  - Data clumps (repeated parameter groups)

Each smell receives a severity score (0.0–1.0) and a fix suggestion.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import NamedTuple


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Thresholds (tunable)
# ---------------------------------------------------------------------------

DEFAULT_MAX_METHOD_STATEMENTS = 20
DEFAULT_MAX_NESTING_DEPTH = 4
DEFAULT_MAX_CLASS_METHODS = 15
DEFAULT_MAX_CLASS_ATTRIBUTES = 10
DEFAULT_FEATURE_ENVY_RATIO = 0.6  # fraction of attribute accesses on other objects
DEFAULT_DATA_CLUMP_MIN_SIZE = 3   # min params to be a clump
DEFAULT_DATA_CLUMP_MIN_REPEAT = 2  # must appear in at least N functions


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class SmellInstance(NamedTuple):
    """A single detected smell."""

    smell_type: str   # e.g. "long_method"
    location: str     # e.g. "MyClass.do_thing" or "module"
    severity: float   # 0.0 (mild) – 1.0 (critical)
    details: str
    suggestion: str


@dataclass
class SmellReport:
    """Aggregated smell detection results."""

    smells: list[SmellInstance] = field(default_factory=list)
    total_severity: float = 0.0
    smell_counts: dict[str, int] = field(default_factory=dict)

    def add(self, smell: SmellInstance) -> None:
        self.smells.append(smell)
        self.total_severity += smell.severity
        self.smell_counts[smell.smell_type] = (
            self.smell_counts.get(smell.smell_type, 0) + 1
        )

    @property
    def average_severity(self) -> float:
        if not self.smells:
            return 0.0
        return self.total_severity / len(self.smells)

    def has_smell(self, smell_type: str) -> bool:
        return smell_type in self.smell_counts

    def summary(self) -> str:
        if not self.smells:
            return "No smells detected."
        lines = [f"Found {len(self.smells)} smell(s), avg severity {self.average_severity:.2f}:"]
        for stype, cnt in sorted(self.smell_counts.items()):
            lines.append(f"  {stype}: {cnt}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class CodeSmellDetector:
    """Detect code smells in Python source.

    Parameters
    ----------
    max_method_statements:
        Threshold for long-method detection.
    max_nesting_depth:
        Maximum allowed block nesting depth.
    max_class_methods:
        Maximum methods before flagging god class.
    max_class_attributes:
        Maximum instance attributes before flagging god class.
    feature_envy_ratio:
        Fraction of foreign-object attribute accesses that triggers envy.
    data_clump_min_size:
        Minimum number of shared parameters to form a clump.
    data_clump_min_repeat:
        Number of functions that must share the parameter group.
    """

    def __init__(
        self,
        max_method_statements: int = DEFAULT_MAX_METHOD_STATEMENTS,
        max_nesting_depth: int = DEFAULT_MAX_NESTING_DEPTH,
        max_class_methods: int = DEFAULT_MAX_CLASS_METHODS,
        max_class_attributes: int = DEFAULT_MAX_CLASS_ATTRIBUTES,
        feature_envy_ratio: float = DEFAULT_FEATURE_ENVY_RATIO,
        data_clump_min_size: int = DEFAULT_DATA_CLUMP_MIN_SIZE,
        data_clump_min_repeat: int = DEFAULT_DATA_CLUMP_MIN_REPEAT,
    ) -> None:
        self.max_method_statements = max_method_statements
        self.max_nesting_depth = max_nesting_depth
        self.max_class_methods = max_class_methods
        self.max_class_attributes = max_class_attributes
        self.feature_envy_ratio = feature_envy_ratio
        self.data_clump_min_size = data_clump_min_size
        self.data_clump_min_repeat = data_clump_min_repeat

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, code: str) -> SmellReport:
        """Detect smells in *code* and return a SmellReport."""
        report = SmellReport()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return report

        self._check_long_methods(tree, report)
        self._check_deep_nesting(tree, report)
        self._check_god_classes(tree, report)
        self._check_feature_envy(tree, report)
        self._check_data_clumps(tree, report)
        return report

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def _check_long_methods(self, tree: ast.AST, report: SmellReport) -> None:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            stmt_count = sum(1 for _ in ast.walk(node) if isinstance(_, ast.stmt))
            if stmt_count > self.max_method_statements:
                excess = stmt_count - self.max_method_statements
                severity = min(1.0, excess / self.max_method_statements)
                report.add(SmellInstance(
                    smell_type="long_method",
                    location=node.name,
                    severity=round(severity, 3),
                    details=f"{stmt_count} statements (limit {self.max_method_statements})",
                    suggestion=(
                        "Extract sub-tasks into helper functions. "
                        "Aim for single-responsibility methods."
                    ),
                ))

    def _check_deep_nesting(self, tree: ast.AST, report: SmellReport) -> None:
        """Walk the tree tracking nesting depth of block statements."""
        _BLOCK_TYPES = (
            ast.If, ast.For, ast.While, ast.With,
            ast.Try, ast.ExceptHandler,
        )

        def _walk(node: ast.AST, depth: int, func_name: str) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                depth = 0  # reset per-function

            if isinstance(node, _BLOCK_TYPES):
                depth += 1
                if depth > self.max_nesting_depth:
                    excess = depth - self.max_nesting_depth
                    severity = min(1.0, excess / self.max_nesting_depth)
                    report.add(SmellInstance(
                        smell_type="deep_nesting",
                        location=func_name or "module",
                        severity=round(severity, 3),
                        details=f"Nesting depth {depth} (limit {self.max_nesting_depth})",
                        suggestion=(
                            "Use early returns/guard clauses or extract "
                            "nested blocks into separate functions."
                        ),
                    ))
                    # Don't double-report deeper nesting in same branch
                    return

            for child in ast.iter_child_nodes(node):
                _walk(child, depth, func_name)

        _walk(tree, 0, "module")

    def _check_god_classes(self, tree: ast.AST, report: SmellReport) -> None:
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            methods = [
                n for n in ast.walk(node)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            # Count instance attributes set via self.x = ...
            attrs: set[str] = set()
            for method in methods:
                for n in ast.walk(method):
                    if (
                        isinstance(n, ast.Assign)
                        and isinstance(n.targets[0] if n.targets else None, ast.Attribute)
                        and isinstance(n.targets[0].value, ast.Name)  # type: ignore[union-attr]
                        and n.targets[0].value.id == "self"  # type: ignore[union-attr]
                    ):
                        attrs.add(n.targets[0].attr)  # type: ignore[union-attr]

            method_violation = len(methods) > self.max_class_methods
            attr_violation = len(attrs) > self.max_class_attributes

            if method_violation or attr_violation:
                severity = max(
                    min(1.0, len(methods) / (self.max_class_methods * 2)) if method_violation else 0.0,
                    min(1.0, len(attrs) / (self.max_class_attributes * 2)) if attr_violation else 0.0,
                )
                parts = []
                if method_violation:
                    parts.append(f"{len(methods)} methods")
                if attr_violation:
                    parts.append(f"{len(attrs)} attributes")
                report.add(SmellInstance(
                    smell_type="god_class",
                    location=node.name,
                    severity=round(severity, 3),
                    details=", ".join(parts),
                    suggestion=(
                        "Break this class into smaller, focused classes using "
                        "composition or inheritance."
                    ),
                ))

    def _check_feature_envy(self, tree: ast.AST, report: SmellReport) -> None:
        """Detect methods that access more attributes of other objects than self."""
        for cls_node in ast.walk(tree):
            if not isinstance(cls_node, ast.ClassDef):
                continue
            for method in ast.walk(cls_node):
                if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                self_accesses = 0
                other_accesses = 0
                for n in ast.walk(method):
                    if not isinstance(n, ast.Attribute):
                        continue
                    if isinstance(n.value, ast.Name) and n.value.id == "self":
                        self_accesses += 1
                    elif isinstance(n.value, ast.Name):
                        other_accesses += 1

                total = self_accesses + other_accesses
                if total < 4:
                    continue  # not enough data
                ratio = other_accesses / total
                if ratio >= self.feature_envy_ratio:
                    severity = min(1.0, ratio)
                    report.add(SmellInstance(
                        smell_type="feature_envy",
                        location=f"{cls_node.name}.{method.name}",
                        severity=round(severity, 3),
                        details=(
                            f"{other_accesses} foreign accesses vs "
                            f"{self_accesses} self accesses ({ratio:.0%} foreign)"
                        ),
                        suggestion=(
                            "Move this method to the class whose data it uses most, "
                            "or delegate to that class."
                        ),
                    ))

    def _check_data_clumps(self, tree: ast.AST, report: SmellReport) -> None:
        """Find parameter groups that repeatedly appear together."""
        func_params: list[frozenset[str]] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            params = {a.arg for a in args.args + args.posonlyargs + args.kwonlyargs}
            params.discard("self")
            params.discard("cls")
            if len(params) >= self.data_clump_min_size:
                func_params.append(frozenset(params))

        if len(func_params) < self.data_clump_min_repeat:
            return

        # Find subsets of size >= min_size that appear in >= min_repeat functions
        seen_clumps: set[frozenset[str]] = set()
        for i, params_i in enumerate(func_params):
            for j in range(i + 1, len(func_params)):
                params_j = func_params[j]
                common = params_i & params_j
                if len(common) < self.data_clump_min_size:
                    continue
                clump = frozenset(common)
                if clump in seen_clumps:
                    continue
                # Count how many functions have ALL these params
                count = sum(1 for ps in func_params if clump.issubset(ps))
                if count >= self.data_clump_min_repeat:
                    seen_clumps.add(clump)
                    severity = min(1.0, count / (self.data_clump_min_repeat * 3))
                    report.add(SmellInstance(
                        smell_type="data_clump",
                        location="module",
                        severity=round(severity, 3),
                        details=(
                            f"Parameters {sorted(clump)} appear together "
                            f"in {count} functions"
                        ),
                        suggestion=(
                            "Group these parameters into a dataclass or named tuple "
                            "to reduce coupling."
                        ),
                    ))
