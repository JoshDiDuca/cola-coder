"""Coverage Estimator: estimate test coverage of generated code statically.

Uses AST analysis to identify:
  - Branch points (if/elif/else, for/while loops, try/except, match/case)
  - Reachable vs unreachable paths
  - Functions / methods that have no corresponding test stubs detected
  - Approximate branch coverage percentage

No test runner is invoked — this is a static estimate only.

For a TS dev: similar to Istanbul's uncovered-branch report, but derived
purely from reading the source tree rather than instrumented execution.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BranchPoint:
    """A single branch point in the code."""

    node_type: str  # "if", "for", "while", "try", "with", "match"
    lineno: int
    branches: int  # estimated number of branches at this point


@dataclass
class CoverageReport:
    """Static coverage estimate."""

    total_functions: int
    tested_functions: int  # functions that appear to have test stubs
    branch_points: list[BranchPoint] = field(default_factory=list)
    unreachable_lines: list[int] = field(default_factory=list)  # after return/raise/break

    @property
    def total_branches(self) -> int:
        return sum(bp.branches for bp in self.branch_points)

    @property
    def estimated_coverage(self) -> float:
        """Rough 0.0–1.0 estimate.

        Combines function coverage and branch density heuristic.
        """
        if self.total_functions == 0:
            return 1.0
        func_cov = self.tested_functions / self.total_functions
        # Penalise for high branch count with no tests
        density_penalty = min(0.3, len(self.branch_points) * 0.02)
        if self.tested_functions == 0:
            return max(0.0, 0.2 - density_penalty)
        return max(0.0, min(1.0, func_cov * 0.7 + (1.0 - density_penalty) * 0.3))

    @property
    def untested_functions(self) -> int:
        return max(0, self.total_functions - self.tested_functions)

    @property
    def summary(self) -> str:
        return (
            f"Functions: {self.tested_functions}/{self.total_functions} tested, "
            f"Branches: {self.total_branches}, "
            f"Estimated coverage: {self.estimated_coverage:.0%}"
        )


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

_TEST_PREFIX_RE = re.compile(r"^test_?", re.IGNORECASE)


class CoverageEstimator:
    """Statically estimate test coverage for Python source code."""

    def estimate(self, source: str, test_source: str = "") -> CoverageReport:
        """Estimate coverage of *source*, optionally using *test_source* to detect tests.

        Parameters
        ----------
        source:
            The production code to analyse.
        test_source:
            Optional test file content — used to detect which functions are covered.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return CoverageReport(total_functions=0, tested_functions=0)

        functions = self._collect_function_names(tree)
        branch_points = self._collect_branch_points(tree)
        unreachable = self._detect_unreachable(tree)

        if test_source:
            try:
                test_tree = ast.parse(test_source)
                tested = self._match_tested_functions(functions, test_tree)
            except SyntaxError:
                tested = 0
        else:
            tested = 0

        return CoverageReport(
            total_functions=len(functions),
            tested_functions=tested,
            branch_points=branch_points,
            unreachable_lines=unreachable,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_function_names(tree: ast.Module) -> list[str]:
        return [
            n.name
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

    @staticmethod
    def _collect_branch_points(tree: ast.Module) -> list[BranchPoint]:
        points: list[BranchPoint] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # if + else = 2 branches (or 1 if no else)
                points.append(
                    BranchPoint("if", node.lineno, 2 if node.orelse else 1)
                )
            elif isinstance(node, (ast.For, ast.While)):
                points.append(BranchPoint("loop", node.lineno, 2))
            elif isinstance(node, ast.Try):
                branches = 1 + len(node.handlers)
                if node.orelse:
                    branches += 1
                points.append(BranchPoint("try", node.lineno, branches))
            elif isinstance(node, ast.Match):
                points.append(BranchPoint("match", node.lineno, len(node.cases)))
        return points

    @staticmethod
    def _detect_unreachable(tree: ast.Module) -> list[int]:
        """Find lines after return/raise/break/continue inside a block."""
        unreachable: list[int] = []
        for node in ast.walk(tree):
            stmts: list[ast.stmt] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stmts = list(node.body)
            elif isinstance(node, ast.If):
                stmts = list(node.body) + list(node.orelse)
            elif isinstance(node, (ast.For, ast.While)):
                stmts = list(node.body)
            elif isinstance(node, ast.Try):
                stmts = list(node.body)

            for i, stmt in enumerate(stmts[:-1]):
                if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                    if hasattr(stmts[i + 1], "lineno"):
                        unreachable.append(stmts[i + 1].lineno)
        return sorted(set(unreachable))

    @staticmethod
    def _match_tested_functions(functions: list[str], test_tree: ast.Module) -> int:
        """Count how many production functions are referenced in test_tree."""
        # Collect all names referenced anywhere in the test file
        referenced: set[str] = set()
        for node in ast.walk(test_tree):
            if isinstance(node, ast.Name):
                referenced.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # test_foo -> foo pattern
                name = node.name
                if _TEST_PREFIX_RE.match(name):
                    clean = _TEST_PREFIX_RE.sub("", name)
                    referenced.add(clean)

        return sum(1 for fn in functions if fn in referenced)
