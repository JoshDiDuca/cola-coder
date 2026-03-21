"""Code Graph Builder: build call graphs and dependency graphs from Python source.

Provides:
  - Call graph: which functions call which other functions
  - Import dependency graph: which modules import which
  - Entry point detection (functions called from module-level)
  - Dead code detection (defined but never called)
  - Circular dependency detection in imports

For a TS dev: think of this as a lightweight bundler dependency graph
(like the one webpack produces) but for Python function calls and imports.
"""

from __future__ import annotations

import ast
from collections import defaultdict, deque
from dataclasses import dataclass, field


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CallGraph:
    """Call graph of a Python module."""

    # caller -> set of callees (functions they call)
    edges: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # all defined function names
    defined: set[str] = field(default_factory=set)
    # functions called at module level (entry points)
    entry_points: set[str] = field(default_factory=set)

    @property
    def dead_code(self) -> set[str]:
        """Functions defined but never called by any other function."""
        called_anywhere: set[str] = set()
        for callees in self.edges.values():
            called_anywhere.update(callees)
        called_anywhere.update(self.entry_points)
        return {fn for fn in self.defined if fn not in called_anywhere}

    @property
    def callers_of(self) -> dict[str, set[str]]:
        """Reverse edges: callee -> set of callers."""
        rev: dict[str, set[str]] = defaultdict(set)
        for caller, callees in self.edges.items():
            for callee in callees:
                rev[callee].add(caller)
        return rev

    def reachable_from(self, start: str) -> set[str]:
        """BFS from *start* — returns all reachable function names."""
        visited: set[str] = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in self.edges.get(node, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        return visited


@dataclass
class ImportGraph:
    """Import dependency graph."""

    # module/file identifier -> set of imported names
    imports: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    @property
    def all_imports(self) -> set[str]:
        result: set[str] = set()
        for vals in self.imports.values():
            result.update(vals)
        return result

    def circular_pairs(self) -> list[tuple[str, str]]:
        """Find (A, B) pairs where A imports B and B imports A."""
        pairs: list[tuple[str, str]] = []
        items = list(self.imports.items())
        for i, (mod_a, deps_a) in enumerate(items):
            for mod_b, deps_b in items[i + 1 :]:
                if mod_b in deps_a and mod_a in deps_b:
                    pairs.append((mod_a, mod_b))
        return pairs


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class CodeGraphBuilder:
    """Build call and import graphs from Python source strings."""

    def build_call_graph(self, source: str) -> CallGraph:
        """Parse *source* and return a :class:`CallGraph`."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return CallGraph()

        graph = CallGraph()

        # Collect all defined function names first
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                graph.defined.add(node.name)

        # Build edges: for each function, record calls it makes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                caller = node.name
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        callee = self._resolve_call_name(child)
                        if callee:
                            graph.edges[caller].add(callee)

        # Entry points: calls at module level (outside any function)
        for stmt in tree.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                callee = self._resolve_call_name(stmt.value)
                if callee:
                    graph.entry_points.add(callee)
            # Also detect direct calls in if __name__ == "__main__": blocks
            if isinstance(stmt, ast.If):
                test = stmt.test
                is_main_guard = (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                )
                if is_main_guard:
                    for body_stmt in stmt.body:
                        if isinstance(body_stmt, ast.Expr) and isinstance(
                            body_stmt.value, ast.Call
                        ):
                            callee = self._resolve_call_name(body_stmt.value)
                            if callee:
                                graph.entry_points.add(callee)

        return graph

    def build_import_graph(self, source: str, module_name: str = "<module>") -> ImportGraph:
        """Parse *source* and return an :class:`ImportGraph`."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return ImportGraph()

        graph = ImportGraph()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    graph.imports[module_name].add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    graph.imports[module_name].add(node.module)
        return graph

    @staticmethod
    def _resolve_call_name(call: ast.Call) -> str | None:
        """Extract a simple string name from a call node, or None."""
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return None
