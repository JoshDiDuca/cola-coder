"""Code Dependency Resolver.

Resolve and order Python code files/modules by their import dependencies.
Ensures training data includes dependencies before dependents via topological sort.

Handles:
  - Absolute imports (import os, from pathlib import Path)
  - Relative imports (from . import sibling)
  - Circular dependency detection
  - Multi-file dependency graphs
"""

from __future__ import annotations

import ast
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ImportEdge(NamedTuple):
    """A directed dependency edge: source depends on target."""

    source: str   # module name that contains the import
    target: str   # module name that is imported
    is_relative: bool


@dataclass
class DependencyGraph:
    """Directed graph of module dependencies."""

    nodes: set[str] = field(default_factory=set)
    edges: list[ImportEdge] = field(default_factory=list)
    # adjacency list: module -> set of modules it imports
    adjacency: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # reverse: module -> set of modules that import it
    reverse_adjacency: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_node(self, name: str) -> None:
        self.nodes.add(name)

    def add_edge(self, source: str, target: str, is_relative: bool = False) -> None:
        self.nodes.add(source)
        self.nodes.add(target)
        edge = ImportEdge(source=source, target=target, is_relative=is_relative)
        if edge not in self.edges:
            self.edges.append(edge)
        self.adjacency[source].add(target)
        self.reverse_adjacency[target].add(source)

    def dependencies_of(self, module: str) -> set[str]:
        """Return all modules that *module* directly depends on."""
        return set(self.adjacency.get(module, set()))

    def dependents_of(self, module: str) -> set[str]:
        """Return all modules that directly depend on *module*."""
        return set(self.reverse_adjacency.get(module, set()))


@dataclass
class ResolvedOrder:
    """Result of topological sort."""

    order: list[str]             # modules in dependency-first order
    has_cycles: bool = False
    cycles: list[list[str]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)  # external/unknown modules

    def summary(self) -> str:
        lines = [f"Resolved {len(self.order)} modules"]
        if self.has_cycles:
            lines.append(f"WARNING: {len(self.cycles)} cycle(s) detected")
            for cycle in self.cycles[:3]:
                lines.append(f"  Cycle: {' -> '.join(cycle)}")
        if self.unresolved:
            lines.append(f"External/unresolved: {self.unresolved[:5]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------


def extract_imports(code: str, module_name: str = "<unknown>") -> list[ImportEdge]:
    """Parse *code* and extract all import edges."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    edges: list[ImportEdge] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.append(ImportEdge(
                    source=module_name,
                    target=alias.name,
                    is_relative=False,
                ))

        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # Relative import
                dots = "." * node.level
                target = f"{dots}{node.module or ''}"
                edges.append(ImportEdge(
                    source=module_name,
                    target=target,
                    is_relative=True,
                ))
            elif node.module:
                edges.append(ImportEdge(
                    source=module_name,
                    target=node.module,
                    is_relative=False,
                ))

    return edges


def module_name_from_path(path: str, root: str = "") -> str:
    """Convert a file path to a Python module name."""
    p = Path(path)
    if root:
        try:
            p = p.relative_to(root)
        except ValueError:
            pass
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


# ---------------------------------------------------------------------------
# Topological sort (Kahn's algorithm)
# ---------------------------------------------------------------------------


def topological_sort(graph: DependencyGraph, known_modules: set[str] | None = None) -> ResolvedOrder:
    """Kahn's topological sort over known modules.

    Modules with no known dependencies are placed first.
    External/unknown dependencies are ignored for ordering purposes.

    Parameters
    ----------
    graph:
        The dependency graph.
    known_modules:
        Subset of nodes to sort.  If None, all nodes are used.
    """
    nodes = known_modules if known_modules is not None else graph.nodes
    # Only consider edges between known modules.
    # Edge direction: source imports/depends on target.
    # For dependency-first ordering we build the REVERSE graph (target → source)
    # and do Kahn's on it: nodes with no *dependents* (leaves in original) come first.
    external: set[str] = set()
    # dep_count[n] = number of known dependencies n has (= out-degree in original)
    dep_count: dict[str, int] = {n: 0 for n in nodes}
    # reverse_adj[n] = set of nodes that depend on n (= nodes whose dep_count we decrement)
    reverse_adj: dict[str, set[str]] = {n: set() for n in nodes}
    # forward_adj kept only for cycle detection
    forward_adj: dict[str, set[str]] = {n: set() for n in nodes}

    for edge in graph.edges:
        if edge.source not in nodes:
            continue
        if edge.target not in nodes:
            external.add(edge.target)
            continue
        if edge.target not in forward_adj[edge.source]:
            forward_adj[edge.source].add(edge.target)
            reverse_adj[edge.target].add(edge.source)
            dep_count[edge.source] += 1

    # Queue: nodes with zero dependencies come first (pure leaves / no imports)
    queue: deque[str] = deque(sorted(n for n, cnt in dep_count.items() if cnt == 0))
    order: list[str] = []
    visited: set[str] = set()

    while queue:
        node = queue.popleft()
        order.append(node)
        visited.add(node)
        # Decrement dep_count for everyone that depends on this node
        for dependent in sorted(reverse_adj.get(node, set())):
            dep_count[dependent] -= 1
            if dep_count[dependent] == 0:
                queue.append(dependent)

    unprocessed = [n for n in nodes if n not in visited]
    has_cycles = bool(unprocessed)
    cycles: list[list[str]] = []
    if has_cycles:
        cycles = _find_cycles(forward_adj, unprocessed)

    return ResolvedOrder(
        order=order,
        has_cycles=has_cycles,
        cycles=cycles,
        unresolved=sorted(external),
    )


def _find_cycles(adj: dict[str, set[str]], candidates: list[str]) -> list[list[str]]:
    """Find simple cycles among candidate nodes using DFS."""
    cycles: list[list[str]] = []
    visited: set[str] = set()

    def dfs(node: str, path: list[str], path_set: set[str]) -> None:
        visited.add(node)
        path.append(node)
        path_set.add(node)
        for neighbour in adj.get(node, set()):
            if neighbour in path_set:
                # Found a cycle
                cycle_start = path.index(neighbour)
                cycles.append(path[cycle_start:] + [neighbour])
            elif neighbour not in visited and neighbour in candidates:
                dfs(neighbour, path, path_set)
        path.pop()
        path_set.discard(node)

    for node in candidates:
        if node not in visited:
            dfs(node, [], set())
            if len(cycles) >= 10:  # limit cycle reporting
                break

    return cycles


# ---------------------------------------------------------------------------
# High-level resolver
# ---------------------------------------------------------------------------


class DependencyResolver:
    """Resolve import dependencies for a collection of Python modules.

    Parameters
    ----------
    stdlib_modules:
        Set of standard library module names to treat as external.
        Defaults to a broad built-in set.
    """

    _DEFAULT_STDLIB = frozenset(
        "abc ast asyncio builtins collections contextlib copy dataclasses "
        "datetime enum functools hashlib importlib inspect io itertools json "
        "logging math operator os pathlib pickle platform pprint queue random "
        "re shutil signal socket string struct subprocess sys tempfile textwrap "
        "threading time traceback types typing unicodedata unittest urllib uuid "
        "warnings weakref".split()
    )

    def __init__(self, stdlib_modules: frozenset[str] | None = None) -> None:
        self.stdlib_modules = stdlib_modules or self._DEFAULT_STDLIB
        self.graph = DependencyGraph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_module(self, module_name: str, code: str) -> None:
        """Register a module with its source code."""
        self.graph.add_node(module_name)
        for edge in extract_imports(code, module_name=module_name):
            root = edge.target.lstrip(".").split(".")[0]
            if root not in self.stdlib_modules:
                self.graph.add_edge(edge.source, edge.target, edge.is_relative)

    def add_modules(self, modules: dict[str, str]) -> None:
        """Register multiple modules at once. Keys are module names, values are code."""
        for name, code in modules.items():
            self.add_module(name, code)

    def resolve(self) -> ResolvedOrder:
        """Compute dependency-first ordering for all registered modules."""
        return topological_sort(self.graph, known_modules=self.graph.nodes)

    def dependencies_of(self, module_name: str, transitive: bool = False) -> set[str]:
        """Return direct (or all transitive) dependencies of *module_name*."""
        if not transitive:
            return self.graph.dependencies_of(module_name)
        visited: set[str] = set()
        stack = list(self.graph.dependencies_of(module_name))
        while stack:
            dep = stack.pop()
            if dep in visited:
                continue
            visited.add(dep)
            stack.extend(self.graph.dependencies_of(dep))
        return visited

    def suggest_training_order(self) -> list[str]:
        """Return only the known modules in dependency-first order."""
        result = self.resolve()
        return [m for m in result.order if m in self.graph.nodes]
