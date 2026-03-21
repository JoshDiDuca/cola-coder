"""Tests for CodeGraphBuilder (features/code_graph.py)."""

from __future__ import annotations


from cola_coder.features.code_graph import (
    FEATURE_ENABLED,
    CodeGraphBuilder,
    ImportGraph,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Sample code
# ---------------------------------------------------------------------------

CALL_GRAPH_CODE = """\
def helper():
    return 42


def compute():
    return helper() + 1


def main():
    result = compute()
    print(result)


if __name__ == "__main__":
    main()
"""

DEAD_CODE_CODE = """\
def used():
    return 1

def unused():
    return 2

def entry():
    return used()
"""

IMPORT_CODE = """\
import os
import sys
from pathlib import Path
from collections import defaultdict
"""

CIRCULAR_SETUP = {
    "module_a": "import module_b\n",
    "module_b": "import module_a\n",
}

SYNTAX_ERROR = "def broken(:\n    pass"


class TestIsEnabled:
    def test_feature_enabled(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled_returns_true(self):
        assert is_enabled() is True


class TestCallGraph:
    def test_detects_defined_functions(self):
        builder = CodeGraphBuilder()
        graph = builder.build_call_graph(CALL_GRAPH_CODE)
        assert "helper" in graph.defined
        assert "compute" in graph.defined
        assert "main" in graph.defined

    def test_detects_edges(self):
        builder = CodeGraphBuilder()
        graph = builder.build_call_graph(CALL_GRAPH_CODE)
        # compute calls helper
        assert "helper" in graph.edges.get("compute", set())

    def test_detects_entry_points(self):
        builder = CodeGraphBuilder()
        graph = builder.build_call_graph(CALL_GRAPH_CODE)
        assert "main" in graph.entry_points

    def test_dead_code_detection(self):
        builder = CodeGraphBuilder()
        graph = builder.build_call_graph(DEAD_CODE_CODE)
        assert "unused" in graph.dead_code
        assert "used" not in graph.dead_code

    def test_reachable_from(self):
        builder = CodeGraphBuilder()
        graph = builder.build_call_graph(CALL_GRAPH_CODE)
        reachable = graph.reachable_from("main")
        assert "compute" in reachable
        assert "helper" in reachable

    def test_reachable_from_leaf(self):
        builder = CodeGraphBuilder()
        graph = builder.build_call_graph(CALL_GRAPH_CODE)
        # helper calls nothing else
        reachable = graph.reachable_from("helper")
        assert reachable == {"helper"}

    def test_callers_of(self):
        builder = CodeGraphBuilder()
        graph = builder.build_call_graph(CALL_GRAPH_CODE)
        callers = graph.callers_of
        # helper is called by compute
        assert "compute" in callers.get("helper", set())

    def test_syntax_error_returns_empty(self):
        builder = CodeGraphBuilder()
        graph = builder.build_call_graph(SYNTAX_ERROR)
        assert len(graph.defined) == 0


class TestImportGraph:
    def test_detects_plain_imports(self):
        builder = CodeGraphBuilder()
        graph = builder.build_import_graph(IMPORT_CODE)
        assert "os" in graph.all_imports
        assert "sys" in graph.all_imports

    def test_detects_from_imports(self):
        builder = CodeGraphBuilder()
        graph = builder.build_import_graph(IMPORT_CODE)
        assert "pathlib" in graph.all_imports
        assert "collections" in graph.all_imports

    def test_no_imports_empty_graph(self):
        builder = CodeGraphBuilder()
        graph = builder.build_import_graph("x = 1\n")
        assert len(graph.all_imports) == 0


class TestCircularDependency:
    def test_detects_circular_imports(self):
        builder = CodeGraphBuilder()
        graph = ImportGraph()
        for mod, src in CIRCULAR_SETUP.items():
            sub = builder.build_import_graph(src, module_name=mod)
            for k, v in sub.imports.items():
                graph.imports[k].update(v)
        pairs = graph.circular_pairs()
        assert len(pairs) > 0
        pair_set = {frozenset(p) for p in pairs}
        assert frozenset({"module_a", "module_b"}) in pair_set

    def test_no_circular_in_linear_imports(self):
        graph = ImportGraph()
        graph.imports["a"].add("b")
        graph.imports["b"].add("c")
        # a->b->c: no cycle
        pairs = graph.circular_pairs()
        assert len(pairs) == 0
