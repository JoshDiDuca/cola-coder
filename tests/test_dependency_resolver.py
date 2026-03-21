"""Tests for dependency_resolver.py."""

from __future__ import annotations


from cola_coder.features.dependency_resolver import (
    FEATURE_ENABLED,
    DependencyGraph,
    DependencyResolver,
    extract_imports,
    is_enabled,
    module_name_from_path,
    topological_sort,
)


class TestIsEnabled:
    def test_constant(self):
        assert FEATURE_ENABLED is True

    def test_function(self):
        assert is_enabled() is True


class TestExtractImports:
    def test_simple_import(self):
        code = "import os\nimport sys\n"
        edges = extract_imports(code, "mymodule")
        targets = {e.target for e in edges}
        assert "os" in targets
        assert "sys" in targets

    def test_from_import(self):
        code = "from pathlib import Path\n"
        edges = extract_imports(code, "mymodule")
        targets = {e.target for e in edges}
        assert "pathlib" in targets

    def test_relative_import(self):
        code = "from . import sibling\n"
        edges = extract_imports(code, "pkg.module")
        relative = [e for e in edges if e.is_relative]
        assert len(relative) >= 1

    def test_invalid_syntax_returns_empty(self):
        edges = extract_imports("def broken(:\n    pass\n", "mod")
        assert edges == []

    def test_source_is_module_name(self):
        code = "import os\n"
        edges = extract_imports(code, "mymod")
        assert all(e.source == "mymod" for e in edges)


class TestModuleNameFromPath:
    def test_simple_path(self):
        name = module_name_from_path("src/cola_coder/features/foo.py")
        assert name.endswith("foo")

    def test_init_stripped(self):
        name = module_name_from_path("pkg/__init__.py")
        assert "__init__" not in name


class TestDependencyGraph:
    def test_add_edge_creates_nodes(self):
        graph = DependencyGraph()
        graph.add_edge("a", "b")
        assert "a" in graph.nodes
        assert "b" in graph.nodes

    def test_adjacency_correct(self):
        graph = DependencyGraph()
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        assert graph.dependencies_of("a") == {"b", "c"}

    def test_reverse_adjacency(self):
        graph = DependencyGraph()
        graph.add_edge("a", "b")
        assert "a" in graph.dependents_of("b")

    def test_no_duplicate_edges(self):
        graph = DependencyGraph()
        graph.add_edge("a", "b")
        graph.add_edge("a", "b")
        assert len(graph.edges) == 1


class TestTopologicalSort:
    def test_linear_chain(self):
        graph = DependencyGraph()
        graph.add_edge("b", "a")  # b depends on a
        graph.add_edge("c", "b")  # c depends on b
        result = topological_sort(graph)
        assert result.order.index("a") < result.order.index("b")
        assert result.order.index("b") < result.order.index("c")

    def test_no_cycle_flag(self):
        graph = DependencyGraph()
        graph.add_edge("b", "a")
        result = topological_sort(graph)
        assert not result.has_cycles

    def test_cycle_detected(self):
        graph = DependencyGraph()
        graph.add_edge("a", "b")
        graph.add_edge("b", "a")  # cycle
        result = topological_sort(graph)
        assert result.has_cycles

    def test_independent_nodes_all_present(self):
        graph = DependencyGraph()
        graph.add_node("x")
        graph.add_node("y")
        graph.add_node("z")
        result = topological_sort(graph)
        assert set(result.order) == {"x", "y", "z"}


class TestDependencyResolver:
    def test_simple_resolution(self):
        resolver = DependencyResolver()
        resolver.add_module("pkg.utils", "import os\n")
        resolver.add_module("pkg.main", "from pkg import utils\n")
        order = resolver.suggest_training_order()
        assert "pkg.utils" in order
        assert "pkg.main" in order

    def test_stdlib_excluded_from_graph(self):
        resolver = DependencyResolver()
        resolver.add_module("mymod", "import os\nimport sys\n")
        # os and sys are stdlib — should not be in the graph nodes
        assert "os" not in resolver.graph.nodes
        assert "sys" not in resolver.graph.nodes

    def test_dependencies_of_direct(self):
        resolver = DependencyResolver()
        resolver.add_module("pkg.b", "import pkg.a\n")
        resolver.add_module("pkg.a", "")
        deps = resolver.dependencies_of("pkg.b")
        assert "pkg.a" in deps

    def test_add_modules_dict(self):
        resolver = DependencyResolver()
        resolver.add_modules({
            "mod_a": "",
            "mod_b": "import mod_a\n",
        })
        assert "mod_a" in resolver.graph.nodes
        assert "mod_b" in resolver.graph.nodes

    def test_summary_str(self):
        resolver = DependencyResolver()
        resolver.add_module("a", "")
        result = resolver.resolve()
        s = result.summary()
        assert "Resolved" in s
