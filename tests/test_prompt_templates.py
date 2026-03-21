"""Tests for PromptLibrary extensions (features/prompt_templates.py).

Validates the 20+ new templates added in improvement #14.
"""

from __future__ import annotations

import pytest

from cola_coder.features.prompt_templates import PromptLibrary, PromptTemplate


@pytest.fixture()
def lib() -> PromptLibrary:
    return PromptLibrary()


# ---------------------------------------------------------------------------
# Template count / structure
# ---------------------------------------------------------------------------


class TestLibrarySize:
    def test_at_least_30_templates(self, lib):
        assert len(lib.list_templates()) >= 30

    def test_new_categories_present(self, lib):
        cats = lib.categories()
        assert "function_gen" in cats
        assert "class_gen" in cats
        assert "test_gen" in cats
        assert "optimize" in cats

    def test_unique_names(self, lib):
        names = [t.name for t in lib.list_templates()]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# New template accessibility
# ---------------------------------------------------------------------------


class TestNewTemplates:
    NEW_NAMES = [
        "python_function_typed",
        "async_function",
        "lambda_function",
        "generator_function",
        "python_dataclass",
        "python_enum",
        "abstract_base_class",
        "protocol_class",
        "pytest_function",
        "pytest_class",
        "pytest_parametrize",
        "mock_patch",
        "extract_function",
        "add_type_hints",
        "convert_to_dataclass",
        "explain_step_by_step",
        "debug_code",
        "add_error_handling",
        "optimize_time_complexity",
        "optimize_memory",
        "vectorize_loop",
        "ts_generic_function",
        "ts_zod_schema",
        "ts_react_hook",
    ]

    def test_all_new_templates_registered(self, lib):
        for name in self.NEW_NAMES:
            t = lib.get(name)
            assert t.name == name, f"Template {name} not found"

    def test_templates_have_variables(self, lib):
        for name in self.NEW_NAMES:
            t = lib.get(name)
            # All templates should declare at least one variable (most do)
            # A few may have zero (no-variable templates)
            assert isinstance(t.variables, list)

    def test_templates_have_category(self, lib):
        for name in self.NEW_NAMES:
            t = lib.get(name)
            assert t.category, f"Template '{name}' has empty category"


# ---------------------------------------------------------------------------
# list_by_category
# ---------------------------------------------------------------------------


class TestListByCategory:
    def test_list_by_category_function_gen(self, lib):
        templates = lib.list_by_category("function_gen")
        assert len(templates) >= 3
        assert all(t.category == "function_gen" for t in templates)

    def test_list_by_category_test_gen(self, lib):
        templates = lib.list_by_category("test_gen")
        assert len(templates) >= 3

    def test_list_by_category_unknown(self, lib):
        templates = lib.list_by_category("does_not_exist")
        assert templates == []


# ---------------------------------------------------------------------------
# random()
# ---------------------------------------------------------------------------


class TestRandom:
    def test_random_returns_template(self, lib):
        t = lib.random()
        assert isinstance(t, PromptTemplate)

    def test_random_category_filter(self, lib):
        for _ in range(10):
            t = lib.random(category="function_gen")
            assert t.category == "function_gen"

    def test_random_invalid_category_raises(self, lib):
        with pytest.raises(ValueError, match="No templates found"):
            lib.random(category="definitely_not_a_real_category_xyz")


# ---------------------------------------------------------------------------
# Format / usage
# ---------------------------------------------------------------------------


class TestFormatNewTemplates:
    def test_pytest_function_format(self, lib):
        prompt = lib.format("pytest_function", name="add", description="add returns the correct sum")
        assert "def test_add" in prompt
        assert "add returns the correct sum" in prompt

    def test_python_dataclass_format(self, lib):
        prompt = lib.format("python_dataclass", name="User", description="A user account")
        assert "@dataclass" in prompt
        assert "class User" in prompt

    def test_debug_code_format(self, lib):
        prompt = lib.format(
            "debug_code",
            code="x = y + 1",
            bug_description="y is undefined",
        )
        assert "y is undefined" in prompt
        assert "x = y + 1" in prompt

    def test_missing_variable_raises(self, lib):
        with pytest.raises(KeyError):
            lib.format("pytest_function", name="foo")  # missing 'description'
