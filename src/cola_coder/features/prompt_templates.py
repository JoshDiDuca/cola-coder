"""Prompt Templates: reusable templates for common code generation tasks.

Provides structured prompts that produce better results from the model by
framing the task clearly. Templates use {variable} substitution.

For a TS dev: like template literals (`Hello ${name}`) but for AI prompts,
with built-in templates for common coding tasks.
"""

from dataclasses import dataclass, field

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class PromptTemplate:
    """A reusable prompt template with variable substitution."""
    name: str
    description: str
    template: str
    variables: list[str] = field(default_factory=list)
    category: str = "general"
    stop_tokens: list[str] = field(default_factory=list)

    def format(self, **kwargs) -> str:
        """Fill in template variables.

        Args:
            **kwargs: Variable name -> value mappings

        Returns:
            Formatted prompt string

        Raises:
            KeyError: If a required variable is missing
        """
        missing = [v for v in self.variables if v not in kwargs]
        if missing:
            raise KeyError(f"Missing template variables: {missing}")
        return self.template.format(**kwargs)

    def format_safe(self, **kwargs) -> str:
        """Fill in template variables, leaving missing ones as {name}."""
        result = self.template
        for key, value in kwargs.items():
            result = result.replace("{" + key + "}", str(value))
        return result


class PromptLibrary:
    """Collection of prompt templates for code generation."""

    def __init__(self):
        self._templates: dict[str, PromptTemplate] = {}
        self._load_builtin_templates()

    def _load_builtin_templates(self):
        """Register all built-in templates."""
        builtins = [
            # ── Function Completion ──
            PromptTemplate(
                name="function_complete",
                description="Complete a function given its signature",
                template="// Complete this function\n{signature}\n",
                variables=["signature"],
                category="completion",
                stop_tokens=["\n\n", "\nfunction ", "\nexport ", "\nclass "],
            ),
            PromptTemplate(
                name="function_body",
                description="Generate function body from name, params, and description",
                template=(
                    "/**\n * {description}\n */\n"
                    "function {name}({params}): {return_type} {{\n"
                ),
                variables=["name", "params", "return_type", "description"],
                category="completion",
                stop_tokens=["\n}\n"],
            ),

            # ── Docstring / Documentation ──
            PromptTemplate(
                name="docstring",
                description="Generate a docstring for existing code",
                template=(
                    "{code}\n\n"
                    "// Write a JSDoc comment for the above function:\n"
                    "/**\n * "
                ),
                variables=["code"],
                category="documentation",
                stop_tokens=["*/"],
            ),
            PromptTemplate(
                name="explain",
                description="Explain what code does",
                template=(
                    "{code}\n\n"
                    "// Explanation of the above code:\n// "
                ),
                variables=["code"],
                category="documentation",
                stop_tokens=["\n\n"],
            ),

            # ── Test Generation ──
            PromptTemplate(
                name="unit_test",
                description="Generate unit tests for a function",
                template=(
                    "{code}\n\n"
                    "// Unit tests for {function_name}:\n"
                    "describe('{function_name}', () => {{\n"
                    "  it('"
                ),
                variables=["code", "function_name"],
                category="testing",
                stop_tokens=["\n});"],
            ),
            PromptTemplate(
                name="test_case",
                description="Generate a single test case",
                template=(
                    "// Test that {description}\n"
                    "it('{description}', () => {{\n"
                ),
                variables=["description"],
                category="testing",
                stop_tokens=["\n});"],
            ),

            # ── Type Definitions ──
            PromptTemplate(
                name="interface",
                description="Generate a TypeScript interface",
                template=(
                    "// {description}\n"
                    "interface {name} {{\n"
                ),
                variables=["name", "description"],
                category="types",
                stop_tokens=["\n}\n"],
            ),
            PromptTemplate(
                name="type_from_data",
                description="Infer TypeScript types from example data",
                template=(
                    "// Given this data:\n"
                    "const example = {data};\n\n"
                    "// The TypeScript type for this data is:\n"
                    "type {name} = "
                ),
                variables=["data", "name"],
                category="types",
                stop_tokens=[";\n"],
            ),

            # ── Refactoring ──
            PromptTemplate(
                name="refactor",
                description="Refactor code with a specific goal",
                template=(
                    "// Original code:\n{code}\n\n"
                    "// Refactored version ({goal}):\n"
                ),
                variables=["code", "goal"],
                category="refactoring",
                stop_tokens=["\n\n\n"],
            ),
            PromptTemplate(
                name="convert_to_async",
                description="Convert synchronous code to async/await",
                template=(
                    "// Synchronous version:\n{code}\n\n"
                    "// Async version using async/await:\n"
                ),
                variables=["code"],
                category="refactoring",
                stop_tokens=["\n\n\n"],
            ),

            # ── Code Patterns ──
            PromptTemplate(
                name="react_component",
                description="Generate a React component",
                template=(
                    "import React from 'react';\n\n"
                    "interface {name}Props {{\n"
                    "  {props}\n"
                    "}}\n\n"
                    "/**\n * {description}\n */\n"
                    "export const {name}: React.FC<{name}Props> = ("
                ),
                variables=["name", "props", "description"],
                category="patterns",
                stop_tokens=["\nexport "],
            ),
            PromptTemplate(
                name="api_endpoint",
                description="Generate an API endpoint handler",
                template=(
                    "// {method} {path}\n"
                    "// {description}\n"
                    "export async function handle{name}(req: Request): Promise<Response> {{\n"
                ),
                variables=["method", "path", "name", "description"],
                category="patterns",
                stop_tokens=["\n}\n"],
            ),
            PromptTemplate(
                name="error_handler",
                description="Generate error handling code",
                template=(
                    "// Error handling for: {context}\n"
                    "try {{\n"
                    "  {code}\n"
                    "}} catch (error) {{\n"
                ),
                variables=["context", "code"],
                category="patterns",
                stop_tokens=["\n}\n"],
            ),

            # ── Fill in the Middle (FIM) ──
            PromptTemplate(
                name="fim",
                description="Fill in the middle — complete code between prefix and suffix",
                template="{prefix}<FILL>{suffix}",
                variables=["prefix", "suffix"],
                category="fim",
                stop_tokens=["<FILL>"],
            ),

            # ── Python-specific ──
            PromptTemplate(
                name="python_function",
                description="Generate a Python function",
                template=(
                    "def {name}({params}):\n"
                    '    """{description}"""\n'
                    "    "
                ),
                variables=["name", "params", "description"],
                category="python",
                stop_tokens=["\n\ndef ", "\n\nclass "],
            ),
            PromptTemplate(
                name="python_class",
                description="Generate a Python class",
                template=(
                    "class {name}:\n"
                    '    """{description}"""\n\n'
                    "    def __init__(self"
                ),
                variables=["name", "description"],
                category="python",
                stop_tokens=["\n\nclass "],
            ),
        ]

        for template in builtins:
            self._templates[template.name] = template

        self._load_extended_templates()

    def _load_extended_templates(self):  # noqa: PLR0915
        """Register extended code-specific templates (20+)."""
        extended = [
            # ── Function generation ───────────────────────────────────────
            PromptTemplate(
                name="python_function_typed",
                description="Generate a fully-typed Python function with docstring",
                template=(
                    "def {name}({params}) -> {return_type}:\n"
                    '    """{description}\n\n'
                    "    Args:\n"
                    "        {args_doc}\n\n"
                    "    Returns:\n"
                    "        {return_doc}\n"
                    '    """\n'
                    "    "
                ),
                variables=["name", "params", "return_type", "description", "args_doc", "return_doc"],
                category="function_gen",
            ),
            PromptTemplate(
                name="async_function",
                description="Generate an async Python function",
                template=(
                    "async def {name}({params}) -> {return_type}:\n"
                    '    """{description}"""\n'
                    "    "
                ),
                variables=["name", "params", "return_type", "description"],
                category="function_gen",
                stop_tokens=["\n\nasync def ", "\n\ndef "],
            ),
            PromptTemplate(
                name="lambda_function",
                description="Generate a Python lambda",
                template="{name} = lambda {params}: ",
                variables=["name", "params"],
                category="function_gen",
                stop_tokens=["\n"],
            ),
            PromptTemplate(
                name="generator_function",
                description="Generate a Python generator function",
                template=(
                    "def {name}({params}):\n"
                    '    """Yield {description}."""\n'
                    "    "
                ),
                variables=["name", "params", "description"],
                category="function_gen",
                stop_tokens=["\n\ndef "],
            ),
            # ── Class generation ──────────────────────────────────────────
            PromptTemplate(
                name="python_dataclass",
                description="Generate a Python dataclass",
                template=(
                    "from dataclasses import dataclass, field\n\n"
                    "@dataclass\n"
                    "class {name}:\n"
                    '    """{description}"""\n\n'
                    "    "
                ),
                variables=["name", "description"],
                category="class_gen",
                stop_tokens=["\n\nclass "],
            ),
            PromptTemplate(
                name="python_enum",
                description="Generate a Python Enum class",
                template=(
                    "from enum import Enum\n\n"
                    "class {name}(Enum):\n"
                    '    """{description}"""\n\n'
                    "    "
                ),
                variables=["name", "description"],
                category="class_gen",
                stop_tokens=["\n\nclass "],
            ),
            PromptTemplate(
                name="abstract_base_class",
                description="Generate an abstract base class",
                template=(
                    "from abc import ABC, abstractmethod\n\n"
                    "class {name}(ABC):\n"
                    '    """{description}"""\n\n'
                    "    @abstractmethod\n"
                    "    def {method}(self"
                ),
                variables=["name", "description", "method"],
                category="class_gen",
                stop_tokens=["\n\nclass "],
            ),
            PromptTemplate(
                name="protocol_class",
                description="Generate a typing.Protocol",
                template=(
                    "from typing import Protocol, runtime_checkable\n\n"
                    "@runtime_checkable\n"
                    "class {name}(Protocol):\n"
                    '    """{description}"""\n\n'
                    "    def {method}(self"
                ),
                variables=["name", "description", "method"],
                category="class_gen",
                stop_tokens=["\n\nclass "],
            ),
            # ── Unit-test generation ──────────────────────────────────────
            PromptTemplate(
                name="pytest_function",
                description="Generate a pytest test function",
                template=(
                    "def test_{name}():\n"
                    '    """Test that {description}."""\n'
                    "    "
                ),
                variables=["name", "description"],
                category="test_gen",
                stop_tokens=["\n\ndef test_"],
            ),
            PromptTemplate(
                name="pytest_class",
                description="Generate a pytest test class",
                template=(
                    "class Test{name}:\n"
                    '    """{description}."""\n\n'
                    "    def test_{first_test}(self):\n"
                    "        "
                ),
                variables=["name", "description", "first_test"],
                category="test_gen",
                stop_tokens=["\n\nclass Test"],
            ),
            PromptTemplate(
                name="pytest_parametrize",
                description="Generate a parametrized pytest test",
                template=(
                    "import pytest\n\n"
                    "@pytest.mark.parametrize(\"{param}\", {values})\n"
                    "def test_{name}({param}):\n"
                    "    "
                ),
                variables=["param", "values", "name"],
                category="test_gen",
                stop_tokens=["\n\n@pytest", "\n\ndef test_"],
            ),
            PromptTemplate(
                name="mock_patch",
                description="Generate a test with unittest.mock.patch",
                template=(
                    "from unittest.mock import patch, MagicMock\n\n"
                    "def test_{name}():\n"
                    "    with patch(\"{target}\") as mock_{alias}:\n"
                    "        mock_{alias}.return_value = {return_value}\n"
                    "        "
                ),
                variables=["name", "target", "alias", "return_value"],
                category="test_gen",
                stop_tokens=["\n\ndef test_"],
            ),
            # ── Refactor / transform ──────────────────────────────────────
            PromptTemplate(
                name="extract_function",
                description="Extract a block of code into its own function",
                template=(
                    "# Extract the following block into a separate function named {name}:\n"
                    "{code}\n\n"
                    "# Extracted function:\n"
                    "def {name}("
                ),
                variables=["name", "code"],
                category="refactor",
            ),
            PromptTemplate(
                name="add_type_hints",
                description="Add type hints to an existing function",
                template=(
                    "# Original (no type hints):\n{code}\n\n"
                    "# With full type annotations:\n"
                ),
                variables=["code"],
                category="refactor",
                stop_tokens=["\n\n\n"],
            ),
            PromptTemplate(
                name="convert_to_dataclass",
                description="Convert a plain class to a dataclass",
                template=(
                    "# Original class:\n{code}\n\n"
                    "# As a dataclass:\n"
                    "from dataclasses import dataclass\n\n"
                    "@dataclass\n"
                ),
                variables=["code"],
                category="refactor",
                stop_tokens=["\n\n\n"],
            ),
            # ── Explain / debug ───────────────────────────────────────────
            PromptTemplate(
                name="explain_step_by_step",
                description="Explain code step by step",
                template=(
                    "{code}\n\n"
                    "# Step-by-step explanation:\n"
                    "# Step 1:"
                ),
                variables=["code"],
                category="explain",
                stop_tokens=["\n\n\n"],
            ),
            PromptTemplate(
                name="debug_code",
                description="Find and fix a bug in the given code",
                template=(
                    "# Buggy code:\n{code}\n\n"
                    "# Bug: {bug_description}\n\n"
                    "# Fixed code:\n"
                ),
                variables=["code", "bug_description"],
                category="debug",
                stop_tokens=["\n\n\n"],
            ),
            PromptTemplate(
                name="add_error_handling",
                description="Add error handling to existing code",
                template=(
                    "# Original code without error handling:\n{code}\n\n"
                    "# Version with proper error handling:\n"
                ),
                variables=["code"],
                category="debug",
                stop_tokens=["\n\n\n"],
            ),
            # ── Optimise ─────────────────────────────────────────────────
            PromptTemplate(
                name="optimize_time_complexity",
                description="Optimise code for better time complexity",
                template=(
                    "# O({current_complexity}) implementation:\n{code}\n\n"
                    "# Optimised version (target: O({target_complexity})):\n"
                ),
                variables=["current_complexity", "code", "target_complexity"],
                category="optimize",
                stop_tokens=["\n\n\n"],
            ),
            PromptTemplate(
                name="optimize_memory",
                description="Optimise code to reduce memory usage",
                template=(
                    "# Memory-heavy implementation:\n{code}\n\n"
                    "# Memory-efficient version:\n"
                ),
                variables=["code"],
                category="optimize",
                stop_tokens=["\n\n\n"],
            ),
            PromptTemplate(
                name="vectorize_loop",
                description="Replace Python loop with vectorised numpy/list-comp",
                template=(
                    "# Loop-based version:\n{code}\n\n"
                    "# Vectorised / list-comprehension version:\n"
                ),
                variables=["code"],
                category="optimize",
                stop_tokens=["\n\n\n"],
            ),
            # ── TypeScript-specific ───────────────────────────────────────
            PromptTemplate(
                name="ts_generic_function",
                description="Generate a generic TypeScript function",
                template=(
                    "function {name}<{type_param}>({params}: {input_type}): {return_type} {{\n"
                    "  "
                ),
                variables=["name", "type_param", "params", "input_type", "return_type"],
                category="typescript",
                stop_tokens=["\n}\n"],
            ),
            PromptTemplate(
                name="ts_zod_schema",
                description="Generate a Zod validation schema",
                template=(
                    "import {{ z }} from 'zod';\n\n"
                    "const {name}Schema = z.object({{\n"
                    "  "
                ),
                variables=["name"],
                category="typescript",
                stop_tokens=["});\n"],
            ),
            PromptTemplate(
                name="ts_react_hook",
                description="Generate a custom React hook",
                template=(
                    "import {{ useState, useEffect }} from 'react';\n\n"
                    "export function use{name}({params}) {{\n"
                    "  "
                ),
                variables=["name", "params"],
                category="typescript",
                stop_tokens=["\n}\n"],
            ),
        ]

        for template in extended:
            self._templates[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        """Get a template by name.

        Raises:
            KeyError: If template not found
        """
        if name not in self._templates:
            raise KeyError(
                f"Unknown template '{name}'. "
                f"Available: {', '.join(sorted(self._templates.keys()))}"
            )
        return self._templates[name]

    def register(self, template: PromptTemplate) -> None:
        """Register a custom template."""
        self._templates[template.name] = template

    def list_templates(self, category: str | None = None) -> list[PromptTemplate]:
        """List available templates, optionally filtered by category."""
        templates = list(self._templates.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return sorted(templates, key=lambda t: (t.category, t.name))

    def categories(self) -> list[str]:
        """List all template categories."""
        return sorted(set(t.category for t in self._templates.values()))

    def format(self, template_name: str, **kwargs) -> str:
        """Shortcut: get template and format it in one call."""
        return self.get(template_name).format(**kwargs)

    def format_with_stops(self, template_name: str, **kwargs) -> tuple[str, list[str]]:
        """Format template and return (prompt, stop_tokens) tuple.

        Useful for passing directly to a generator.
        """
        template = self.get(template_name)
        return template.format(**kwargs), template.stop_tokens

    def list_by_category(self, category: str) -> list[PromptTemplate]:
        """Return all templates for a given category (alias for list_templates)."""
        return self.list_templates(category=category)

    def random(self, category: str | None = None) -> PromptTemplate:
        """Return a random template, optionally filtered by category.

        Raises:
            ValueError: If no templates match the filter.
        """
        import random as _random

        candidates = self.list_templates(category=category)
        if not candidates:
            raise ValueError(
                f"No templates found for category '{category}'."
                if category
                else "Template library is empty."
            )
        return _random.choice(candidates)


# Default library instance
default_library = PromptLibrary()
