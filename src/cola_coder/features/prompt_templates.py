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


# Default library instance
default_library = PromptLibrary()
