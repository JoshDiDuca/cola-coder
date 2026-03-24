"""Tool registry for defining available tools.

Each tool has a name, description, parameter schema, and handler function.
The registry provides the tool definitions to the model in its system prompt.
"""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolParameter:
    """A single parameter for a tool."""

    name: str
    type: str  # "string", "integer", "boolean", "array"
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolDefinition:
    """Definition of a callable tool."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable | None = None
    enabled: bool = True
    category: str = "general"  # "testing", "linting", "git", "file", "search"

    def to_schema(self) -> dict:
        """Convert to JSON schema format for model prompts."""
        props: dict[str, dict] = {}
        required: list[str] = []
        for param in self.parameters:
            props[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        }


class ToolRegistry:
    """Registry of available tools.

    Tools can be registered manually or via built-in definitions.
    The registry generates system prompt text describing available tools.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._register_builtins()

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Unregister a tool. Returns True if found."""
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_enabled(self) -> list[ToolDefinition]:
        """List only enabled tools."""
        return [t for t in self._tools.values() if t.enabled]

    def enable(self, name: str) -> bool:
        """Enable a tool."""
        tool = self._tools.get(name)
        if tool:
            tool.enabled = True
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a tool."""
        tool = self._tools.get(name)
        if tool:
            tool.enabled = False
            return True
        return False

    def get_system_prompt(self) -> str:
        """Generate system prompt text describing available tools.

        Returns:
            Text to include in the system prompt showing available tools.
        """
        enabled = self.list_enabled()
        if not enabled:
            return ""

        lines = ["# Available Tools\n"]
        lines.append("You can call tools using the <tool_call> format:\n")
        lines.append("```")
        lines.append("<tool_call>")
        lines.append('{"name": "tool_name", "arguments": {"param": "value"}}')
        lines.append("</tool_call>")
        lines.append("```\n")
        lines.append("## Tools:\n")

        for tool in enabled:
            lines.append(f"### {tool.name}")
            lines.append(tool.description)
            if tool.parameters:
                lines.append("Parameters:")
                for param in tool.parameters:
                    req = " (required)" if param.required else " (optional)"
                    lines.append(f"  - {param.name}: {param.type} — {param.description}{req}")
            lines.append("")

        return "\n".join(lines)

    def _register_builtins(self) -> None:
        """Register built-in coding tools."""
        builtins = [
            ToolDefinition(
                name="run_tests",
                description="Run the test suite and return results.",
                parameters=[
                    ToolParameter(
                        "test_path", "string", "Path to test file or directory", required=False
                    ),
                    ToolParameter(
                        "verbose", "boolean", "Show verbose output", required=False, default=False
                    ),
                ],
                category="testing",
            ),
            ToolDefinition(
                name="lint",
                description="Run linter (ruff) on code and return diagnostics.",
                parameters=[
                    ToolParameter("file_path", "string", "Path to file to lint"),
                    ToolParameter(
                        "fix",
                        "boolean",
                        "Auto-fix fixable issues",
                        required=False,
                        default=False,
                    ),
                ],
                category="linting",
            ),
            ToolDefinition(
                name="typecheck",
                description="Run TypeScript type checker (tsc) on code.",
                parameters=[
                    ToolParameter("code", "string", "TypeScript code to check"),
                    ToolParameter(
                        "strict", "boolean", "Use strict mode", required=False, default=True
                    ),
                ],
                category="linting",
            ),
            ToolDefinition(
                name="search_code",
                description="Search the codebase for relevant code snippets.",
                parameters=[
                    ToolParameter("query", "string", "Search query"),
                    ToolParameter(
                        "max_results",
                        "integer",
                        "Max results to return",
                        required=False,
                        default=5,
                    ),
                ],
                category="search",
            ),
            ToolDefinition(
                name="read_file",
                description="Read the contents of a file.",
                parameters=[
                    ToolParameter("path", "string", "File path to read"),
                    ToolParameter(
                        "start_line", "integer", "Start line (optional)", required=False
                    ),
                    ToolParameter("end_line", "integer", "End line (optional)", required=False),
                ],
                category="file",
            ),
            ToolDefinition(
                name="git_diff",
                description="Show recent git changes.",
                parameters=[
                    ToolParameter(
                        "ref", "string", "Git ref to diff against", required=False, default="HEAD"
                    ),
                    ToolParameter(
                        "file_path", "string", "Specific file to diff", required=False
                    ),
                ],
                category="git",
            ),
            ToolDefinition(
                name="git_log",
                description="Show recent git commit history.",
                parameters=[
                    ToolParameter(
                        "count",
                        "integer",
                        "Number of commits to show",
                        required=False,
                        default=5,
                    ),
                ],
                category="git",
            ),
        ]

        for tool in builtins:
            self.register(tool)
