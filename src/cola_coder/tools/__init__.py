"""Tool calling framework for cola-coder.

Enables the model to call external tools (run tests, lint, typecheck, etc.)
instead of hallucinating results. Supports structured JSON tool calls
and safe execution with timeouts.

Research backing:
- ToolBench/ToolLLM (2024): Fine-tuned small models achieve 77.55% tool accuracy
- BFCL V4: AST accuracy, execution accuracy, irrelevance detection metrics

Usage:
    from cola_coder.tools import ToolRegistry, ToolExecutor
"""

from cola_coder.tools.registry import ToolDefinition, ToolRegistry
from cola_coder.tools.executor import ToolExecutor
from cola_coder.tools.formatter import format_tool_call, parse_tool_call

__all__ = [
    "ToolRegistry",
    "ToolDefinition",
    "ToolExecutor",
    "format_tool_call",
    "parse_tool_call",
]
