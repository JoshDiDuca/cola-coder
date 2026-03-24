"""Format and parse tool calls for model I/O.

The model generates tool calls in a structured format:
<tool_call>
{"name": "tool_name", "arguments": {"param": "value"}}
</tool_call>

This module handles formatting tool definitions for prompts
and parsing tool calls from model output.
"""

import json
import re
from typing import Any


def format_tool_call(name: str, arguments: dict) -> str:
    """Format a tool call for model output.

    Args:
        name: Tool name
        arguments: Tool arguments

    Returns:
        Formatted tool call string
    """
    call = {"name": name, "arguments": arguments}
    return f"<tool_call>\n{json.dumps(call, indent=2)}\n</tool_call>"


def format_tool_result(name: str, result: str, success: bool = True) -> str:
    """Format a tool result for feeding back to the model.

    Args:
        name: Tool name
        result: Tool output text
        success: Whether the tool succeeded

    Returns:
        Formatted tool result string for the 'tool' role
    """
    status = "success" if success else "error"
    return f"[{name}] ({status}):\n{result}"


def parse_tool_call(text: str) -> list[dict[str, Any]]:
    """Parse tool calls from model output.

    Extracts all <tool_call>...</tool_call> blocks from the text
    and parses the JSON content.

    Args:
        text: Model output text that may contain tool calls

    Returns:
        List of parsed tool call dicts, each with 'name' and 'arguments'
    """
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)

    calls: list[dict[str, Any]] = []
    for match in matches:
        try:
            parsed = json.loads(match)
            if "name" in parsed:
                calls.append(
                    {
                        "name": parsed["name"],
                        "arguments": parsed.get("arguments", {}),
                    }
                )
        except json.JSONDecodeError:
            continue

    return calls


def has_tool_call(text: str) -> bool:
    """Check if text contains a tool call.

    Args:
        text: Model output text

    Returns:
        True if at least one tool call is present
    """
    return "<tool_call>" in text and "</tool_call>" in text


def strip_tool_calls(text: str) -> str:
    """Remove all tool call blocks from text.

    Args:
        text: Text possibly containing tool calls

    Returns:
        Text with tool call blocks removed
    """
    pattern = r"<tool_call>\s*.*?\s*</tool_call>"
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()
