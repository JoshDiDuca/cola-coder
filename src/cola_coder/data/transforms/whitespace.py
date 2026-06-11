"""Whitespace normalization transform.

Normalizes indentation and removes trailing whitespace.
Keeps the code semantically identical but cleans up formatting noise.
"""

from __future__ import annotations

from cola_coder.data.pipeline import DataRecord, Transform
from cola_coder.data.registry import register_transform


@register_transform("normalize_whitespace")
class NormalizeWhitespace(Transform):
    """Normalize indentation and trailing whitespace in code.

    Operations:
    - Convert tabs to spaces (configurable tab width, default 4)
    - Remove trailing whitespace from each line
    - Collapse multiple consecutive blank lines into at most 2
    - Remove trailing newlines at end of file, ensure single final newline
    """

    def __init__(self, tab_width: int = 4) -> None:
        self._tab_width = tab_width

    def name(self) -> str:
        return "normalize_whitespace"

    def apply(self, record: DataRecord) -> DataRecord:
        content = record.content

        # Convert tabs to spaces
        content = content.expandtabs(self._tab_width)

        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in content.split("\n")]

        # Collapse multiple consecutive blank lines into max 2
        result_lines: list[str] = []
        blank_count = 0
        for line in lines:
            if line == "":
                blank_count += 1
                if blank_count <= 2:
                    result_lines.append(line)
            else:
                blank_count = 0
                result_lines.append(line)

        # Strip trailing blank lines, ensure single final newline
        while result_lines and result_lines[-1] == "":
            result_lines.pop()
        content = "\n".join(result_lines) + "\n"

        return DataRecord(content=content, metadata=record.metadata)
