"""Length-based filter plugin.

Simple filter that rejects files based on line count.
Configurable min and max lines.
"""

from __future__ import annotations

from typing import Any

from cola_coder.data.pipeline import DataRecord, FilterPlugin
from cola_coder.data.registry import register_filter


@register_filter("length")
class LengthFilter(FilterPlugin):
    """Filter records by line count.

    Config options (via setup() or YAML):
        min_lines: Minimum number of lines (default: 1)
        max_lines: Maximum number of lines (default: 100000)
    """

    def __init__(
        self,
        min_lines: int = 1,
        max_lines: int = 100_000,
    ) -> None:
        self._min_lines = min_lines
        self._max_lines = max_lines

    def name(self) -> str:
        return f"length(min={self._min_lines}, max={self._max_lines})"

    def setup(self, config: dict[str, Any]) -> None:
        self._min_lines = config.get("min_lines", self._min_lines)
        self._max_lines = config.get("max_lines", self._max_lines)

    def check(self, record: DataRecord) -> tuple[bool, str]:
        line_count = record.content.count("\n") + 1

        if line_count < self._min_lines:
            return False, f"too_short ({line_count} lines, min={self._min_lines})"

        if line_count > self._max_lines:
            return False, f"too_long ({line_count} lines, max={self._max_lines})"

        return True, ""
