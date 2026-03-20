"""Reasoning Trace Viewer: parse and format model reasoning traces for human inspection.

Structures raw generation text into typed steps (thinking, deciding, coding, verifying),
highlights key decision points, and exports to plain text or markdown. Useful for
debugging why a model chose a particular approach or where reasoning went wrong.

For a TS dev: like a structured logger for model thought — each step is a typed log
entry, and the trace is the full log session you can query and format.

CLI flag: --trace-viewer
"""

import re
from dataclasses import dataclass, field
from typing import Literal

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Step type literal
# ---------------------------------------------------------------------------

StepType = Literal["thinking", "deciding", "coding", "verifying"]


# ---------------------------------------------------------------------------
# TraceStep dataclass
# ---------------------------------------------------------------------------

@dataclass
class TraceStep:
    """A single step in a reasoning trace.

    Attributes:
        step_type: One of 'thinking', 'deciding', 'coding', 'verifying'.
        content:   The text content of this step.
        tokens_used: Approximate token count consumed by this step.
    """
    step_type: StepType
    content: str
    tokens_used: int = 0


# ---------------------------------------------------------------------------
# Decision-indicator patterns used by highlight_decisions and the parser
# ---------------------------------------------------------------------------

_DECISION_PATTERNS = [
    re.compile(r"\b(I decide[ds]?|decided to|will use|chose|choosing|going with|going to use|"
               r"let'?s use|I'?ll use|best approach|optimal solution|I choose|"
               r"use a|using a|approach:)\b", re.IGNORECASE),
]

_THINKING_PATTERNS = [
    re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"\bthink(?:ing)?\b[:\-–]?\s*(.+)", re.IGNORECASE),
]

_DECIDING_PATTERNS = [
    re.compile(r"\bdecision[:\-–]?\s*(.+)", re.IGNORECASE),
    re.compile(r"\bI decide[ds]?\s+(.+)", re.IGNORECASE),
    re.compile(r"\bwill use\s+(.+)", re.IGNORECASE),
    re.compile(r"\bchose\s+(.+)", re.IGNORECASE),
]

_CODING_PATTERNS = [
    re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL),
    re.compile(r"\bcode[:\-–]?\s*(.+)", re.IGNORECASE),
]

_VERIFYING_PATTERNS = [
    re.compile(r"\bverif(?:y|ying|ied)[:\-–]?\s*(.+)", re.IGNORECASE),
    re.compile(r"\bcheck[s]?[:\-–]?\s*(.+)", re.IGNORECASE),
    re.compile(r"\blooks? correct\b(.*)"),
    re.compile(r"\btest[s]?[:\-–]?\s*(.+)", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# ReasoningTrace
# ---------------------------------------------------------------------------

class ReasoningTrace:
    """A structured representation of a model reasoning trace.

    Parse from raw text with __init__(raw_text), add steps programmatically
    with add_step(), then inspect or export with the various format/query methods.

    Usage::

        trace = ReasoningTrace('<think>Hmm...</think>\\nDecision: Use a hash map')
        trace.add_step('verifying', 'Handles edge cases', tokens_used=5)
        print(trace.format_markdown())
    """

    def __init__(self, raw_text: str) -> None:
        """Parse a reasoning trace from raw generation text.

        Looks for common patterns like <think> tags, "Decision:", "Code:", etc.
        Steps are appended in the order they are discovered in the text.

        Args:
            raw_text: The raw generation string to parse.
        """
        self._steps: list[TraceStep] = []
        if raw_text and raw_text.strip():
            self._parse(raw_text)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_step(self, step_type: StepType, content: str, tokens_used: int = 0) -> None:
        """Append a step to the trace.

        Args:
            step_type:   One of 'thinking', 'deciding', 'coding', 'verifying'.
            content:     The step's text content.
            tokens_used: Approximate token count for this step.
        """
        self._steps.append(TraceStep(step_type=step_type, content=content, tokens_used=tokens_used))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_steps(self) -> list[TraceStep]:
        """Return all steps in insertion order."""
        return list(self._steps)

    def get_decisions(self) -> list[TraceStep]:
        """Return only steps of type 'deciding'."""
        return [s for s in self._steps if s.step_type == "deciding"]

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_plain(self) -> str:
        """Format the trace as plain text.

        Each step is prefixed with its type in uppercase brackets, e.g.::

            [THINKING] Need to consider the data structure
            [DECIDING] Will use a hash map for O(1) lookups
            [CODING] const map = new Map<string, number>()
            [VERIFYING] This handles the lookup requirement

        Returns:
            Multi-line plain-text string. Empty string if there are no steps.
        """
        if not self._steps:
            return ""
        lines: list[str] = []
        for step in self._steps:
            label = f"[{step.step_type.upper()}]"
            tokens_note = f" ({step.tokens_used} tokens)" if step.tokens_used else ""
            lines.append(f"{label}{tokens_note}")
            lines.append(step.content.strip())
            lines.append("")
        return "\n".join(lines).rstrip()

    def format_markdown(self) -> str:
        """Format the trace as Markdown.

        Each step becomes a level-3 heading with its type, followed by a
        blockquote (thinking), fenced code block (coding), or plain paragraph
        (deciding / verifying).

        Returns:
            Markdown string. Empty string if there are no steps.
        """
        if not self._steps:
            return ""

        _MD_ICONS = {
            "thinking": "💭",
            "deciding": "🎯",
            "coding": "💻",
            "verifying": "✅",
        }

        parts: list[str] = []
        for step in self._steps:
            icon = _MD_ICONS.get(step.step_type, "•")
            tokens_note = f" _{step.tokens_used} tokens_" if step.tokens_used else ""
            heading = f"### {icon} {step.step_type.capitalize()}{tokens_note}"
            parts.append(heading)
            content = step.content.strip()

            if step.step_type == "thinking":
                quoted = "\n".join(f"> {line}" for line in content.splitlines())
                parts.append(quoted)
            elif step.step_type == "coding":
                parts.append(f"```\n{content}\n```")
            else:
                parts.append(content)
            parts.append("")

        return "\n".join(parts).rstrip()

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def total_tokens(self) -> int:
        """Return the sum of tokens_used across all steps."""
        return sum(s.tokens_used for s in self._steps)

    def summary(self) -> dict:
        """Return a summary dictionary describing the trace.

        Keys:
            total_steps (int): Total number of steps.
            step_counts (dict[str, int]): Count of each step type.
            total_tokens (int): Total tokens across all steps.
            decision_count (int): Number of 'deciding' steps.
            has_code (bool): True if at least one 'coding' step exists.
            has_verification (bool): True if at least one 'verifying' step exists.
        """
        step_counts: dict[str, int] = {}
        for step in self._steps:
            step_counts[step.step_type] = step_counts.get(step.step_type, 0) + 1

        return {
            "total_steps": len(self._steps),
            "step_counts": step_counts,
            "total_tokens": self.total_tokens(),
            "decision_count": step_counts.get("deciding", 0),
            "has_code": step_counts.get("coding", 0) > 0,
            "has_verification": step_counts.get("verifying", 0) > 0,
        }

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _parse(self, text: str) -> None:
        """Heuristically parse raw text into steps.

        Strategy (in order of specificity):
        1. Extract <think>...</think> blocks as 'thinking' steps.
        2. Extract fenced code blocks (``` ... ```) as 'coding' steps.
        3. Scan remaining lines for keyword prefixes:
           - "Decision:" / "I decided to" / "will use" / "chose" -> 'deciding'
           - "Verify:" / "Check:" / "looks correct" / "Test:" -> 'verifying'
           - "Think:" / "Step N: Think" -> 'thinking'
           - "Code:" -> 'coding'
        4. Lines not matched by any pattern are silently skipped (they may be
           blank or structural text).
        """
        # Track which character ranges are already consumed so we don't
        # double-classify the same text.
        consumed: set[int] = set()

        # --- Pass 1: <think> blocks ---
        for m in re.finditer(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE):
            content = m.group(1).strip()
            if content:
                self._steps.append(TraceStep("thinking", content))
            consumed.update(range(m.start(), m.end()))

        # --- Pass 2: fenced code blocks ---
        for m in re.finditer(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL):
            content = m.group(1).strip()
            if content and not self._range_consumed(m, consumed):
                self._steps.append(TraceStep("coding", content))
                consumed.update(range(m.start(), m.end()))

        # --- Pass 3: line-by-line keyword matching on unconsumed text ---
        # Rebuild consumed as a set of line indices for speed.
        consumed_char_set = consumed

        # Build a char-offset view of lines to know which lines are consumed.
        offset = 0
        for line in text.splitlines(keepends=True):
            line_range = set(range(offset, offset + len(line)))
            if not line_range.intersection(consumed_char_set):
                stripped = line.strip()
                if stripped:
                    step = self._classify_line(stripped)
                    if step is not None:
                        self._steps.append(step)
            offset += len(line)

    @staticmethod
    def _range_consumed(m: re.Match, consumed: set[int]) -> bool:
        return bool(set(range(m.start(), m.end())).intersection(consumed))

    @staticmethod
    def _classify_line(line: str) -> "TraceStep | None":
        """Map a single line of text to a TraceStep, or None if unclassified."""
        lower = line.lower()

        # --- deciding ---
        deciding_re = re.compile(
            r"^(?:decision[:\-–]\s*|i decided?\s+to\s+|will use\s+|chose\s+|"
            r"i choose\s+|going with\s+|going to use\s+|best approach[:\-–]\s*|"
            r"optimal solution[:\-–]\s*)",
            re.IGNORECASE,
        )
        if deciding_re.match(line):
            content = deciding_re.sub("", line).strip() or line
            return TraceStep("deciding", content)

        # Check inline decision signals in the middle of the line
        for pat in _DECISION_PATTERNS:
            if pat.search(line):
                # Only promote to 'deciding' if the match appears prominent
                # (near the start of the line or after a sentence boundary).
                m = pat.search(line)
                if m and m.start() < len(line) // 2:
                    return TraceStep("deciding", line)

        # --- verifying ---
        verifying_re = re.compile(
            r"^(?:verif(?:y|ying|ied)[:\-–]\s*|check[s]?[:\-–]\s*|"
            r"looks?\s+correct[\.,]?|test[s]?[:\-–]\s*)",
            re.IGNORECASE,
        )
        if verifying_re.match(line):
            content = verifying_re.sub("", line).strip() or line
            return TraceStep("verifying", content)

        # --- thinking ---
        thinking_re = re.compile(
            r"^(?:think(?:ing)?[:\-–]\s*|step\s+\d+[:\-–]\s*think(?:ing)?\b[:\-–]?\s*|"
            r"need to\s+|consider(?:ing)?\s+|hmm[,.]?\s*|let me\s+)",
            re.IGNORECASE,
        )
        if thinking_re.match(line):
            content = thinking_re.sub("", line).strip() or line
            return TraceStep("thinking", content)

        # --- coding ---
        coding_re = re.compile(
            r"^(?:code[:\-–]\s*|implementation[:\-–]\s*|def\s+\w|const\s+\w|"
            r"function\s+\w|class\s+\w|return\s+)",
            re.IGNORECASE,
        )
        if coding_re.match(line):
            content = coding_re.sub("", line, count=1).strip() or line
            return TraceStep("coding", content)

        return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def parse_trace(text: str) -> "ReasoningTrace":
    """Parse a reasoning trace from a raw text string.

    Convenience wrapper around ``ReasoningTrace(text)``.

    Args:
        text: Raw generation or log text to parse.

    Returns:
        A ``ReasoningTrace`` instance populated with any discovered steps.
    """
    return ReasoningTrace(text)


def highlight_decisions(trace: str) -> str:
    """Extract and highlight decision points from a raw trace string.

    Scans each line for decision-related keywords and wraps matching lines
    in a ``[DECISION] ... [/DECISION]`` marker so they stand out.

    Args:
        trace: Raw reasoning trace text (string).

    Returns:
        The trace with decision lines wrapped in ``[DECISION]`` markers.
        Non-decision lines are passed through unchanged.
    """
    _INLINE_DECISION_RE = re.compile(
        r"\b(decide[ds]?|decided\s+to|will\s+use|chose|choosing|going\s+with|"
        r"going\s+to\s+use|let'?s\s+use|I'?ll\s+use|best\s+approach|"
        r"optimal\s+solution|I\s+choose|use\s+a|using\s+a)\b",
        re.IGNORECASE,
    )

    result_lines: list[str] = []
    for line in trace.splitlines():
        if _INLINE_DECISION_RE.search(line):
            result_lines.append(f"[DECISION] {line.strip()} [/DECISION]")
        else:
            result_lines.append(line)

    return "\n".join(result_lines)
