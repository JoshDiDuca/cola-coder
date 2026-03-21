"""Output Format Validator: validate generated code output formatting.

Checks:
  - Proper indentation: consistent spaces or tabs (no mixed)
  - Consistent line endings: all CRLF or all LF (no mixed)
  - No trailing whitespace on any line
  - UTF-8 compliance (no lone surrogates, no null bytes outside strings)
  - Max line length (default: 100 chars per project standard)
  - No BOM (byte-order mark) at start
  - Final newline present

For a TS dev: like Prettier's format check — all the mechanical style rules
that should pass before a diff is even reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FormatViolation:
    """A single formatting violation."""

    check: str  # e.g. "trailing_whitespace"
    line: int | None  # 1-indexed line number, or None for file-level
    message: str


@dataclass
class FormatReport:
    """Results of output format validation."""

    violations: list[FormatViolation] = field(default_factory=list)
    line_ending: str = "lf"  # "lf" | "crlf" | "mixed" | "none"
    indentation: str = "spaces"  # "spaces" | "tabs" | "mixed" | "none"
    max_line_length: int = 100
    longest_line: int = 0

    @property
    def is_valid(self) -> bool:
        return len(self.violations) == 0

    @property
    def score(self) -> float:
        """0.0–1.0; deduct per violation, capped at 0."""
        deduction = min(1.0, len(self.violations) * 0.1)
        return max(0.0, 1.0 - deduction)

    @property
    def summary(self) -> str:
        if self.is_valid:
            return "OK — no format violations"
        checks = sorted({v.check for v in self.violations})
        return f"{len(self.violations)} violation(s): {', '.join(checks)}"


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class OutputFormatValidator:
    """Validate the format of a generated code string."""

    def __init__(self, max_line_length: int = 100) -> None:
        self.max_line_length = max_line_length

    def validate(self, text: str) -> FormatReport:
        """Run all format checks on *text* and return a :class:`FormatReport`."""
        report = FormatReport(max_line_length=self.max_line_length)
        violations = report.violations

        # UTF-8 / encoding checks on the raw bytes (if we have a str, encode it)
        try:
            raw = text.encode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            violations.append(
                FormatViolation("utf8_compliance", None, "Source contains non-UTF-8 characters")
            )
            return report

        # BOM check
        if raw.startswith(b"\xef\xbb\xbf"):
            violations.append(
                FormatViolation("bom", None, "File starts with UTF-8 BOM")
            )

        # Null bytes
        if b"\x00" in raw:
            violations.append(
                FormatViolation("null_bytes", None, "File contains null bytes")
            )

        if not text:
            return report

        # Line ending detection
        has_crlf = "\r\n" in text
        # After removing CRLF, check for lone CR
        has_lone_cr = "\r" in text.replace("\r\n", "")
        has_lf = "\n" in text.replace("\r\n", "")

        if has_crlf and has_lf:
            report.line_ending = "mixed"
            violations.append(
                FormatViolation("line_endings", None, "Mixed CRLF and LF line endings")
            )
        elif has_crlf:
            report.line_ending = "crlf"
        elif has_lf:
            report.line_ending = "lf"
        else:
            report.line_ending = "none"

        if has_lone_cr:
            violations.append(
                FormatViolation("line_endings", None, "Lone CR characters found (old Mac line endings)")
            )

        # Final newline
        if not text.endswith("\n"):
            violations.append(
                FormatViolation("final_newline", None, "File does not end with a newline")
            )

        # Per-line checks
        lines = text.splitlines()
        indent_spaces = 0
        indent_tabs = 0

        for i, line in enumerate(lines, start=1):
            # Trailing whitespace
            if line != line.rstrip():
                violations.append(
                    FormatViolation(
                        "trailing_whitespace",
                        i,
                        f"Line {i} has trailing whitespace",
                    )
                )

            # Line length
            line_len = len(line)
            if line_len > report.longest_line:
                report.longest_line = line_len
            if line_len > self.max_line_length:
                violations.append(
                    FormatViolation(
                        "line_too_long",
                        i,
                        f"Line {i} is {line_len} chars (limit {self.max_line_length})",
                    )
                )

            # Indentation counting (skip blank lines)
            if line and line[0] in (" ", "\t"):
                leading = len(line) - len(line.lstrip())
                leading_chars = line[:leading]
                if " " in leading_chars:
                    indent_spaces += 1
                if "\t" in leading_chars:
                    indent_tabs += 1

        # Indentation consistency
        if indent_spaces > 0 and indent_tabs > 0:
            report.indentation = "mixed"
            violations.append(
                FormatViolation(
                    "indentation",
                    None,
                    f"Mixed indentation: {indent_spaces} space-indented, {indent_tabs} tab-indented lines",
                )
            )
        elif indent_tabs > 0:
            report.indentation = "tabs"
        elif indent_spaces > 0:
            report.indentation = "spaces"
        else:
            report.indentation = "none"

        return report
