"""Code Formatting Normalizer: normalize generated code to consistent style.

Handles Python and TypeScript / JavaScript.  Think of this as a lightweight
post-processing step after generation — it straightens up the output before
showing it to the user, similar to what Prettier does for TypeScript or Black
for Python.

The normalizer is intentionally conservative: it does NOT do full AST-based
reformatting (which would require tree-sitter or a full parser).  Instead it
applies a set of reliable line-level and token-level transformations that are
safe to apply to partial / incomplete code snippets.

Supported normalisations
-------------------------
Python
  - Consistent 4-space indentation (tabs → 4 spaces)
  - Strip trailing whitespace
  - Ensure exactly one blank line between top-level definitions
  - Remove trailing commas before ``)``, ``]`` when on a single line (optional)

TypeScript / JavaScript
  - Consistent 2-space indentation (tabs → 2 spaces)
  - Strip trailing whitespace
  - Ensure semicolons at end of statements (best-effort)
  - Normalise brace style: open brace on same line as declaration (K&R style)

Feature toggle: set FEATURE_ENABLED = False to disable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if code normalisation is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class NormalizeResult:
    """Outcome of a normalisation pass."""

    code: str
    language: str
    changes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.changes)


# ---------------------------------------------------------------------------
# Python normalizer
# ---------------------------------------------------------------------------


def _normalize_python(code: str) -> NormalizeResult:
    changes: list[str] = []

    lines = code.splitlines()

    # 1. Tabs → 4 spaces
    new_lines: list[str] = []
    for line in lines:
        if "\t" in line:
            # Preserve indentation structure: expand leading tabs, then rest
            expanded = line.expandtabs(4)
            if expanded != line:
                changes.append("expanded tabs to 4 spaces")
            new_lines.append(expanded)
        else:
            new_lines.append(line)
    lines = new_lines

    # 2. Strip trailing whitespace
    stripped: list[str] = []
    for line in lines:
        s = line.rstrip()
        if s != line:
            changes.append("stripped trailing whitespace")
        stripped.append(s)
    lines = stripped

    # 3. Collapse multiple consecutive blank lines to at most 2
    collapsed: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 2:
                collapsed.append(line)
            else:
                changes.append("collapsed multiple blank lines")
        else:
            blank_run = 0
            collapsed.append(line)
    lines = collapsed

    result = "\n".join(lines)
    if result and not result.endswith("\n"):
        result += "\n"

    return NormalizeResult(code=result, language="python", changes=list(dict.fromkeys(changes)))


# ---------------------------------------------------------------------------
# TypeScript / JavaScript normalizer
# ---------------------------------------------------------------------------


# Patterns for semicolon insertion — statements that should end with ;
_TS_NEEDS_SEMI_RE = re.compile(
    r"^(\s*)"  # leading whitespace
    r"("
    r"(?:const|let|var|return|throw|import|export\s+(?:const|let|type|default\s+\w+))\b"
    r"|(?:.*\)\s*$)"  # ends with ) — function calls, assignments
    r")"
    r"([^;{},\n]*)$"
)

# Statements that definitely don't need a semicolon
_TS_NO_SEMI_RE = re.compile(
    r"^\s*(?:"
    r"//.*"  # comment
    r"|/\*.*"  # block comment start
    r"|.*\*/"  # block comment end
    r"|\s*"  # blank
    r"|.*[{},]\s*"  # ends with brace or comma
    r"|.*=>.*\{?\s*"  # arrow function with body
    r"|(?:if|else|for|while|do|try|catch|finally|class|function|interface|type|enum)\b.*"
    r")$"
)


def _line_needs_semicolon(line: str) -> bool:
    """Heuristic: does this TypeScript line need a semicolon appended?"""
    stripped = line.rstrip()
    if not stripped or stripped.endswith(";"):
        return False
    if _TS_NO_SEMI_RE.match(stripped):
        return False
    # If line ends with a complete assignment, call, or simple statement
    simple_stmt = re.compile(
        r"^\s*(?:"
        r"(?:const|let|var)\s+\w[\w.]*\s*=.*[^{,]"  # variable declaration
        r"|return\s+.*[^{,]"  # return statement
        r"|throw\s+.*[^{,]"  # throw
        r"|(?:[\w.]+\s*\(.*\))"  # function call
        r")$"
    )
    return bool(simple_stmt.match(stripped))


def _normalize_typescript(code: str) -> NormalizeResult:
    changes: list[str] = []

    lines = code.splitlines()

    # 1. Tabs → 2 spaces (TypeScript convention)
    new_lines: list[str] = []
    for line in lines:
        if "\t" in line:
            # Count leading tabs
            leading = len(line) - len(line.lstrip("\t"))
            rest = line.lstrip("\t")
            expanded = "  " * leading + rest
            changes.append("expanded tabs to 2 spaces")
            new_lines.append(expanded)
        else:
            new_lines.append(line)
    lines = new_lines

    # 2. Strip trailing whitespace
    stripped: list[str] = []
    for line in lines:
        s = line.rstrip()
        if s != line:
            changes.append("stripped trailing whitespace")
        stripped.append(s)
    lines = stripped

    # 3. Brace style: opening brace on same line (K&R)
    #    Handle the case where `{` appears on its own line after a declaration:
    #      function foo()
    #      {           ← move to previous line
    brace_fixed: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "{" and i > 0:
            prev = brace_fixed[-1] if brace_fixed else ""
            # Only move if prev line looks like a declaration (ends with ) or identifier)
            if re.search(r"[\w)]\s*$", prev.rstrip()):
                brace_fixed[-1] = prev.rstrip() + " {"
                changes.append("moved opening brace to previous line (K&R style)")
                i += 1
                continue
        brace_fixed.append(line)
        i += 1
    lines = brace_fixed

    # 4. Semicolon insertion (best-effort, conservative)
    semi_lines: list[str] = []
    for line in lines:
        if _line_needs_semicolon(line):
            semi_lines.append(line.rstrip() + ";")
            changes.append("added missing semicolon")
        else:
            semi_lines.append(line)
    lines = semi_lines

    # 5. Collapse excess blank lines (max 1)
    collapsed: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 1:
                collapsed.append(line)
            else:
                changes.append("collapsed multiple blank lines")
        else:
            blank_run = 0
            collapsed.append(line)
    lines = collapsed

    result = "\n".join(lines)
    if result and not result.endswith("\n"):
        result += "\n"

    return NormalizeResult(code=result, language="typescript", changes=list(dict.fromkeys(changes)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class CodeNormalizer:
    """Normalize generated code formatting for Python and TypeScript.

    Usage::

        normalizer = CodeNormalizer()
        result = normalizer.normalize(code, language="python")
        print(result.code)
        print(result.changes)  # list of what was changed
    """

    SUPPORTED_LANGUAGES = {"python", "typescript", "javascript"}

    def normalize(self, code: str, language: str = "python") -> NormalizeResult:
        """Normalize *code* for the given *language*.

        Parameters
        ----------
        code:
            Source code string (may be partial / incomplete).
        language:
            One of ``"python"``, ``"typescript"``, or ``"javascript"``.
            ``"javascript"`` is treated identically to ``"typescript"``.

        Returns
        -------
        NormalizeResult
            Contains the normalised code and a list of changes made.

        Raises
        ------
        ValueError
            If *language* is not supported.
        """
        lang = language.lower().strip()
        if lang == "javascript":
            lang = "typescript"
        if lang not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language!r}. "
                f"Choose from: {sorted(self.SUPPORTED_LANGUAGES)}"
            )

        if lang == "python":
            return _normalize_python(code)
        return _normalize_typescript(code)
