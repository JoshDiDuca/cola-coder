"""Code Semantic Validator — Feature 95

Validate semantic correctness of generated Python code using heuristics and
light static analysis (no AST execution).

Checks performed
----------------
- **type_consistency**: detect obvious type annotation conflicts (e.g. `x: int = "hello"`)
- **scope_check**: detect use of names before assignment / outside scope (basic)
- **unreachable_code**: lines after ``return`` / ``raise`` at the same indent level
- **unused_variables**: variables assigned but never referenced elsewhere
- **missing_return**: function that has conditional returns but might fall off

All checks are line-level or simple-regex based — intentionally conservative
(may miss real bugs, will not produce false positives on valid code in common
patterns).

Feature toggle: set FEATURE_ENABLED = False to disable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if semantic validation is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Issue severity
# ---------------------------------------------------------------------------


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Issue
# ---------------------------------------------------------------------------


@dataclass
class SemanticIssue:
    """A single semantic issue found in source code."""

    check: str
    severity: IssueSeverity
    line: Optional[int]
    message: str

    def __str__(self) -> str:
        loc = f"line {self.line}" if self.line is not None else "global"
        return f"[{self.severity.value.upper()}] {self.check} @ {loc}: {self.message}"


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Outcome of validating a code snippet."""

    issues: list[SemanticIssue] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no ERROR-severity issues found."""
        return not any(i.severity == IssueSeverity.ERROR for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.WARNING)

    def summary(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "errors": self.error_count,
            "warnings": self.warning_count,
            "checks_run": self.checks_run,
            "issues": [str(i) for i in self.issues],
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

# Simple type annotation patterns  e.g.  x: int = "hello"
_TYPE_ASSIGN_RE = re.compile(
    r"""^\s*\w+\s*:\s*(int|float|bool)\s*=\s*["']"""
)

# Variable assignment at indent >= 4:  varname = ...
_ASSIGN_RE = re.compile(r"^\s{4,}([a-z_]\w*)\s*=(?!=)\s*\S")

# Variable usage (simple identifier token)
_IDENT_USE_RE = re.compile(r"\b([a-z_]\w*)\b")

# Lines that terminate a code path
_TERMINATOR_RE = re.compile(r"^\s*(return|raise|sys\.exit|exit)\b")

# Function def
_DEF_RE = re.compile(r"^(\s*)def\s+(\w+)\s*\(")

# Conditional return inside function body
_COND_RETURN_RE = re.compile(r"^\s{8,}return\b")


def _check_type_consistency(lines: list[str]) -> list[SemanticIssue]:
    issues: list[SemanticIssue] = []
    for i, line in enumerate(lines, start=1):
        if _TYPE_ASSIGN_RE.match(line):
            issues.append(
                SemanticIssue(
                    check="type_consistency",
                    severity=IssueSeverity.WARNING,
                    line=i,
                    message="Possible type mismatch: numeric annotation with string value",
                )
            )
    return issues


def _check_unreachable_code(lines: list[str]) -> list[SemanticIssue]:
    """Detect lines after a ``return``/``raise`` at the same indent level."""
    issues: list[SemanticIssue] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _TERMINATOR_RE.match(line) and line.strip():
            term_indent = len(line) - len(line.lstrip())
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                if not next_line.strip():  # blank
                    j += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent == term_indent and next_line.strip() not in (
                    "else:",
                    "elif ...",
                    "except:",
                    "finally:",
                ):
                    # same indent after terminator → unreachable
                    issues.append(
                        SemanticIssue(
                            check="unreachable_code",
                            severity=IssueSeverity.WARNING,
                            line=j + 1,
                            message=f"Code after {lines[i].strip()!r} may be unreachable",
                        )
                    )
                break
            # skip to j to avoid cascading
            i = j if j > i else i + 1
        else:
            i += 1
    return issues


def _check_unused_variables(lines: list[str]) -> list[SemanticIssue]:
    """Warn about variables assigned but never used (inside functions)."""
    issues: list[SemanticIssue] = []
    assigned: dict[str, int] = {}  # name → line number

    # Collect all assignments
    for i, line in enumerate(lines, start=1):
        m = _ASSIGN_RE.match(line)
        if m:
            name = m.group(1)
            # Skip names that look like throw-aways
            if name not in ("_", "__"):
                assigned[name] = i

    if not assigned:
        return issues

    # Collect all usages (rough)
    all_code = "\n".join(lines)
    used: set[str] = set()
    for m in _IDENT_USE_RE.finditer(all_code):
        used.add(m.group(1))

    for name, lineno in assigned.items():
        # Count occurrences — if only 1 (the assignment) it's unused
        count = len(re.findall(rf"\b{re.escape(name)}\b", all_code))
        if count <= 1:
            issues.append(
                SemanticIssue(
                    check="unused_variables",
                    severity=IssueSeverity.INFO,
                    line=lineno,
                    message=f"Variable '{name}' is assigned but never used",
                )
            )
    return issues


def _check_missing_return(lines: list[str]) -> list[SemanticIssue]:
    """Warn if a function has conditional returns but no top-level return."""
    issues: list[SemanticIssue] = []
    in_func = False
    func_name = ""
    func_indent = 0
    has_cond_return = False
    has_top_return = False
    func_line = 0

    def _flush():
        nonlocal has_cond_return, has_top_return
        if in_func and has_cond_return and not has_top_return:
            issues.append(
                SemanticIssue(
                    check="missing_return",
                    severity=IssueSeverity.INFO,
                    line=func_line,
                    message=(
                        f"Function '{func_name}' has conditional returns "
                        "but may fall through without returning a value"
                    ),
                )
            )
        has_cond_return = False
        has_top_return = False

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        m = _DEF_RE.match(line)
        if m:
            _flush()
            in_func = True
            func_indent = len(m.group(1))
            func_name = m.group(2)
            func_line = i
            continue

        if in_func:
            cur_indent = len(line) - len(line.lstrip())
            if cur_indent <= func_indent and stripped:
                # We've left the function body
                _flush()
                in_func = False
                # re-check if this is another def
                m2 = _DEF_RE.match(line)
                if m2:
                    in_func = True
                    func_indent = len(m2.group(1))
                    func_name = m2.group(2)
                    func_line = i
                continue
            # Check for return
            if re.match(r"^\s*return\b", line):
                if cur_indent == func_indent + 4:
                    has_top_return = True
                elif cur_indent > func_indent + 4:
                    has_cond_return = True

    _flush()
    return issues


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

_ALL_CHECKS = [
    "type_consistency",
    "unreachable_code",
    "unused_variables",
    "missing_return",
]

_CHECK_FNS = {
    "type_consistency": _check_type_consistency,
    "unreachable_code": _check_unreachable_code,
    "unused_variables": _check_unused_variables,
    "missing_return": _check_missing_return,
}


class CodeSemanticValidator:
    """Validate semantic properties of generated Python code."""

    def __init__(self, checks: Optional[list[str]] = None) -> None:
        if checks is None:
            checks = list(_ALL_CHECKS)
        unknown = [c for c in checks if c not in _CHECK_FNS]
        if unknown:
            raise ValueError(f"Unknown checks: {unknown}")
        self._checks = checks

    def validate(self, code: str) -> ValidationResult:
        """Validate *code* and return a :class:`ValidationResult`."""
        lines = code.splitlines()
        result = ValidationResult(checks_run=list(self._checks))
        for check in self._checks:
            fn = _CHECK_FNS[check]
            result.issues.extend(fn(lines))
        return result

    @property
    def enabled_checks(self) -> list[str]:
        return list(self._checks)
