"""
Synthetic bug injection for training bug-detection capabilities.

Injects realistic bugs into Python code and produces (buggy, clean, bug_info)
training triples that can be used to teach a model to detect and fix bugs.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

FEATURE_ENABLED: bool = True


def is_enabled() -> bool:
    """Return True if this feature is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Bug taxonomy
# ---------------------------------------------------------------------------


class BugType(str, Enum):
    OFF_BY_ONE = "off_by_one"
    WRONG_OPERATOR = "wrong_operator"
    MISSING_RETURN = "missing_return"
    WRONG_VARIABLE = "wrong_variable"
    TYPE_ERROR = "type_error"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class InjectedBug:
    bug_type: BugType
    original_line: str
    modified_line: str
    line_number: int          # 1-based
    description: str


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

# Patterns used by multiple injectors
_INT_LITERAL_RE = re.compile(r"\b(\d+)\b")
_COMPARISON_OPS = {"==", "!=", "<", ">", "<=", ">="}
_ARITHMETIC_OPS = {"+", "-", "*", "/", "//", "%", "**"}
_LOGICAL_OPS = {"and", "or"}

# Operator replacement tables
_CMP_SWAP: dict[str, list[str]] = {
    "==": ["!=", "<", ">"],
    "!=": ["==", "<", ">"],
    "<":  ["<=", ">", "=="],
    ">":  [">=", "<", "=="],
    "<=": ["<", ">", "!="],
    ">=": [">", "<", "!="],
}
_ARITH_SWAP: dict[str, list[str]] = {
    "+":  ["-", "*"],
    "-":  ["+", "*"],
    "*":  ["+", "//"],
    "/":  ["*", "+"],
    "//": ["/", "%"],
    "%":  ["+", "*"],
    "**": ["*", "+"],
}
_LOGIC_SWAP: dict[str, list[str]] = {
    "and": ["or"],
    "or":  ["and"],
}

# Type coercion mutations for type_error injection
_TYPE_MUTATIONS: list[tuple[re.Pattern, str]] = [
    # int literal → str literal
    (re.compile(r"\b(\d+)\b"), lambda m: f'"{m.group(1)}"'),  # type: ignore[arg-type]
    # str literal → int (only if it looks numeric)
    (re.compile(r'"(\d+)"'), lambda m: m.group(1)),           # type: ignore[arg-type]
    # True/False swap
    (re.compile(r"\bTrue\b"),  "False"),
    (re.compile(r"\bFalse\b"), "True"),
]


def _pick(seq: list) -> object:
    """Return a random element from seq."""
    return seq[random.randint(0, len(seq) - 1)]


def _lines(code: str) -> list[str]:
    """Split code preserving trailing newline behaviour."""
    return code.splitlines(keepends=True)


def _join(lines: list[str]) -> str:
    return "".join(lines)


# ---------------------------------------------------------------------------
# BugInjector
# ---------------------------------------------------------------------------


class BugInjector:
    """Inject synthetic bugs into Python source code."""

    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def available_bug_types(self) -> list[BugType]:
        """Return all supported bug types."""
        return list(BugType)

    def inject(
        self, code: str, n_bugs: int = 1
    ) -> tuple[str, list[InjectedBug]]:
        """
        Inject up to *n_bugs* bugs into *code*.

        Returns the modified code and a list of InjectedBug records.
        Multiple injections are applied sequentially so that line numbers
        reflect the state of the code after previous mutations.
        """
        injectors = [
            self.inject_off_by_one,
            self.inject_wrong_operator,
            self.inject_missing_return,
            self._inject_wrong_variable,
            self._inject_type_error,
        ]
        random.shuffle(injectors)

        bugs: list[InjectedBug] = []
        current_code = code

        for injector_fn in injectors:
            if len(bugs) >= n_bugs:
                break
            new_code, bug = injector_fn(current_code)
            if bug is not None:
                current_code = new_code
                bugs.append(bug)

        return current_code, bugs

    def inject_off_by_one(
        self, code: str
    ) -> tuple[str, Optional[InjectedBug]]:
        """
        Mutate an integer literal by ±1 on a line that contains a loop
        boundary, comparison, or slice-like pattern.
        """
        lines = _lines(code)
        candidates: list[int] = []

        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            # Target lines that look like loop/comparison boundaries
            if any(
                kw in stripped
                for kw in ("range(", "<=", ">=", "<", ">", "[", ":")
            ) and _INT_LITERAL_RE.search(line):
                candidates.append(idx)

        if not candidates:
            return code, None

        idx = _pick(candidates)  # type: ignore[assignment]
        original_line = lines[idx]

        # Find all integer literal positions
        matches = list(_INT_LITERAL_RE.finditer(original_line))
        if not matches:
            return code, None

        m = _pick(matches)  # type: ignore[assignment]
        value = int(m.group(1))
        delta = random.choice([-1, 1])
        new_value = value + delta

        modified_line = (
            original_line[: m.start(1)]
            + str(new_value)
            + original_line[m.end(1):]
        )
        lines[idx] = modified_line

        bug = InjectedBug(
            bug_type=BugType.OFF_BY_ONE,
            original_line=original_line,
            modified_line=modified_line,
            line_number=idx + 1,
            description=(
                f"Off-by-one: changed {value} → {new_value} "
                f"(delta {delta:+d})"
            ),
        )
        return _join(lines), bug

    def inject_wrong_operator(
        self, code: str
    ) -> tuple[str, Optional[InjectedBug]]:
        """
        Replace a comparison, arithmetic, or logical operator with a
        semantically different one.
        """
        lines = _lines(code)

        # Build a priority-ordered list of (line_idx, op, swap_table)
        Candidate = tuple  # (idx, op, swap_table)
        candidates: list[Candidate] = []

        for idx, line in enumerate(lines):
            # Check comparison operators (token-boundary match)
            for op, swaps in _CMP_SWAP.items():
                # Use word-boundary-aware regex for multi-char ops
                pattern = re.escape(op)
                if re.search(pattern, line):
                    candidates.append((idx, op, swaps))

            # Check arithmetic operators
            for op, swaps in _ARITH_SWAP.items():
                if op in line:
                    candidates.append((idx, op, swaps))

            # Check logical operators
            for op, swaps in _LOGIC_SWAP.items():
                if re.search(rf"\b{op}\b", line):
                    candidates.append((idx, op, swaps))

        if not candidates:
            return code, None

        idx, op, swaps = _pick(candidates)  # type: ignore[misc]
        original_line = lines[idx]
        replacement = _pick(swaps)  # type: ignore[assignment]

        # Replace only the first occurrence to keep the mutation minimal
        if re.search(rf"\b{re.escape(op)}\b", original_line):
            modified_line = re.sub(
                rf"\b{re.escape(op)}\b", replacement, original_line, count=1
            )
        else:
            modified_line = original_line.replace(op, replacement, 1)

        lines[idx] = modified_line

        bug = InjectedBug(
            bug_type=BugType.WRONG_OPERATOR,
            original_line=original_line,
            modified_line=modified_line,
            line_number=idx + 1,
            description=f"Wrong operator: '{op}' replaced with '{replacement}'",
        )
        return _join(lines), bug

    def inject_missing_return(
        self, code: str
    ) -> tuple[str, Optional[InjectedBug]]:
        """
        Remove a `return` statement from a function that has one.

        The removed statement is replaced with a comment so that
        indentation and surrounding structure are preserved.
        """
        lines = _lines(code)
        return_indices: list[int] = []

        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            # Only remove explicit value returns, not bare `return`
            if re.match(r"return\s+\S", stripped):
                return_indices.append(idx)

        if not return_indices:
            return code, None

        idx = _pick(return_indices)  # type: ignore[assignment]
        original_line = lines[idx]
        indent = len(original_line) - len(original_line.lstrip())
        modified_line = " " * indent + "# return statement removed\n"
        lines[idx] = modified_line

        bug = InjectedBug(
            bug_type=BugType.MISSING_RETURN,
            original_line=original_line,
            modified_line=modified_line,
            line_number=idx + 1,
            description="Missing return: return statement commented out",
        )
        return _join(lines), bug

    def _inject_wrong_variable(
        self, code: str
    ) -> tuple[str, Optional[InjectedBug]]:
        """
        Replace a variable reference with a different variable name that
        appears in the same scope.
        """
        lines = _lines(code)

        # Collect identifier names from the entire snippet
        all_names = re.findall(r"\b([a-z_][a-zA-Z0-9_]*)\b", code)
        # Filter out Python keywords and builtins
        _KEYWORDS = {
            "and", "as", "assert", "async", "await", "break", "class",
            "continue", "def", "del", "elif", "else", "except", "finally",
            "for", "from", "global", "if", "import", "in", "is", "lambda",
            "nonlocal", "not", "or", "pass", "raise", "return", "try",
            "while", "with", "yield", "none", "true", "false",
            "print", "len", "range", "int", "str", "float", "list",
            "dict", "set", "tuple", "type", "None", "True", "False",
        }
        names = [n for n in set(all_names) if n not in _KEYWORDS and len(n) > 1]

        if len(names) < 2:
            return code, None

        # Find candidate lines: assignment targets or expressions
        candidates: list[int] = []
        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(("def ", "class ", "#", "import", "from")):
                continue
            if any(n in line for n in names):
                candidates.append(idx)

        if not candidates:
            return code, None

        idx = _pick(candidates)  # type: ignore[assignment]
        original_line = lines[idx]

        # Find variable names present on this line
        line_names = [
            n for n in names if re.search(rf"\b{re.escape(n)}\b", original_line)
        ]
        if not line_names:
            return code, None

        target = _pick(line_names)  # type: ignore[assignment]
        replacements = [n for n in names if n != target]
        if not replacements:
            return code, None

        replacement = _pick(replacements)  # type: ignore[assignment]
        modified_line = re.sub(
            rf"\b{re.escape(target)}\b", replacement, original_line, count=1
        )

        if modified_line == original_line:
            return code, None

        lines[idx] = modified_line

        bug = InjectedBug(
            bug_type=BugType.WRONG_VARIABLE,
            original_line=original_line,
            modified_line=modified_line,
            line_number=idx + 1,
            description=(
                f"Wrong variable: '{target}' replaced with '{replacement}'"
            ),
        )
        return _join(lines), bug

    def _inject_type_error(
        self, code: str
    ) -> tuple[str, Optional[InjectedBug]]:
        """
        Introduce a type mismatch, e.g. wrap a numeric literal in quotes
        or swap True/False.
        """
        lines = _lines(code)
        candidates: list[tuple[int, re.Pattern, object]] = []

        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(("#", "def ", "class ")):
                continue
            for pattern, replacement in _TYPE_MUTATIONS:
                if pattern.search(line):
                    candidates.append((idx, pattern, replacement))

        if not candidates:
            return code, None

        idx, pattern, replacement = _pick(candidates)  # type: ignore[misc]
        original_line = lines[idx]

        if callable(replacement):
            modified_line = pattern.sub(replacement, original_line, count=1)
        else:
            modified_line = pattern.sub(replacement, original_line, count=1)  # type: ignore[arg-type]

        if modified_line == original_line:
            return code, None

        lines[idx] = modified_line

        bug = InjectedBug(
            bug_type=BugType.TYPE_ERROR,
            original_line=original_line,
            modified_line=modified_line,
            line_number=idx + 1,
            description="Type error: value literal mutated to incompatible type",
        )
        return _join(lines), bug

    # ------------------------------------------------------------------
    # Training data helpers
    # ------------------------------------------------------------------

    def create_training_pair(self, code: str) -> dict:
        """
        Create a (clean, buggy, bug_info) training triple.

        Returns a dict with keys:
          - ``clean``   : original source code
          - ``buggy``   : source code with one injected bug
          - ``modified``: alias for ``buggy`` (convenience)
          - ``bugs``    : list of InjectedBug records (as dicts)
          - ``n_bugs``  : number of bugs injected
          - ``has_bug`` : bool flag
        """
        buggy_code, bugs = self.inject(code, n_bugs=1)
        bug_dicts = [
            {
                "bug_type": b.bug_type.value,
                "original_line": b.original_line,
                "modified_line": b.modified_line,
                "line_number": b.line_number,
                "description": b.description,
            }
            for b in bugs
        ]
        return {
            "clean": code,
            "buggy": buggy_code,
            "modified": buggy_code,   # alias
            "bugs": bug_dicts,
            "n_bugs": len(bugs),
            "has_bug": len(bugs) > 0,
        }
