"""Data Augmentation Engine — Feature 93

Code-specific data augmentation strategies for training data diversity.

Augmentation types
------------------
- **variable_rename**: Rename local variables to generic names (x0, x1 …)
  while preserving function/class names and Python keywords.
- **comment_remove**: Strip ``# …`` inline comments and standalone comment lines.
- **comment_add**: Insert a simple placeholder comment before function defs.
- **whitespace_vary**: Randomly add or remove blank lines between top-level blocks.
- **import_reorder**: Sort import lines alphabetically (stdlib pattern only).
- **dead_code_insert**: Insert an unreachable ``pass`` block after a random function.

All transforms are line-level / regex-based — no AST required — so they
work on partial / malformed snippets too.

Feature toggle: set FEATURE_ENABLED = False to disable.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if data augmentation is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class AugmentResult:
    """Outcome of a single augmentation pass."""

    code: str
    strategy: str
    changed: bool
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

# Python keywords + builtins we must never rename
_PY_KEYWORDS = frozenset(
    """False None True and as assert async await break class continue def del
    elif else except finally for from global if import in is lambda nonlocal
    not or pass raise return try while with yield""".split()
)

_SIMPLE_BUILTINS = frozenset(
    "print len range int str float list dict tuple set bool type object".split()
)

_PROTECTED = _PY_KEYWORDS | _SIMPLE_BUILTINS


def _strategy_variable_rename(code: str, rng: random.Random) -> AugmentResult:
    """Replace simple local variable names (1-2 lowercase chars) with x0, x1 …"""
    lines = code.splitlines(keepends=True)
    # Collect single-letter / two-letter lowercase identifiers used as lvalues
    lvalue_re = re.compile(r"^\s{4,}([a-z]{1,2})\s*=\s*")
    candidates: dict[str, str] = {}
    counter = 0
    out_lines: list[str] = []
    changed = False
    for line in lines:
        m = lvalue_re.match(line)
        if m:
            orig = m.group(1)
            if orig not in _PROTECTED and orig not in candidates:
                candidates[orig] = f"x{counter}"
                counter += 1
        out_lines.append(line)

    if not candidates:
        return AugmentResult(code=code, strategy="variable_rename", changed=False)

    # Apply renames — use word-boundary replacement
    new_code = code
    for orig, new in candidates.items():
        new_code = re.sub(rf"\b{re.escape(orig)}\b", new, new_code)
    changed = new_code != code
    return AugmentResult(
        code=new_code,
        strategy="variable_rename",
        changed=changed,
        notes=[f"renamed {len(candidates)} variable(s)"],
    )


def _strategy_comment_remove(code: str, rng: random.Random) -> AugmentResult:
    """Remove all ``# …`` comments."""
    lines = code.splitlines(keepends=True)
    out: list[str] = []
    removed = 0
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            removed += 1
            # Keep blank line to preserve structure
            out.append("\n")
        else:
            # Remove inline comments (outside strings — best-effort)
            cleaned = re.sub(r"\s*#[^\n]*", "", line.rstrip())
            if cleaned != line.rstrip():
                removed += 1
                out.append(cleaned + "\n" if line.endswith("\n") else cleaned)
            else:
                out.append(line)
    new_code = "".join(out)
    return AugmentResult(
        code=new_code,
        strategy="comment_remove",
        changed=removed > 0,
        notes=[f"removed {removed} comment(s)"] if removed else [],
    )


def _strategy_comment_add(code: str, rng: random.Random) -> AugmentResult:
    """Insert ``# Implementation`` comment before ``def`` lines."""
    lines = code.splitlines(keepends=True)
    out: list[str] = []
    added = 0
    for line in lines:
        stripped = line.lstrip()
        if re.match(r"def\s+\w+", stripped):
            indent = line[: len(line) - len(stripped)]
            out.append(f"{indent}# Implementation\n")
            added += 1
        out.append(line)
    new_code = "".join(out)
    return AugmentResult(
        code=new_code,
        strategy="comment_add",
        changed=added > 0,
        notes=[f"inserted {added} comment(s)"] if added else [],
    )


def _strategy_whitespace_vary(code: str, rng: random.Random) -> AugmentResult:
    """Randomly add or remove blank lines between top-level blocks."""
    lines = code.splitlines(keepends=True)
    out: list[str] = []
    changed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        # Between two non-blank lines that are at column-0, vary spacing
        if (
            i + 1 < len(lines)
            and line.strip()
            and lines[i + 1].strip()
            and not line[0:1] == " "
            and not lines[i + 1][0:1] == " "
        ):
            action = rng.choice(["add", "remove", "keep"])
            if action == "add":
                out.append("\n")
                changed = True
            # "remove" would affect the next line if it's blank; handle below
        # Skip consecutive blank lines (collapse to one)
        if line == "\n" and i + 1 < len(lines) and lines[i + 1] == "\n":
            changed = True
            i += 1  # skip duplicate blank
        i += 1
    new_code = "".join(out)
    return AugmentResult(
        code=new_code,
        strategy="whitespace_vary",
        changed=changed or new_code != code,
    )


def _strategy_import_reorder(code: str, rng: random.Random) -> AugmentResult:
    """Sort consecutive import lines alphabetically."""
    lines = code.splitlines(keepends=True)
    out: list[str] = []
    changed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^(import|from)\s", line):
            # Collect the consecutive import block
            block: list[str] = [line]
            j = i + 1
            while j < len(lines) and re.match(r"^(import|from)\s", lines[j]):
                block.append(lines[j])
                j += 1
            sorted_block = sorted(block)
            if sorted_block != block:
                changed = True
            out.extend(sorted_block)
            i = j
        else:
            out.append(line)
            i += 1
    new_code = "".join(out)
    return AugmentResult(
        code=new_code,
        strategy="import_reorder",
        changed=changed,
        notes=["imports sorted"] if changed else [],
    )


def _strategy_dead_code_insert(code: str, rng: random.Random) -> AugmentResult:
    """Insert an unreachable ``if False: pass`` block after a random function def."""
    lines = code.splitlines(keepends=True)
    def_indices = [
        i for i, line in enumerate(lines) if re.match(r"\s*def\s+\w+", line)
    ]
    if not def_indices:
        return AugmentResult(code=code, strategy="dead_code_insert", changed=False)

    target = rng.choice(def_indices)
    indent = re.match(r"(\s*)", lines[target]).group(1)  # type: ignore[union-attr]
    dead_block = f"{indent}    if False:\n{indent}        pass\n"

    # Find end of function signature line (may be multiline) and insert after
    insert_at = target + 1
    # Walk forward until we hit the body (a line with deeper indent)
    while insert_at < len(lines) and not lines[insert_at].strip():
        insert_at += 1

    new_lines = lines[:insert_at] + [dead_block] + lines[insert_at:]
    new_code = "".join(new_lines)
    return AugmentResult(
        code=new_code,
        strategy="dead_code_insert",
        changed=True,
        notes=["inserted dead code block"],
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_STRATEGIES: dict[str, Callable[[str, random.Random], AugmentResult]] = {
    "variable_rename": _strategy_variable_rename,
    "comment_remove": _strategy_comment_remove,
    "comment_add": _strategy_comment_add,
    "whitespace_vary": _strategy_whitespace_vary,
    "import_reorder": _strategy_import_reorder,
    "dead_code_insert": _strategy_dead_code_insert,
}

AVAILABLE_STRATEGIES: list[str] = list(_STRATEGIES.keys())


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DataAugmentationEngine:
    """Apply code-specific augmentation strategies to training samples."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Single-strategy augment
    # ------------------------------------------------------------------

    def augment(self, code: str, strategy: str) -> AugmentResult:
        """Apply a named strategy to *code*.

        Parameters
        ----------
        code:
            Source code to augment.
        strategy:
            One of :data:`AVAILABLE_STRATEGIES`.

        Returns
        -------
        AugmentResult
        """
        if strategy not in _STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {list(_STRATEGIES.keys())}"
            )
        fn = _STRATEGIES[strategy]
        return fn(code, self._rng)

    # ------------------------------------------------------------------
    # Multi-strategy / random
    # ------------------------------------------------------------------

    def augment_random(
        self,
        code: str,
        strategies: Optional[list[str]] = None,
    ) -> AugmentResult:
        """Apply a randomly chosen strategy from *strategies*.

        Parameters
        ----------
        code:
            Source code to augment.
        strategies:
            Subset of :data:`AVAILABLE_STRATEGIES`.  Defaults to all.
        """
        pool = strategies or AVAILABLE_STRATEGIES
        chosen = self._rng.choice(pool)
        return self.augment(code, chosen)

    def augment_pipeline(
        self, code: str, strategies: list[str]
    ) -> list[AugmentResult]:
        """Apply each strategy in *strategies* in sequence, chaining outputs.

        Returns a list of results — one per strategy — so callers can inspect
        intermediate states.
        """
        results: list[AugmentResult] = []
        current = code
        for s in strategies:
            result = self.augment(current, s)
            results.append(result)
            current = result.code
        return results

    def generate_variants(
        self,
        code: str,
        n: int = 3,
        strategies: Optional[list[str]] = None,
    ) -> list[AugmentResult]:
        """Generate *n* augmented variants of *code*, each using a random strategy.

        Parameters
        ----------
        code:
            Original source.
        n:
            Number of variants to produce.
        strategies:
            Strategy pool.  Defaults to all.
        """
        return [self.augment_random(code, strategies) for _ in range(n)]
