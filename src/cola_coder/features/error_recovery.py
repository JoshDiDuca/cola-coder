"""Error Recovery Analyzer: score how well generated code handles errors.

Checks (static analysis — no execution required):
  - try/except coverage: fraction of risky call sites wrapped in try/except
  - Bare except clauses (catches everything, considered poor practice)
  - Exception specificity: using broad Exception vs specific types
  - Validation of inputs: isinstance / if-guard patterns before risky ops
  - Raising without message (raise Exception() vs raise Exception("msg"))
  - Finally / context-manager usage for resource cleanup

For a TS dev: similar to analysing try/catch coverage and error typing
(using `unknown` vs typed catch blocks in TS 4.0+).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# AST node types that represent "risky" operations worth guarding
_RISKY_CALL_NAMES = frozenset(
    {
        "open",
        "read",
        "write",
        "connect",
        "execute",
        "run",
        "check_output",
        "urlopen",
        "request",
        "get",
        "post",
        "put",
        "delete",
        "load",
        "loads",
        "dump",
        "dumps",
        "decode",
        "encode",
        "int",
        "float",
        "json",
    }
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ErrorRecoveryReport:
    """Results of error recovery analysis."""

    total_functions: int
    functions_with_try: int
    bare_except_count: int
    broad_except_count: int  # catches Exception or BaseException
    specific_except_count: int
    raises_without_message: int
    has_finally_or_context_manager: bool
    input_validations: int  # isinstance / type-guard patterns

    @property
    def try_coverage(self) -> float:
        """Fraction of functions that contain at least one try block."""
        if self.total_functions == 0:
            return 0.0
        return self.functions_with_try / self.total_functions

    @property
    def score(self) -> float:
        """0.0–1.0 error handling quality score."""
        if self.total_functions == 0:
            return 0.5  # neutral for empty code

        base = self.try_coverage * 0.4
        # Penalise bare/broad excepts
        total_excepts = self.bare_except_count + self.broad_except_count + self.specific_except_count
        if total_excepts > 0:
            specificity_ratio = self.specific_except_count / total_excepts
            base += specificity_ratio * 0.25
        else:
            base += 0.0  # no try blocks at all

        # Penalise raises without messages
        if self.raises_without_message > 0:
            base -= min(0.1, self.raises_without_message * 0.05)

        # Bonus for finally / context managers
        if self.has_finally_or_context_manager:
            base += 0.1

        # Bonus for input validation
        if self.input_validations > 0:
            base += min(0.15, self.input_validations * 0.05)

        # Penalise bare excepts heavily
        if self.bare_except_count > 0:
            base -= min(0.2, self.bare_except_count * 0.1)

        return max(0.0, min(1.0, base))

    @property
    def issues(self) -> list[str]:
        out: list[str] = []
        if self.bare_except_count:
            out.append(f"{self.bare_except_count} bare except clause(s) (catch-all)")
        if self.broad_except_count:
            out.append(f"{self.broad_except_count} broad except Exception/BaseException")
        if self.raises_without_message:
            out.append(f"{self.raises_without_message} raise(s) without message")
        if self.try_coverage < 0.3 and self.total_functions > 1:
            out.append(f"Low try/except coverage ({self.try_coverage:.0%})")
        return out


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class ErrorRecoveryAnalyzer:
    """Statically analyse error handling patterns in Python source."""

    def analyze(self, source: str) -> ErrorRecoveryReport:
        """Return an :class:`ErrorRecoveryReport` for *source*."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return ErrorRecoveryReport(
                total_functions=0,
                functions_with_try=0,
                bare_except_count=0,
                broad_except_count=0,
                specific_except_count=0,
                raises_without_message=0,
                has_finally_or_context_manager=False,
                input_validations=0,
            )

        functions = [
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        functions_with_try = sum(
            1 for fn in functions if any(isinstance(n, ast.Try) for n in ast.walk(fn))
        )

        bare_except = 0
        broad_except = 0
        specific_except = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    bare_except += 1
                elif isinstance(node.type, ast.Name) and node.type.id in (
                    "Exception",
                    "BaseException",
                ):
                    broad_except += 1
                else:
                    specific_except += 1

        raises_no_msg = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and node.exc is not None:
                exc = node.exc
                # raise SomeError() with no args
                if isinstance(exc, ast.Call) and not exc.args and not exc.keywords:
                    raises_no_msg += 1

        has_finally = any(
            isinstance(n, ast.Try) and n.finalbody for n in ast.walk(tree)
        )
        has_with = any(isinstance(n, ast.With) for n in ast.walk(tree))

        input_validations = sum(
            1
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "isinstance"
        )

        return ErrorRecoveryReport(
            total_functions=len(functions),
            functions_with_try=functions_with_try,
            bare_except_count=bare_except,
            broad_except_count=broad_except,
            specific_except_count=specific_except,
            raises_without_message=raises_no_msg,
            has_finally_or_context_manager=has_finally or has_with,
            input_validations=input_validations,
        )
