"""Batch type checking for GRPO groups.

Instead of spawning tsc per file, writes ALL generated files to a temp
directory with a tsconfig.json and runs tsc ONCE.

Speed: ~200ms for 16 files (vs ~800ms spawning 16 processes).

This is the preferred method during GRPO training where we generate
groups of 8-16 solutions and need to score them all.

For a TS dev: this is like having a monorepo with 16 files and running
tsc once -- much faster than running tsc 16 times separately.

All tsc execution is delegated to TscRunner, which runs through
SandboxedRunner with a hardened tsconfig.json.
"""

import logging

from .tsc_runner import TscRunner, TscError
from .type_check import SYNTAX_ERROR_RANGE

logger = logging.getLogger(__name__)


def _tsc_error_to_dict(error: TscError) -> dict:
    """Convert a TscError dataclass to the dict format expected by BatchTypeChecker.

    TscError.code is a string like "TS2322"; the dict format uses int 2322.
    """
    code_str = error.code
    if code_str.startswith("TS"):
        code_str = code_str[2:]
    try:
        code_int = int(code_str)
    except ValueError:
        code_int = 0

    return {
        "file": error.file,
        "line": error.line,
        "col": error.col,
        "code": code_int,
        "message": error.message,
    }


class BatchTypeChecker:
    """Fast batch type checking for GRPO groups.

    Writes all generated files to a temp directory with a tsconfig.json
    and runs tsc ONCE through TscRunner (SandboxedRunner), then parses
    per-file errors from the output.
    """

    def __init__(
        self,
        strict: bool = True,
        timeout: int = 30,
    ):
        """Initialize the batch type checker.

        Args:
            strict: Use --strict mode.
            timeout: Timeout in seconds for the batch tsc run.
        """
        self.strict = strict
        self.timeout = timeout
        self._tsc_runner = TscRunner(
            strict=strict,
            timeout=timeout,
        )
        # Keep _tsc_path for backward compat (used in detailed_batch guard)
        self._tsc_path = "tsc" if TscRunner.is_available() else None

    @staticmethod
    def is_available() -> bool:
        """Check if tsc is installed."""
        return TscRunner.is_available()

    def score_batch(self, codes: list[str]) -> list[float]:
        """Type-check a batch of code files simultaneously.

        Args:
            codes: List of TypeScript code strings.

        Returns:
            List of float scores (same length as codes).
        """
        results = self.detailed_batch(codes)
        return [r["score"] for r in results]

    def detailed_batch(self, codes: list[str]) -> list[dict]:
        """Return detailed diagnostics for each file in batch.

        Args:
            codes: List of TypeScript code strings.

        Returns:
            List of dicts with score, num_errors, errors, etc.
        """
        if self._tsc_path is None:
            logger.warning("tsc not available -- returning zero scores")
            return [
                {"score": 0.0, "num_errors": -1, "errors": [], "tsc_failed": True}
                for _ in codes
            ]

        if not codes:
            return []

        # Delegate to TscRunner for batch checking
        batch_errors = self._tsc_runner.check_batch(codes)

        # Check if any file had syntax errors -- if so, tsc may have
        # skipped type-checking other files in the batch. In that case,
        # fall back to individual checking for files with 0 reported errors.
        any_syntax_errors = any(
            any(self._error_code_int(e) in SYNTAX_ERROR_RANGE for e in errors)
            for errors in batch_errors.values()
            if errors
        )

        # Convert to result dicts
        results: list[dict] = []
        for i in range(len(codes)):
            tsc_errors = batch_errors.get(i, [])
            errors = [_tsc_error_to_dict(e) for e in tsc_errors]

            # If batch had syntax errors in other files and this file
            # reported 0 errors, re-check individually to be accurate
            if any_syntax_errors and not errors:
                individual_tsc_errors = self._tsc_runner.check(codes[i])
                errors = [_tsc_error_to_dict(e) for e in individual_tsc_errors]

            score = self._errors_to_score(errors)
            has_syntax = any(e["code"] in SYNTAX_ERROR_RANGE for e in errors)
            results.append({
                "score": score,
                "num_errors": len(errors),
                "errors": errors,
                "error_codes": [f"TS{e['code']}" for e in errors],
                "has_syntax_errors": has_syntax,
                "tsc_failed": False,
            })

        return results

    @staticmethod
    def _error_code_int(error: TscError) -> int:
        """Extract numeric error code from TscError."""
        code_str = error.code
        if code_str.startswith("TS"):
            code_str = code_str[2:]
        try:
            return int(code_str)
        except ValueError:
            return 0

    @staticmethod
    def _errors_to_score(errors: list[dict]) -> float:
        """Convert error list to a score (same logic as TypeCheckReward)."""
        if not errors:
            return 1.0

        has_syntax = any(e["code"] in SYNTAX_ERROR_RANGE for e in errors)
        if has_syntax:
            return -0.5

        n = len(errors)
        if n <= 2:
            return 0.7
        elif n <= 5:
            return 0.3
        else:
            return 0.0
