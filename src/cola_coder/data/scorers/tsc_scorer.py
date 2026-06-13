"""TypeScript compiler scorer -- thin ScorerProtocol adapter around TscRunner.

Security: All tsc execution goes through TscRunner -> SandboxedRunner with
hardened tsconfig (plugins=[], types=[], typeRoots=[]).
"""

from __future__ import annotations

import shutil

from cola_coder.data.scorers.language_detect import is_typescript
from cola_coder.data.scorers.protocol import ScorerResult
from cola_coder.data.scorers.sandbox import SandboxedRunner
from cola_coder.data.scorers.utils import ScoreMapper
from cola_coder.reasoning.rewards.tsc_runner import SANDBOX_UNAVAILABLE_CODE, TscRunner


def _is_unverified(errors) -> bool:
    """True if tsc did not actually run (sandbox unavailable) — SEC-016."""
    return any(e.code == SANDBOX_UNAVAILABLE_CODE for e in errors)


# Score mapping: error count -> quality score
_TSC_SCORE_MAP = ScoreMapper([
    (0, 1.0),     # No errors = perfect
    (1, 0.8),     # 1 error = good
    (3, 0.6),     # 2-3 errors = decent
    (5, 0.4),     # 4-5 errors = average
    (10, 0.2),    # 6-10 errors = poor
])


class TscScorer:
    """Score TypeScript files using tsc --noEmit via SandboxedRunner.

    Delegates all tsc execution to TscRunner (SOLID Single Responsibility).
    Security: hardened tsconfig blocks plugin execution, @types loading, path traversal.
    """

    name: str = "tsc"

    def __init__(
        self,
        strict: bool = True,
        timeout: int = 10,
        cache_size: int = 256,
        runner: SandboxedRunner | None = None,
    ) -> None:
        self._runner = runner or SandboxedRunner(timeout=timeout)
        # TscRunner handles all tsc execution through the runner
        self._tsc = TscRunner(
            strict=strict,
            timeout=timeout,
            runner=self._runner,  # SECURITY: pass sandbox runner through
            cache_size=cache_size,
        )

    def score(self, code: str, metadata: dict[str, object] | None = None) -> ScorerResult:
        """Score a single TypeScript code sample."""
        if not is_typescript(code, metadata):
            return ScorerResult(
                score=0.5, scorer_name=self.name,
                details={"skipped": True, "reason": "not_typescript"},
            )

        errors = self._tsc.check(code)
        if _is_unverified(errors):
            # tsc could not run (sandbox unavailable) — do NOT score this as a
            # clean 0-error perfect; fail closed so unverified code can't enter
            # the corpus as high-quality (SEC-016).
            return ScorerResult(
                score=0.0, scorer_name=self.name,
                details={"not_verified": True, "reason": "sandbox_unavailable"},
            )
        num_errors = len(errors)
        has_syntax = any(e.code.startswith("TS1") for e in errors)

        score = _TSC_SCORE_MAP.map(num_errors)
        if has_syntax:
            score = min(score, 0.3)

        return ScorerResult(
            score=score,
            scorer_name=self.name,
            details={
                "num_errors": num_errors,
                "has_syntax_errors": has_syntax,
                "errors": [
                    {"code": e.code, "message": e.message, "line": e.line}
                    for e in errors[:5]
                ],
            },
        )

    def score_batch(self, items: list[tuple[str, dict[str, object] | None]]) -> list[ScorerResult]:
        """Score multiple samples -- delegates to TscRunner.check_batch()."""
        if not items:
            return []

        results: list[ScorerResult | None] = [None] * len(items)
        ts_codes: list[str] = []
        ts_indices: list[int] = []

        for i, (code, metadata) in enumerate(items):
            if not is_typescript(code, metadata):
                results[i] = ScorerResult(
                    score=0.5, scorer_name=self.name,
                    details={"skipped": True, "reason": "not_typescript"},
                )
            else:
                ts_codes.append(code)
                ts_indices.append(i)

        if not ts_codes:
            return [r for r in results if r is not None]

        # Delegate batch to TscRunner
        batch_results = self._tsc.check_batch(ts_codes)

        for batch_idx, orig_idx in enumerate(ts_indices):
            file_errors = batch_results.get(batch_idx, [])
            if _is_unverified(file_errors):
                results[orig_idx] = ScorerResult(
                    score=0.0, scorer_name=self.name,
                    details={"not_verified": True, "reason": "sandbox_unavailable"},
                )
                continue
            num_errors = len(file_errors)
            has_syntax = any(e.code.startswith("TS1") for e in file_errors)
            score = _TSC_SCORE_MAP.map(num_errors)
            if has_syntax:
                score = min(score, 0.3)

            results[orig_idx] = ScorerResult(
                score=score,
                scorer_name=self.name,
                details={
                    "num_errors": num_errors,
                    "has_syntax_errors": has_syntax,
                    "errors": [{"code": e.code, "message": e.message} for e in file_errors[:3]],
                },
            )

        return [r if r is not None else ScorerResult(score=0.5, scorer_name=self.name) for r in results]

    @staticmethod
    def is_available() -> bool:
        """Check if tsc is installed."""
        return shutil.which("tsc") is not None
