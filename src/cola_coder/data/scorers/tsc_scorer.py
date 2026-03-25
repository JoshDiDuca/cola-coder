"""TypeScript compiler scorer — wraps TypeCheckReward for data quality scoring."""

from __future__ import annotations

from cola_coder.data.scorers.protocol import ScorerResult
from cola_coder.data.scorers.sandbox import SandboxedRunner


class TscScorer:
    """Score TypeScript files using tsc --noEmit."""

    name: str = "tsc"

    def __init__(
        self,
        strict: bool = True,
        timeout: int = 10,
        cache_size: int = 256,
        runner: SandboxedRunner | None = None,
    ) -> None:
        self._strict = strict
        self._timeout = timeout
        self._cache_size = cache_size
        self._runner = runner or SandboxedRunner(timeout=timeout)
        self._checker: object | None = None  # Lazy init

    def _get_checker(self):
        """Lazy-init TypeCheckReward to avoid import cost at registration time."""
        if self._checker is None:
            from cola_coder.reasoning.rewards.type_check import TypeCheckReward
            self._checker = TypeCheckReward(
                strict=self._strict,
                timeout=self._timeout,
                cache_size=self._cache_size,
            )
        return self._checker

    def score(self, code: str, metadata: dict[str, object] | None = None) -> ScorerResult:
        """Score a single code sample.

        Only scores TypeScript/JavaScript files. Returns neutral score (0.5)
        for other languages.
        """
        # Language filter: only score TS/JS
        if not self._is_typescript(code, metadata):
            return ScorerResult(
                score=0.5, scorer_name=self.name,
                details={"skipped": True, "reason": "not_typescript"},
            )

        checker = self._get_checker()
        detail = checker.detailed_score(code)

        # Remap from TypeCheckReward's [-0.5, 1.0] range to [0.0, 1.0]
        raw = detail["score"]
        normalized = max(0.0, min(1.0, (raw + 0.5) / 1.5))

        return ScorerResult(
            score=normalized,
            scorer_name=self.name,
            details={
                "num_errors": detail.get("num_errors", 0),
                "error_codes": detail.get("error_codes", []),
                "has_syntax_errors": detail.get("has_syntax_errors", False),
                "raw_score": raw,
            },
        )

    def score_batch(self, items: list[tuple[str, dict[str, object] | None]]) -> list[ScorerResult]:
        """Score multiple samples sequentially (tsc already has MD5 caching)."""
        return [self.score(code, meta) for code, meta in items]

    @staticmethod
    def is_available() -> bool:
        """Check if tsc is installed."""
        try:
            from cola_coder.reasoning.rewards.type_check import TypeCheckReward
            return TypeCheckReward.is_available()
        except ImportError:
            return False

    @staticmethod
    def _is_typescript(code: str, metadata: dict[str, object] | None) -> bool:
        """Detect if code is TypeScript/JavaScript."""
        # Check metadata first
        if metadata:
            lang = str(metadata.get("language", "")).lower()
            if lang in ("typescript", "javascript", "ts", "js", "tsx", "jsx"):
                return True
            file_path = str(metadata.get("file_path", ""))
            if file_path:
                ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
                if ext in ("ts", "tsx", "js", "jsx", "mts", "cts", "mjs", "cjs"):
                    return True

        # Heuristic detection from code content
        ts_indicators = [
            "interface ", "type ", ": string", ": number", ": boolean",
            "=> {", "const ", "let ", "import ", "export ",
            "async function", "Promise<",
        ]
        matches = sum(1 for ind in ts_indicators if ind in code)
        return matches >= 3
