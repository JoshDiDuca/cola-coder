"""Heuristic scorer — adapts existing CodeScorer to ScorerProtocol.

Optimized for large batch scoring via multiprocessing (CPU-bound signals).
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

from cola_coder.data.scorers.protocol import ScorerResult

if TYPE_CHECKING:
    from cola_coder.features.code_scorer import CodeScorer

# Worker-local CodeScorer instance (avoid pickling)
_worker_scorer: CodeScorer | None = None


def _init_worker() -> None:
    """Initialize CodeScorer once per worker process."""
    global _worker_scorer
    from cola_coder.features.code_scorer import CodeScorer
    _worker_scorer = CodeScorer()


def _score_one(args: tuple[str, str]) -> tuple[float, str, dict[str, float]]:
    """Score a single item in a worker process.

    Args:
        args: (code, language) tuple.

    Returns:
        (overall, tier, breakdown) tuple (picklable primitives).
    """
    global _worker_scorer
    if _worker_scorer is None:
        _init_worker()
    assert _worker_scorer is not None

    code, language = args
    result = _worker_scorer.score(code, language)
    return result.overall, result.tier, dict(result.breakdown)


class HeuristicScorer:
    """Wraps the existing 13-signal CodeScorer as a ScorerProtocol implementor.

    Uses multiprocessing for batches > 500 items to parallelize
    CPU-bound signal computation across all available cores.
    """

    name: str = "heuristic"

    # Minimum batch size before engaging multiprocessing (overhead threshold)
    _MP_THRESHOLD: int = 1000

    def __init__(self, max_workers: int | None = None) -> None:
        self._scorer: CodeScorer | None = None
        # Default: min(4, cpu_count-1) — 4 workers is the IPC sweet spot on Windows
        cpu_count = os.cpu_count() or 4
        self._max_workers = max_workers or min(4, max(1, cpu_count - 1))

    def _get_scorer(self) -> CodeScorer:
        if self._scorer is None:
            from cola_coder.features.code_scorer import CodeScorer
            self._scorer = CodeScorer()
        return self._scorer

    def score(self, code: str, metadata: dict[str, object] | None = None) -> ScorerResult:
        language = ""
        if metadata:
            language = str(metadata.get("language", ""))

        scorer = self._get_scorer()
        result = scorer.score(code, language)

        return ScorerResult(
            score=result.overall,
            scorer_name=self.name,
            details={
                "tier": result.tier,
                "breakdown": result.breakdown,
            },
        )

    def score_batch(self, items: list[tuple[str, dict[str, object] | None]]) -> list[ScorerResult]:
        """Score a batch of items, using multiprocessing for large batches."""
        if not items:
            return []

        # Small batches: sequential (multiprocessing overhead not worth it)
        if len(items) <= self._MP_THRESHOLD:
            return [self.score(code, meta) for code, meta in items]

        # Large batches: parallelize across CPU cores
        return self._score_batch_parallel(items)

    def _score_batch_parallel(
        self,
        items: list[tuple[str, dict[str, object] | None]],
    ) -> list[ScorerResult]:
        """Score items in parallel using ProcessPoolExecutor."""
        # Prepare picklable args: (code, language) tuples
        args_list: list[tuple[str, str]] = []
        for code, metadata in items:
            language = ""
            if metadata:
                language = str(metadata.get("language", ""))
            args_list.append((code, language))

        results: list[ScorerResult] = []

        try:
            with ProcessPoolExecutor(
                max_workers=self._max_workers,
                initializer=_init_worker,
            ) as executor:
                # Use chunksize for efficient dispatch (reduce IPC overhead)
                chunksize = max(1, len(args_list) // (self._max_workers * 4))
                for overall, tier, breakdown in executor.map(
                    _score_one, args_list, chunksize=chunksize,
                ):
                    results.append(ScorerResult(
                        score=overall,
                        scorer_name=self.name,
                        details={"tier": tier, "breakdown": breakdown},
                    ))
        except (OSError, RuntimeError):
            # Fallback to sequential if multiprocessing fails
            # (e.g. on some Windows configs, or inside existing subprocesses)
            return [self.score(code, meta) for code, meta in items]

        return results

    @staticmethod
    def is_available() -> bool:
        try:
            from cola_coder.features.code_scorer import CodeScorer
            return True
        except ImportError:
            return False
