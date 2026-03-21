"""Hyperparameter Logger (improvement #70).

Logs and diffs hyperparameters across training runs.
  - Records run configs with timestamps
  - Diffs which hyperparams changed between runs
  - Stores run history as JSON
  - Suggests values based on past runs (best-loss heuristic)

TypeScript analogy: like a version-control system for your training configs —
similar to how git diff shows what changed, but for training hyperparameters,
with a bonus feature that tells you which settings correlated with lower loss.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Feature toggle (project convention)
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if hyperparameter logging is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """One training run's hyperparameters and outcome."""

    run_id: str
    timestamp: float
    params: Dict[str, Any]
    final_loss: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HyperparamDiff:
    """Diff between two runs' hyperparameters."""

    run_a: str
    run_b: str
    changed: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)   # name -> (old, new)
    added: Dict[str, Any] = field(default_factory=dict)                  # in B but not A
    removed: Dict[str, Any] = field(default_factory=dict)               # in A but not B

    @property
    def has_changes(self) -> bool:
        return bool(self.changed or self.added or self.removed)

    @property
    def num_changes(self) -> int:
        return len(self.changed) + len(self.added) + len(self.removed)


@dataclass
class SuggestionReport:
    """Hyperparameter suggestions based on historical runs."""

    best_run_id: str
    best_loss: float
    suggestions: Dict[str, Any] = field(default_factory=dict)   # param -> suggested value
    rationale: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class HyperparamLogger:
    """Log, diff, and suggest hyperparameters across training runs.

    Supports both in-memory mode (default, for testing) and persistent
    JSON-file mode (pass log_path to __init__).

    Usage::

        logger = HyperparamLogger()
        run_id = logger.log_run({"lr": 1e-4, "batch_size": 32}, final_loss=2.3)
        run_id2 = logger.log_run({"lr": 5e-5, "batch_size": 32}, final_loss=2.1)
        diff = logger.diff(run_id, run_id2)
        suggestions = logger.suggest()
    """

    def __init__(self, log_path: Optional[str | Path] = None) -> None:
        self.log_path = Path(log_path) if log_path else None
        self._runs: List[RunRecord] = []
        if self.log_path and self.log_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_run(
        self,
        params: Dict[str, Any],
        run_id: Optional[str] = None,
        final_loss: Optional[float] = None,
        notes: str = "",
        timestamp: Optional[float] = None,
    ) -> str:
        """Log a training run's hyperparameters.

        Returns the run_id (auto-generated if not provided).
        """
        ts = timestamp if timestamp is not None else time.time()
        if run_id is None:
            run_id = f"run_{len(self._runs) + 1:04d}"
        rec = RunRecord(
            run_id=run_id,
            timestamp=ts,
            params=dict(params),
            final_loss=final_loss,
            notes=notes,
        )
        self._runs.append(rec)
        if self.log_path:
            self._save()
        return run_id

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        """Retrieve a run record by ID."""
        for r in self._runs:
            if r.run_id == run_id:
                return r
        return None

    def all_runs(self) -> List[RunRecord]:
        """Return all logged runs, oldest first."""
        return list(self._runs)

    def diff(self, run_id_a: str, run_id_b: str) -> HyperparamDiff:
        """Diff hyperparameters between two runs."""
        a = self.get_run(run_id_a)
        b = self.get_run(run_id_b)
        if a is None:
            raise KeyError(f"Run '{run_id_a}' not found")
        if b is None:
            raise KeyError(f"Run '{run_id_b}' not found")
        return _diff_params(a.params, b.params, run_id_a, run_id_b)

    def diff_latest(self) -> Optional[HyperparamDiff]:
        """Diff the last two runs (if at least two exist)."""
        if len(self._runs) < 2:
            return None
        a, b = self._runs[-2], self._runs[-1]
        return _diff_params(a.params, b.params, a.run_id, b.run_id)

    def best_run(self) -> Optional[RunRecord]:
        """Return the run with the lowest final_loss."""
        candidates = [r for r in self._runs if r.final_loss is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda r: r.final_loss)  # type: ignore[arg-type]

    def suggest(self) -> Optional[SuggestionReport]:
        """Suggest hyperparameter values based on best historical run."""
        best = self.best_run()
        if best is None:
            return None
        rationale = [
            f"Based on best run '{best.run_id}' with loss={best.final_loss:.4f}"
        ]
        suggestions = dict(best.params)
        return SuggestionReport(
            best_run_id=best.run_id,
            best_loss=best.final_loss,  # type: ignore[arg-type]
            suggestions=suggestions,
            rationale=rationale,
        )

    def history_for_param(self, param: str) -> List[Tuple[str, Any, Optional[float]]]:
        """Return [(run_id, value, final_loss)] for a specific hyperparameter."""
        return [
            (r.run_id, r.params.get(param), r.final_loss)
            for r in self._runs
            if param in r.params
        ]

    def export(self) -> List[Dict[str, Any]]:
        """Export all runs as a list of dicts."""
        return [r.to_dict() for r in self._runs]

    def reset(self) -> None:
        """Clear all run history."""
        self._runs.clear()
        if self.log_path and self.log_path.exists():
            self.log_path.unlink()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        assert self.log_path is not None
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as fh:
            json.dump(self.export(), fh, indent=2)

    def _load(self) -> None:
        assert self.log_path is not None
        with open(self.log_path) as fh:
            data = json.load(fh)
        self._runs = [
            RunRecord(
                run_id=d["run_id"],
                timestamp=d["timestamp"],
                params=d["params"],
                final_loss=d.get("final_loss"),
                notes=d.get("notes", ""),
            )
            for d in data
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _diff_params(
    params_a: Dict[str, Any],
    params_b: Dict[str, Any],
    id_a: str,
    id_b: str,
) -> HyperparamDiff:
    changed: Dict[str, Tuple[Any, Any]] = {}
    added: Dict[str, Any] = {}
    removed: Dict[str, Any] = {}

    for k, v in params_a.items():
        if k not in params_b:
            removed[k] = v
        elif params_b[k] != v:
            changed[k] = (v, params_b[k])

    for k, v in params_b.items():
        if k not in params_a:
            added[k] = v

    return HyperparamDiff(run_a=id_a, run_b=id_b, changed=changed, added=added, removed=removed)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def log_and_diff(
    runs: List[Dict[str, Any]],
    losses: Optional[List[Optional[float]]] = None,
) -> Tuple[HyperparamLogger, Optional[HyperparamDiff]]:
    """Log a sequence of runs and return (logger, diff_of_last_two)."""
    logger = HyperparamLogger()
    losses = losses or [None] * len(runs)
    for params, loss in zip(runs, losses):
        logger.log_run(params, final_loss=loss)
    diff = logger.diff_latest()
    return logger, diff
