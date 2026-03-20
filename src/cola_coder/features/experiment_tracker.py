"""
Experiment tracker for ML experiments.
Tracks hyperparams, metrics, and artifacts, saving to JSON files.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class Experiment:
    name: str
    hyperparams: dict
    metrics: dict  # name -> list of (value, step) tuples
    status: str  # 'running' | 'completed' | 'failed'
    start_time: str
    end_time: Optional[str] = None
    notes: str = ""
    exp_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class ExperimentTracker:
    def __init__(self, save_dir: str = None):
        if save_dir is None:
            self._tmp_dir = tempfile.mkdtemp()
            self.save_dir = self._tmp_dir
        else:
            self.save_dir = save_dir
            self._tmp_dir = None
        os.makedirs(self.save_dir, exist_ok=True)
        self._experiments: dict[str, Experiment] = {}

    def start(self, name: str, hyperparams: dict) -> str:
        exp_id = str(uuid.uuid4())
        exp = Experiment(
            name=name,
            hyperparams=dict(hyperparams),
            metrics={},
            status="running",
            start_time=datetime.now(timezone.utc).isoformat(),
            exp_id=exp_id,
        )
        self._experiments[exp_id] = exp
        return exp_id

    def log_metric(self, exp_id: str, name: str, value: float, step: int = None):
        exp = self._get_or_raise(exp_id)
        if name not in exp.metrics:
            exp.metrics[name] = []
        exp.metrics[name].append({"value": value, "step": step})

    def end(self, exp_id: str, status: str = "completed"):
        exp = self._get_or_raise(exp_id)
        exp.status = status
        exp.end_time = datetime.now(timezone.utc).isoformat()

    def get(self, exp_id: str) -> Experiment:
        return self._get_or_raise(exp_id)

    def list_experiments(self) -> list[Experiment]:
        return list(self._experiments.values())

    def best_experiment(self, metric: str, maximize: bool = False) -> Experiment:
        candidates = []
        for exp in self._experiments.values():
            if metric in exp.metrics and exp.metrics[metric]:
                last_value = exp.metrics[metric][-1]["value"]
                candidates.append((last_value, exp))
        if not candidates:
            raise ValueError(f"No experiments with metric '{metric}'")
        candidates.sort(key=lambda t: t[0], reverse=maximize)
        return candidates[0][1]

    def compare(self, exp_ids: list[str]) -> dict:
        result = {}
        for exp_id in exp_ids:
            exp = self._get_or_raise(exp_id)
            result[exp_id] = {
                "name": exp.name,
                "hyperparams": exp.hyperparams,
                "status": exp.status,
                "metrics": {
                    name: entries[-1]["value"] if entries else None
                    for name, entries in exp.metrics.items()
                },
                "start_time": exp.start_time,
                "end_time": exp.end_time,
            }
        return result

    def save(self, exp_id: str):
        exp = self._get_or_raise(exp_id)
        path = os.path.join(self.save_dir, f"{exp_id}.json")
        data = {
            "exp_id": exp.exp_id,
            "name": exp.name,
            "hyperparams": exp.hyperparams,
            "metrics": exp.metrics,
            "status": exp.status,
            "start_time": exp.start_time,
            "end_time": exp.end_time,
            "notes": exp.notes,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def summary(self) -> dict:
        total = len(self._experiments)
        by_status: dict[str, int] = {}
        for exp in self._experiments.values():
            by_status[exp.status] = by_status.get(exp.status, 0) + 1
        return {
            "total_experiments": total,
            "by_status": by_status,
            "experiment_names": [e.name for e in self._experiments.values()],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_raise(self, exp_id: str) -> Experiment:
        if exp_id not in self._experiments:
            raise KeyError(f"Experiment '{exp_id}' not found")
        return self._experiments[exp_id]
