"""
Recent runs history feature: tracks training/inference runs with metadata,
persists to JSON, and provides query/summary utilities.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class RunRecord:
    run_id: str
    name: str
    command: str
    status: str  # 'running' | 'completed' | 'failed'
    start_time: str
    end_time: Optional[str]
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


class RunHistory:
    def __init__(self, history_file: str = None):
        if history_file is None:
            tmp_dir = tempfile.gettempdir()
            history_file = os.path.join(tmp_dir, "cola_coder_run_history.json")
        self.history_file = history_file
        self._runs: dict[str, RunRecord] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def start_run(self, name: str, command: str, config: dict = None) -> str:
        run_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        record = RunRecord(
            run_id=run_id,
            name=name,
            command=command,
            status="running",
            start_time=now,
            end_time=None,
            config=config or {},
            metrics={},
        )
        self._runs[run_id] = record
        return run_id

    def end_run(self, run_id: str, status: str, metrics: dict = None):
        record = self._runs.get(run_id)
        if record is None:
            return
        record.status = status
        record.end_time = datetime.now(timezone.utc).isoformat()
        if metrics is not None:
            record.metrics = metrics

    def get_recent(self, n: int = 10) -> list[RunRecord]:
        sorted_runs = sorted(
            self._runs.values(),
            key=lambda r: r.start_time,
            reverse=True,
        )
        return sorted_runs[:n]

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        return self._runs.get(run_id)

    def format_history(self) -> str:
        recent = self.get_recent(n=len(self._runs))
        if not recent:
            return "No runs recorded."
        lines = []
        for r in recent:
            end = r.end_time or "—"
            lines.append(
                f"[{r.status.upper():^10}] {r.name} ({r.run_id[:8]})\n"
                f"  cmd:   {r.command}\n"
                f"  start: {r.start_time}  end: {end}\n"
                f"  config: {json.dumps(r.config)}\n"
                f"  metrics: {json.dumps(r.metrics)}"
            )
        return "\n\n".join(lines)

    def clear_old(self, days: int = 30):
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        to_remove = []
        for run_id, record in self._runs.items():
            try:
                start = datetime.fromisoformat(record.start_time)
                if start < cutoff:
                    to_remove.append(run_id)
            except (ValueError, TypeError):
                pass
        for run_id in to_remove:
            del self._runs[run_id]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        data = {run_id: asdict(record) for run_id, record in self._runs.items()}
        dir_path = os.path.dirname(self.history_file)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(self.history_file, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def load(self):
        if not os.path.exists(self.history_file):
            return
        with open(self.history_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._runs = {}
        for run_id, record_dict in data.items():
            self._runs[run_id] = RunRecord(**record_dict)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        total = len(self._runs)
        status_counts: dict[str, int] = {}
        for record in self._runs.values():
            status_counts[record.status] = status_counts.get(record.status, 0) + 1
        return {
            "total_runs": total,
            "status_counts": status_counts,
            "history_file": self.history_file,
        }
