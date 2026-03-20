# 71 - Experiment Tracker (Local MLflow-Lite)

## Overview

A local experiment tracking system using SQLite as its backend. Automatically logs hyperparameters, per-step metrics, final metrics, and artifacts (checkpoints, configs) for every training run. Provides a CLI to list experiments, compare metrics across runs, show the best runs, and export to CSV. No external services required.

**Feature flag:** `config.experiment_tracker.enabled` (default: `true`)

---

## Motivation

After running 10 training experiments with different learning rates, batch sizes, and dataset versions, it becomes impossible to remember which run produced the best results. Without tracking, you're flying blind.

MLflow solves this but requires a server (or MLflow Tracking URI configuration), adds a heavy dependency, and is overkill for a personal training setup. This plan implements the 20% of MLflow's features that cover 80% of the use cases:

- **Run logging**: hyperparams, metrics, artifacts
- **Run comparison**: show all runs side-by-side sorted by key metric
- **Best run**: find the run with highest/lowest metric value
- **CSV export**: for further analysis in pandas/Excel

---

## Architecture / Design

### Database Schema

```sql
-- Experiments group related runs
CREATE TABLE experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    description TEXT
);

-- One run = one training job
CREATE TABLE runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    run_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',  -- running|completed|failed
    started_at TEXT NOT NULL,
    ended_at TEXT,
    git_hash TEXT,
    notes TEXT
);

-- Hyperparameters (string key-value)
CREATE TABLE params (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (run_id, key)
);

-- Metrics (numeric, optionally per-step)
CREATE TABLE metrics (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    key TEXT NOT NULL,
    value REAL NOT NULL,
    step INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT NOT NULL
);

-- Artifacts (file paths)
CREATE TABLE artifacts (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    artifact_type TEXT  -- checkpoint|config|eval_result|other
);

CREATE INDEX idx_metrics_run_key ON metrics(run_id, key);
CREATE INDEX idx_metrics_step ON metrics(step);
```

### Auto-logging Strategy

The trainer calls `tracker.log_param` / `tracker.log_metric` directly. No monkey-patching or import hooks. The auto-logging is "auto" in the sense that it's integrated into the trainer loop without requiring manual calls in user code.

---

## Implementation Steps

### Step 1: Tracker Core (`tracking/experiment_tracker.py`)

```python
import sqlite3
import json
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    description TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    run_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    started_at TEXT NOT NULL,
    ended_at TEXT,
    git_hash TEXT,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS params (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (run_id, key)
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    key TEXT NOT NULL,
    value REAL NOT NULL,
    step INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    artifact_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_metrics_run_key ON metrics(run_id, key);
"""

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _git_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None

class ExperimentTracker:
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(DB_SCHEMA)
        self._conn.commit()
        self._run_id: int | None = None

    def start_run(
        self,
        experiment_name: str,
        run_name: str = None,
        params: dict = None,
        notes: str = "",
    ) -> int:
        # Ensure experiment exists
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO experiments(name, created_at) VALUES (?, ?)",
            (experiment_name, _now()),
        )
        self._conn.execute(
            "UPDATE experiments SET created_at = created_at WHERE name = ?",
            (experiment_name,),
        )
        exp_id = self._conn.execute(
            "SELECT id FROM experiments WHERE name = ?", (experiment_name,)
        ).fetchone()["id"]

        if run_name is None:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM runs WHERE experiment_id = ?", (exp_id,)
            ).fetchone()[0]
            run_name = f"run-{count + 1:04d}"

        cur = self._conn.execute(
            """INSERT INTO runs(experiment_id, run_name, status, started_at, git_hash, notes)
               VALUES (?, ?, 'running', ?, ?, ?)""",
            (exp_id, run_name, _now(), _git_hash(), notes),
        )
        self._conn.commit()
        self._run_id = cur.lastrowid

        if params:
            self.log_params(params)

        return self._run_id

    def log_param(self, key: str, value: Any):
        assert self._run_id, "Call start_run() first"
        self._conn.execute(
            "INSERT OR REPLACE INTO params(run_id, key, value) VALUES (?, ?, ?)",
            (self._run_id, key, str(value)),
        )
        self._conn.commit()

    def log_params(self, params: dict):
        for k, v in params.items():
            self.log_param(k, v)

    def log_metric(self, key: str, value: float, step: int = 0):
        assert self._run_id, "Call start_run() first"
        self._conn.execute(
            "INSERT INTO metrics(run_id, key, value, step, timestamp) VALUES (?,?,?,?,?)",
            (self._run_id, key, float(value), step, _now()),
        )
        self._conn.commit()

    def log_metrics(self, metrics: dict, step: int = 0):
        for k, v in metrics.items():
            self.log_metric(k, v, step)

    def log_artifact(self, name: str, path: str, artifact_type: str = "other"):
        assert self._run_id, "Call start_run() first"
        self._conn.execute(
            "INSERT INTO artifacts(run_id, name, path, artifact_type) VALUES (?,?,?,?)",
            (self._run_id, name, str(path), artifact_type),
        )
        self._conn.commit()

    def end_run(self, status: str = "completed"):
        if self._run_id:
            self._conn.execute(
                "UPDATE runs SET status=?, ended_at=? WHERE id=?",
                (status, _now(), self._run_id),
            )
            self._conn.commit()
            self._run_id = None

    @contextmanager
    def run(self, experiment_name: str, run_name: str = None, params: dict = None):
        run_id = self.start_run(experiment_name, run_name, params)
        try:
            yield run_id
            self.end_run("completed")
        except Exception:
            self.end_run("failed")
            raise

    def get_runs(self, experiment_name: str = None) -> list[dict]:
        if experiment_name:
            rows = self._conn.execute(
                """SELECT r.*, e.name as experiment_name
                   FROM runs r JOIN experiments e ON r.experiment_id = e.id
                   WHERE e.name = ?
                   ORDER BY r.started_at DESC""",
                (experiment_name,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT r.*, e.name as experiment_name
                   FROM runs r JOIN experiments e ON r.experiment_id = e.id
                   ORDER BY r.started_at DESC""",
            ).fetchall()
        return [dict(r) for r in rows]

    def get_final_metrics(self, run_id: int) -> dict:
        """Get the last logged value for each metric in a run."""
        rows = self._conn.execute(
            """SELECT key, value FROM metrics
               WHERE run_id = ? AND (run_id, key, step) IN (
                   SELECT run_id, key, MAX(step) FROM metrics
                   WHERE run_id = ? GROUP BY key
               )""",
            (run_id, run_id),
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def export_csv(self, output_path: str):
        import csv
        runs = self.get_runs()
        all_param_keys: set = set()
        all_metric_keys: set = set()
        run_data = []

        for r in runs:
            params = dict(self._conn.execute(
                "SELECT key, value FROM params WHERE run_id=?", (r["id"],)
            ).fetchall())
            metrics = self.get_final_metrics(r["id"])
            all_param_keys.update(params.keys())
            all_metric_keys.update(metrics.keys())
            run_data.append({**r, "params": params, "metrics": metrics})

        with open(output_path, "w", newline="") as f:
            fieldnames = (
                ["run_id", "run_name", "experiment_name", "status", "started_at"]
                + [f"param_{k}" for k in sorted(all_param_keys)]
                + [f"metric_{k}" for k in sorted(all_metric_keys)]
            )
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in run_data:
                row = {
                    "run_id": r["id"],
                    "run_name": r["run_name"],
                    "experiment_name": r["experiment_name"],
                    "status": r["status"],
                    "started_at": r["started_at"],
                }
                for k in sorted(all_param_keys):
                    row[f"param_{k}"] = r["params"].get(k, "")
                for k in sorted(all_metric_keys):
                    row[f"metric_{k}"] = r["metrics"].get(k, "")
                writer.writerow(row)
```

### Step 2: Trainer Integration (`training/trainer.py`)

```python
# Auto-logging in trainer:

tracker = ExperimentTracker(db_path=config.experiment_tracker.db_path)

with tracker.run(
    experiment_name=config.experiment_name,
    params={
        "learning_rate": config.lr,
        "batch_size": config.batch_size,
        "max_seq_len": config.max_seq_len,
        "dataset_version": config.dataset_version,
        "n_layers": config.model.n_layers,
        "n_heads": config.model.n_heads,
        "d_model": config.model.d_model,
        "warmup_steps": config.warmup_steps,
        "grad_clip": config.grad_clip,
    }
) as run_id:
    for step, loss in training_loop():
        if step % config.log_interval == 0:
            tracker.log_metric("train_loss", loss, step=step)
        if step % config.eval_interval == 0:
            eval_result = run_eval(...)
            tracker.log_metrics(eval_result, step=step)
        if step % config.checkpoint_interval == 0:
            ckpt_path = save_checkpoint(step)
            tracker.log_artifact(f"checkpoint-{step}", ckpt_path, "checkpoint")
```

### Step 3: CLI Commands (`cli/tracking_cmd.py`)

```python
# cola-coder track list [--experiment NAME]
# cola-coder track compare --metric val_loss [--top N]
# cola-coder track show RUN_ID
# cola-coder track export --output runs.csv

def cmd_list(tracker: ExperimentTracker, experiment: str = None):
    from rich.table import Table
    from rich.console import Console
    runs = tracker.get_runs(experiment)
    console = Console()
    table = Table(title="Experiment Runs")
    table.add_column("Run ID", justify="right", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Experiment")
    table.add_column("Status")
    table.add_column("Started")
    for r in runs:
        color = {"completed": "green", "running": "yellow", "failed": "red"}.get(r["status"], "")
        table.add_row(
            str(r["id"]), r["run_name"], r["experiment_name"],
            f"[{color}]{r['status']}[/]", r["started_at"][:16]
        )
    console.print(table)

def cmd_compare(tracker: ExperimentTracker, metric: str, top_n: int = 10, higher_is_better: bool = False):
    from rich.table import Table
    runs = tracker.get_runs()
    data = []
    for r in runs:
        finals = tracker.get_final_metrics(r["id"])
        val = finals.get(metric)
        if val is not None:
            data.append((r, finals, val))

    data.sort(key=lambda x: -x[2] if higher_is_better else x[2])

    table = Table(title=f"Comparison by {metric}")
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Run")
    table.add_column(metric, justify="right")

    for i, (r, finals, val) in enumerate(data[:top_n]):
        table.add_row(str(i+1), r["run_name"], f"{val:.6f}")
    Console().print(table)
```

---

## Key Files to Modify

- `tracking/experiment_tracker.py` - New file: core tracker
- `cli/tracking_cmd.py` - New file: CLI commands
- `cli/main.py` - Register `track` subcommand
- `training/trainer.py` - Add auto-logging calls
- `config/training.yaml` - Add `experiment_tracker` section

---

## Testing Strategy

1. **CRUD test**: start run, log params/metrics/artifacts, end run, query all—assert data persists.
2. **Duplicate experiment test**: call `start_run` with same experiment name twice, assert no duplicate experiment rows.
3. **Final metrics test**: log metric at steps 0, 100, 200; call `get_final_metrics`, assert returns step 200 value.
4. **CSV export test**: create 3 runs with different params, export CSV, assert correct row count and headers.
5. **Context manager test**: use `with tracker.run(...)`, raise exception inside, assert run status is `"failed"`.
6. **Compare sort test**: create runs with known metric values, call `cmd_compare`, assert ordering.

---

## Performance Considerations

- SQLite is fast enough for logging one metric per step at 10 steps/sec. For very fast training loops (100+ steps/sec), batch metric inserts: accumulate 100 metrics in memory, flush every 10s.
- The `get_final_metrics` query uses a correlated subquery which is O(N) per run. For thousands of steps, add an index on `(run_id, key, step)`.
- Database file grows at ~1KB per 100 metrics. A 10k-step run with 5 metrics = 500KB. Multiple experiments over months → tens of MB. Still fast.

---

## Dependencies

No new Python dependencies. Uses only stdlib `sqlite3`, `csv`, `subprocess`.

---

## Estimated Complexity

**Low-Medium.** SQLite schema and CRUD are straightforward. The trainer integration requires care to not add latency to the training loop (batch inserts help). CLI formatting is routine Rich work. Estimated implementation time: 2-3 days.

---

## 2026 Best Practices

- **Immutable runs**: once a run is ended (`status=completed`), treat it as immutable. If you need to add notes after the fact, add a separate `run_notes` update path, not by mutating metrics.
- **Git hash logging**: always log the git commit hash at run start. This creates a hard link between the training code version and the results, enabling true reproducibility.
- **Auto-naming**: auto-generate run names (`run-0001`, `run-0002`) rather than requiring manual names. Prompting for a name adds friction and discourage logging.
- **DB in project root**: store `experiments.db` in the project root (or `data/`) so it's adjacent to checkpoints. Consider committing the DB to git for team use (it's small and append-only).
- **No remote sync required**: the entire value proposition is local operation. Don't add optional remote sync complexity in v1. If cloud sync is needed, export CSV and upload manually.
