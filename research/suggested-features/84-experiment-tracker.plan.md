# Feature 84: Experiment Tracker

**Status:** Proposed
**CLI Flag:** `--track-experiments` / config key `tracking.enabled: true`
**Complexity:** Medium

---

## Overview

A lightweight local experiment tracking system ("MLflow-lite") that auto-instruments the Cola-Coder trainer to record hyperparameters, per-step metrics, and final results — all in a local SQLite database. Zero manual instrumentation: the tracker hooks into the existing trainer via a context manager. A CLI (`cola-coder-experiments` or `python scripts/experiments.py`) provides commands to list runs, compare metrics in Rich tables, plot training curves in-terminal with `plotext`, export to CSV, and tag/annotate runs.

```
$ python scripts/experiments.py list

┌─────────────────────────────────────────────────────────────────┐
│  Experiments (8 runs)                                           │
├──────┬───────────┬──────────┬──────────┬───────┬───────────────┤
│  ID  │ Config    │ Steps    │ Best Loss│ ppl   │ Started       │
├──────┼───────────┼──────────┼──────────┼───────┼───────────────┤
│  008 │ tiny      │   5000   │  2.814   │ 16.7  │  2h ago       │
│  007 │ tiny-lr   │   5000   │  3.021   │ 20.5  │  6h ago  [★]  │
│  006 │ small     │   2000   │  4.112   │ 61.2  │  1d ago       │
╰──────┴───────────┴──────────┴──────────┴───────┴───────────────╯
```

---

## Motivation

Training runs without tracking are throwaway experiments. After even 5 runs it becomes impossible to remember which config produced which loss, which learning rate worked, or whether a recent architectural change helped or hurt. MLflow and Weights & Biases solve this but require external processes, accounts, or heavy dependencies.

This tracker is:
- **Zero-dependency beyond SQLite** (stdlib) — no server process to manage.
- **Auto-instrumented** — a single `with tracker:` wraps the trainer loop. No per-step manual logging.
- **Persistent across sessions** — SQLite file lives in `experiments/runs.db`.
- **CLI-first** — list, compare, plot, and export from the terminal without leaving Cola-Coder.

The TS analogy: this is like a local Prisma-backed analytics system for training runs, with the trainer acting as the Prisma Client that writes records automatically.

---

## Architecture / Design

```
ExperimentTracker
  │
  ├── RunContext (context manager)
  │     ├── on enter: create run record in SQLite, generate run_id
  │     ├── on step: insert metrics_step record
  │     ├── on exit (success): mark run as completed, write final metrics
  │     └── on exit (exception): mark run as failed
  │
  ├── SQLiteBackend
  │     ├── runs table
  │     ├── hyperparams table (JSON per run)
  │     ├── metrics_steps table (step, loss, lr, grad_norm per run)
  │     └── artifacts table (checkpoint path, eval path per run)
  │
  └── ExperimentsCLI
        ├── list              <- list all runs, Rich table
        ├── compare [id ...]  <- side-by-side metric comparison
        ├── plot [id]         <- terminal training curve with plotext
        ├── export [id]       <- CSV dump
        ├── tag [id] [label]  <- annotate a run
        └── show [id]         <- full detail view of a single run
```

### Database Schema

```sql
-- experiments/runs.db

CREATE TABLE runs (
    run_id      TEXT PRIMARY KEY,     -- e.g. "run_20260320_143201_a3f7"
    config_name TEXT,                 -- "tiny", "small", etc.
    status      TEXT,                 -- "running", "completed", "failed", "interrupted"
    started_at  TEXT,                 -- ISO 8601
    ended_at    TEXT,
    tag         TEXT,                 -- user annotation
    notes       TEXT
);

CREATE TABLE hyperparams (
    run_id      TEXT REFERENCES runs(run_id),
    key         TEXT,
    value       TEXT,                 -- JSON-serialized
    PRIMARY KEY (run_id, key)
);

CREATE TABLE metrics_steps (
    run_id      TEXT REFERENCES runs(run_id),
    step        INTEGER,
    loss        REAL,
    lr          REAL,
    grad_norm   REAL,
    tokens_per_sec REAL,
    PRIMARY KEY (run_id, step)
);

CREATE TABLE final_metrics (
    run_id      TEXT REFERENCES runs(run_id),
    key         TEXT,
    value       REAL,
    PRIMARY KEY (run_id, key)
);

CREATE TABLE artifacts (
    run_id      TEXT REFERENCES runs(run_id),
    artifact_type TEXT,               -- "checkpoint", "eval_results", "config"
    path        TEXT,
    recorded_at TEXT
);

CREATE INDEX idx_metrics_steps_run ON metrics_steps(run_id, step);
```

---

## Implementation Steps

### Step 1: SQLite backend

```python
# src/cola_coder/tracking/backend.py
import sqlite3
import json
from pathlib import Path
from typing import Optional, Any
from contextlib import contextmanager

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, config_name TEXT, status TEXT,
    started_at TEXT, ended_at TEXT, tag TEXT, notes TEXT
);
CREATE TABLE IF NOT EXISTS hyperparams (
    run_id TEXT, key TEXT, value TEXT, PRIMARY KEY (run_id, key)
);
CREATE TABLE IF NOT EXISTS metrics_steps (
    run_id TEXT, step INTEGER, loss REAL, lr REAL,
    grad_norm REAL, tokens_per_sec REAL, PRIMARY KEY (run_id, step)
);
CREATE TABLE IF NOT EXISTS final_metrics (
    run_id TEXT, key TEXT, value REAL, PRIMARY KEY (run_id, key)
);
CREATE TABLE IF NOT EXISTS artifacts (
    run_id TEXT, artifact_type TEXT, path TEXT, recorded_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_ms_run ON metrics_steps(run_id, step);
"""

class SQLiteBackend:
    def __init__(self, db_path: Path = Path("experiments/runs.db")):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()

    @contextmanager
    def transaction(self):
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def create_run(self, run_id: str, config_name: str, started_at: str) -> None:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO runs (run_id, config_name, status, started_at) VALUES (?, ?, 'running', ?)",
                (run_id, config_name, started_at)
            )

    def log_hyperparams(self, run_id: str, params: dict[str, Any]) -> None:
        with self.transaction() as c:
            for key, value in params.items():
                c.execute(
                    "INSERT OR REPLACE INTO hyperparams (run_id, key, value) VALUES (?, ?, ?)",
                    (run_id, key, json.dumps(value))
                )

    def log_step(self, run_id: str, step: int, metrics: dict[str, float]) -> None:
        with self.transaction() as c:
            c.execute(
                """INSERT OR REPLACE INTO metrics_steps
                   (run_id, step, loss, lr, grad_norm, tokens_per_sec)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, step,
                 metrics.get("loss"), metrics.get("lr"),
                 metrics.get("grad_norm"), metrics.get("tokens_per_sec"))
            )

    def log_final_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        with self.transaction() as c:
            for key, value in metrics.items():
                c.execute(
                    "INSERT OR REPLACE INTO final_metrics (run_id, key, value) VALUES (?, ?, ?)",
                    (run_id, key, value)
                )

    def finish_run(self, run_id: str, status: str, ended_at: str) -> None:
        with self.transaction() as c:
            c.execute(
                "UPDATE runs SET status=?, ended_at=? WHERE run_id=?",
                (status, ended_at, run_id)
            )

    def log_artifact(self, run_id: str, artifact_type: str, path: str, recorded_at: str) -> None:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO artifacts (run_id, artifact_type, path, recorded_at) VALUES (?, ?, ?, ?)",
                (run_id, artifact_type, path, recorded_at)
            )

    def get_all_runs(self) -> list[dict]:
        cursor = self._conn.execute("""
            SELECT r.*,
                   MIN(ms.loss) as best_loss,
                   MAX(ms.step) as max_step,
                   fm.value as final_perplexity
            FROM runs r
            LEFT JOIN metrics_steps ms ON r.run_id = ms.run_id
            LEFT JOIN final_metrics fm ON r.run_id = fm.run_id AND fm.key = 'perplexity'
            GROUP BY r.run_id
            ORDER BY r.started_at DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

    def get_run_steps(self, run_id: str) -> list[dict]:
        cursor = self._conn.execute(
            "SELECT * FROM metrics_steps WHERE run_id=? ORDER BY step",
            (run_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_run_hyperparams(self, run_id: str) -> dict[str, Any]:
        cursor = self._conn.execute(
            "SELECT key, value FROM hyperparams WHERE run_id=?", (run_id,)
        )
        return {row["key"]: json.loads(row["value"]) for row in cursor.fetchall()}
```

### Step 2: `ExperimentTracker` context manager

```python
# src/cola_coder/tracking/tracker.py
import uuid
from datetime import datetime, timezone
from typing import Optional, Any
from pathlib import Path
from .backend import SQLiteBackend

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _make_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:4]
    return f"run_{ts}_{uid}"

class ExperimentTracker:
    def __init__(
        self,
        config_name: str,
        hyperparams: dict[str, Any],
        db_path: Path = Path("experiments/runs.db"),
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.config_name = config_name
        self.hyperparams = hyperparams
        self.run_id: Optional[str] = None
        self._backend = SQLiteBackend(db_path) if enabled else None

    def __enter__(self) -> "ExperimentTracker":
        if not self.enabled:
            return self
        self._backend.connect()
        self.run_id = _make_run_id()
        self._backend.create_run(self.run_id, self.config_name, _now())
        self._backend.log_hyperparams(self.run_id, self.hyperparams)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self.enabled:
            return
        status = "failed" if exc_type else "completed"
        self._backend.finish_run(self.run_id, status, _now())
        self._backend.close()

    def log_step(self, step: int, **metrics: float) -> None:
        if not self.enabled or not self.run_id:
            return
        self._backend.log_step(self.run_id, step, metrics)

    def log_final(self, **metrics: float) -> None:
        if not self.enabled or not self.run_id:
            return
        self._backend.log_final_metrics(self.run_id, metrics)

    def log_artifact(self, artifact_type: str, path: Path) -> None:
        if not self.enabled or not self.run_id:
            return
        self._backend.log_artifact(self.run_id, artifact_type, str(path), _now())
```

### Step 3: Trainer integration (zero-touch)

```python
# In src/cola_coder/training/trainer.py — wrap the training loop

# Existing trainer __init__: build tracker from config
from cola_coder.tracking.tracker import ExperimentTracker

def train(config: TrainConfig) -> None:
    tracker = ExperimentTracker(
        config_name=config.name,
        hyperparams={
            "lr": config.learning_rate,
            "batch_size": config.batch_size,
            "seq_len": config.max_seq_len,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "n_kv_heads": config.n_kv_heads,
            "hidden_dim": config.hidden_dim,
            "grad_accum": config.gradient_accumulation_steps,
            "precision": config.precision,
            "warmup_steps": config.warmup_steps,
        },
        enabled=config.track_experiments,  # from config YAML
    )

    with tracker:
        for step in range(config.max_steps):
            loss, grad_norm = train_step(...)  # existing logic

            if step % config.log_every == 0:
                tracker.log_step(
                    step,
                    loss=loss,
                    lr=scheduler.get_last_lr()[0],
                    grad_norm=grad_norm,
                    tokens_per_sec=tokens_per_sec,
                )

            if step % config.checkpoint_every == 0:
                ckpt_path = save_checkpoint(...)
                tracker.log_artifact("checkpoint", ckpt_path)

        # After training loop ends
        tracker.log_final(loss=final_loss, perplexity=math.exp(final_loss))
```

### Step 4: CLI interface

```python
# scripts/experiments.py
import sys
from pathlib import Path
import click
import plotext as plt
from rich.console import Console
from rich.table import Table
from cola_coder.tracking.backend import SQLiteBackend

console = Console()

@click.group()
@click.option("--db", default="experiments/runs.db", help="Path to experiments database")
@click.pass_context
def cli(ctx, db):
    ctx.ensure_object(dict)
    backend = SQLiteBackend(Path(db))
    backend.connect()
    ctx.obj["backend"] = backend

@cli.command()
@click.pass_context
def list(ctx):
    """List all experiment runs."""
    backend = ctx.obj["backend"]
    runs = backend.get_all_runs()
    if not runs:
        console.print("[yellow]No experiments recorded yet.[/yellow]")
        return

    table = Table(title=f"Experiments ({len(runs)} runs)", show_lines=True)
    table.add_column("ID",       style="cyan",  width=6)
    table.add_column("Run ID",   style="dim",   width=20)
    table.add_column("Config",   width=10)
    table.add_column("Steps",    justify="right", width=8)
    table.add_column("Best Loss",justify="right", width=10)
    table.add_column("PPL",      justify="right", width=8)
    table.add_column("Status",   width=12)
    table.add_column("Started",  width=14)
    table.add_column("Tag",      style="yellow")

    for i, run in enumerate(runs, 1):
        best_loss = f"{run['best_loss']:.4f}" if run['best_loss'] else "—"
        ppl = f"{run['final_perplexity']:.1f}" if run['final_perplexity'] else "—"
        steps = str(run['max_step']) if run['max_step'] else "0"
        status_color = {"completed": "green", "running": "yellow", "failed": "red"}.get(run['status'], "dim")
        status = f"[{status_color}]{run['status']}[/{status_color}]"
        tag = run['tag'] or ""
        table.add_row(
            str(i), run['run_id'], run['config_name'] or "?",
            steps, best_loss, ppl, status, _relative_time(run['started_at']), tag
        )
    console.print(table)

@cli.command()
@click.argument("run_ids", nargs=-1, required=True)
@click.pass_context
def compare(ctx, run_ids):
    """Compare metrics across runs. Usage: compare run_20260320_... run_20260319_..."""
    backend = ctx.obj["backend"]
    table = Table(title="Run Comparison", show_lines=True)
    table.add_column("Metric", style="bold")
    for run_id in run_ids:
        table.add_column(run_id[:20], justify="right")

    # Collect hyperparams and final metrics
    metrics_rows = {}
    for run_id in run_ids:
        hp = backend.get_run_hyperparams(run_id)
        for key, val in hp.items():
            if key not in metrics_rows:
                metrics_rows[key] = {}
            metrics_rows[key][run_id] = str(val)

    for metric_name, vals in metrics_rows.items():
        row = [metric_name] + [vals.get(rid, "—") for rid in run_ids]
        table.add_row(*row)
    console.print(table)

@cli.command()
@click.argument("run_id")
@click.pass_context
def plot(ctx, run_id):
    """Plot training loss curve for a run in the terminal."""
    backend = ctx.obj["backend"]
    steps_data = backend.get_run_steps(run_id)
    if not steps_data:
        console.print(f"[red]No step data for run {run_id}[/red]")
        return

    steps = [r["step"] for r in steps_data if r["loss"] is not None]
    losses = [r["loss"] for r in steps_data if r["loss"] is not None]

    plt.clf()
    plt.plot(steps, losses, label="Training Loss", color="cyan")
    plt.title(f"Training Loss — {run_id[:24]}")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.theme("dark")
    plt.show()

@cli.command()
@click.argument("run_id")
@click.argument("output", default=None, required=False)
@click.pass_context
def export(ctx, run_id, output):
    """Export run metrics to CSV."""
    import csv
    backend = ctx.obj["backend"]
    steps_data = backend.get_run_steps(run_id)
    out_path = output or f"{run_id}_metrics.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "lr", "grad_norm", "tokens_per_sec"])
        writer.writeheader()
        writer.writerows(steps_data)
    console.print(f"[green]Exported {len(steps_data)} steps to {out_path}[/green]")

@cli.command()
@click.argument("run_id")
@click.argument("tag_text")
@click.pass_context
def tag(ctx, run_id, tag_text):
    """Tag a run with a label. Example: tag run_20260320_... 'best-so-far'"""
    backend = ctx.obj["backend"]
    with backend.transaction() as c:
        c.execute("UPDATE runs SET tag=? WHERE run_id=?", (tag_text, run_id))
    console.print(f"[green]Tagged {run_id[:20]} as '{tag_text}'[/green]")

if __name__ == "__main__":
    cli()
```

---

## Key Files to Modify

- `src/cola_coder/tracking/backend.py` — new file: `SQLiteBackend`
- `src/cola_coder/tracking/tracker.py` — new file: `ExperimentTracker` context manager
- `src/cola_coder/training/trainer.py` — modify: wrap training loop with tracker
- `scripts/experiments.py` — new file: CLI (list, compare, plot, export, tag)
- `configs/tiny.yaml` etc. — add `tracking.enabled: true` field
- `pyproject.toml` — add `plotext>=5.2` and `click>=8.1` to optional deps

---

## Testing Strategy

```python
# tests/test_tracker.py
import pytest
from pathlib import Path
from cola_coder.tracking.tracker import ExperimentTracker
from cola_coder.tracking.backend import SQLiteBackend

@pytest.fixture
def tmp_backend(tmp_path):
    backend = SQLiteBackend(tmp_path / "test.db")
    backend.connect()
    yield backend
    backend.close()

def test_run_lifecycle(tmp_path):
    tracker = ExperimentTracker(
        config_name="tiny",
        hyperparams={"lr": 3e-4, "batch_size": 8},
        db_path=tmp_path / "test.db",
        enabled=True,
    )
    with tracker:
        tracker.log_step(100, loss=5.2, lr=3e-4, grad_norm=1.1, tokens_per_sec=45000)
        tracker.log_step(200, loss=4.8, lr=3e-4, grad_norm=0.9, tokens_per_sec=45500)
        tracker.log_final(loss=4.8, perplexity=121.5)

    backend = SQLiteBackend(tmp_path / "test.db")
    backend.connect()
    runs = backend.get_all_runs()
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    assert runs[0]["max_step"] == 200
    assert abs(runs[0]["best_loss"] - 4.8) < 0.001

def test_run_marked_failed_on_exception(tmp_path):
    tracker = ExperimentTracker("tiny", {}, db_path=tmp_path / "test.db", enabled=True)
    try:
        with tracker:
            tracker.log_step(50, loss=9.9, lr=3e-4, grad_norm=100.0, tokens_per_sec=0)
            raise RuntimeError("OOM")
    except RuntimeError:
        pass
    backend = SQLiteBackend(tmp_path / "test.db")
    backend.connect()
    runs = backend.get_all_runs()
    assert runs[0]["status"] == "failed"

def test_disabled_tracker_is_noop(tmp_path):
    tracker = ExperimentTracker("tiny", {}, db_path=tmp_path / "test.db", enabled=False)
    with tracker:
        tracker.log_step(100, loss=5.0, lr=3e-4, grad_norm=1.0, tokens_per_sec=40000)
    assert not (tmp_path / "test.db").exists()

def test_hyperparams_stored(tmp_path):
    params = {"lr": 3e-4, "n_layers": 12, "precision": "bf16"}
    tracker = ExperimentTracker("small", params, db_path=tmp_path / "test.db", enabled=True)
    with tracker:
        pass
    backend = SQLiteBackend(tmp_path / "test.db")
    backend.connect()
    runs = backend.get_all_runs()
    hp = backend.get_run_hyperparams(runs[0]["run_id"])
    assert hp["n_layers"] == 12
    assert hp["precision"] == "bf16"
```

---

## Performance Considerations

- **Write-ahead logging**: SQLite in WAL mode handles concurrent writes (training loop + possible CLI queries) without locking. Enable with `PRAGMA journal_mode=WAL` on connection.
- **Batched step writes**: Don't write to SQLite on every single step. The trainer already logs every `log_every` steps (typically every 10-50 steps), which is fine. For very frequent writes, a write buffer in `ExperimentTracker` can batch inserts.
- **No impact when disabled**: `ExperimentTracker` with `enabled=False` is a null object — every method is a single `if not self.enabled: return` check. Zero overhead on the hot training path.
- **`plotext` terminal plots**: Renders curves in-terminal without launching matplotlib or a GUI. No window manager dependency. Ideal for remote SSH sessions or headless Windows environments.
- **Index on `(run_id, step)`**: Keeps `get_run_steps()` fast even with millions of step records across many runs.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `sqlite3` | stdlib | Database backend |
| `click` | `>=8.1` | CLI framework for `scripts/experiments.py` |
| `plotext` | `>=5.2` | In-terminal training curve plots |
| `rich` | `>=13.0` | Already present — table rendering |
| `pyyaml` | `>=6.0` | Already present — config reading |

`click` and `plotext` are lightweight packages. Both can be optional extras in `pyproject.toml` under `[project.optional-dependencies] tracking`.

---

## Estimated Complexity

**Medium.** The SQLite backend and context manager are ~250 lines of straightforward Python. The CLI adds another ~200 lines. The main complexity is:

1. Trainer integration: threading the tracker through the existing training loop without breaking it.
2. Schema evolution: the DB schema may need to grow as new metrics are tracked. Using `CREATE TABLE IF NOT EXISTS` and `INSERT OR REPLACE` handles this gracefully.
3. The `plotext` integration is trivial but requires verifying terminal width and color support.

Estimated implementation time: 6-8 hours including tests and trainer integration.

---

## 2026 Best Practices

- **SQLite as the application database**: SQLite is the correct choice for a local single-user tool in 2026. No PostgreSQL, no Redis, no separate process. The database is a single file you can copy, back up, or delete.
- **Context manager instrumentation**: Wrapping the training loop with `with tracker:` is the Python-idiomatic way to add cross-cutting concerns (logging, timing, cleanup) without polluting business logic.
- **Null object pattern for disabled feature**: `enabled=False` returns the same `ExperimentTracker` object with no-op methods rather than requiring `if config.tracking.enabled:` guards throughout the trainer.
- **WAL mode for concurrent access**: Allowing reads during writes is the correct SQLite configuration for a tool where the CLI might query the database while training is in progress.
- **`plotext` over matplotlib for CLI**: Terminal-native plots avoid GUI dependencies entirely. In 2026 this is the standard for CLI ML tools targeting server/remote environments.
- **Declarative schema with `IF NOT EXISTS`**: The schema self-creates on first run and never requires a migration step for a new install — appropriate for a developer tool.
