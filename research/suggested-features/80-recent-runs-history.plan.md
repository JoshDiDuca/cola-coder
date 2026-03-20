# 80 - Recent Runs History

## Overview

Log every CLI command execution with timestamp, duration, outcome, and key metrics. Display "last 10 runs" in the main menu. Support re-running previous commands and inspecting their config snapshot. Store in SQLite. Useful for reproducibility and debugging.

**Feature flag:** `config.run_history.enabled` (default: `true`)

---

## Motivation

After a week of experiments, it becomes hard to remember:
- Which command produced checkpoint X?
- What were the exact arguments used for that last data prep run?
- Did the eval run last night succeed or fail?
- What learning rate was used in that training run that worked well?

A persistent command history with config snapshots solves all of these. Unlike shell history (`~/.bash_history`), this stores structured data: arguments, outcomes, durations, and key metrics—not just the raw command string.

**Contrast with experiment tracker (plan 71)**: plan 71 tracks training runs with metrics. Plan 80 tracks all CLI invocations (prepare, train, eval, generate, inspect) as a log. They are complementary.

---

## Architecture / Design

### Storage Backend

**SQLite** (`run_history.db`)
- Queryable, structured, handles concurrent writes with WAL mode
- Easy to inspect with `sqlite3` CLI
- Lightweight: ~1-2KB per run entry including config snapshot

### Record Schema

```python
@dataclass
class RunRecord:
    id: int
    command: str              # e.g., "train"
    subcommand: str | None    # e.g., "run" (for pipeline run)
    args: dict                # parsed arguments as dict
    cwd: str                  # working directory
    started_at: str           # ISO8601
    ended_at: str | None
    duration_sec: float | None
    outcome: str              # "success" | "failed" | "interrupted"
    exit_code: int | None
    key_metrics: dict         # e.g., {"final_loss": 2.341, "steps": 5000}
    config_snapshot: dict     # full config YAML at run time
    git_hash: str | None
    notes: str
```

---

## Implementation Steps

### Step 1: Run History Logger (`tracking/run_history.py`)

```python
import sqlite3
import json
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    command TEXT NOT NULL,
    subcommand TEXT,
    args_json TEXT DEFAULT '{}',
    cwd TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_sec REAL,
    outcome TEXT DEFAULT 'unknown',
    exit_code INTEGER,
    key_metrics_json TEXT DEFAULT '{}',
    config_snapshot_json TEXT DEFAULT '{}',
    git_hash TEXT,
    notes TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_command ON runs(command);
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

class RunHistoryLogger:
    def __init__(self, db_path: str = "run_history.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._active_run_id: int | None = None

    def start_run(
        self,
        command: str,
        subcommand: str = None,
        args: dict = None,
        config_snapshot: dict = None,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO runs
               (command, subcommand, args_json, cwd, started_at,
                config_snapshot_json, git_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                command, subcommand,
                json.dumps(args or {}, default=str),
                os.getcwd(),
                _now(),
                json.dumps(config_snapshot or {}, default=str),
                _git_hash(),
            ),
        )
        self._conn.commit()
        self._active_run_id = cur.lastrowid
        return self._active_run_id

    def end_run(
        self,
        outcome: str = "success",
        exit_code: int = 0,
        key_metrics: dict = None,
        notes: str = "",
    ):
        if not self._active_run_id:
            return
        now = _now()
        row = self._conn.execute(
            "SELECT started_at FROM runs WHERE id=?",
            (self._active_run_id,)
        ).fetchone()
        started = row["started_at"] if row else now
        try:
            t_start = datetime.fromisoformat(started.replace("Z", "+00:00"))
            t_end = datetime.fromisoformat(now.replace("Z", "+00:00"))
            duration = (t_end - t_start).total_seconds()
        except Exception:
            duration = None

        self._conn.execute(
            """UPDATE runs
               SET ended_at=?, duration_sec=?, outcome=?,
                   exit_code=?, key_metrics_json=?, notes=?
               WHERE id=?""",
            (now, duration, outcome, exit_code,
             json.dumps(key_metrics or {}, default=str),
             notes, self._active_run_id),
        )
        self._conn.commit()
        self._active_run_id = None

    @contextmanager
    def track_run(
        self,
        command: str,
        subcommand: str = None,
        args: dict = None,
        config_snapshot: dict = None,
    ):
        run_id = self.start_run(command, subcommand, args, config_snapshot)
        metrics = {}
        try:
            yield metrics
            self.end_run("success", 0, metrics)
        except KeyboardInterrupt:
            self.end_run("interrupted", 130, metrics)
            raise
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            self.end_run("failed" if code != 0 else "success", code, metrics)
            raise
        except Exception as e:
            self.end_run("failed", 1, metrics, notes=str(e)[:200])
            raise

    def get_recent(self, n: int = 10, command: str = None) -> list[dict]:
        query = "SELECT * FROM runs"
        params = []
        if command:
            query += " WHERE command = ?"
            params.append(command)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(n)
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_run(self, run_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE id=?", (run_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def _row_to_dict(self, row) -> dict:
        d = dict(row)
        d["args"] = json.loads(d.pop("args_json", "{}"))
        d["key_metrics"] = json.loads(d.pop("key_metrics_json", "{}"))
        d["config_snapshot"] = json.loads(d.pop("config_snapshot_json", "{}"))
        return d

    def rebuild_command(self, run_id: int) -> str | None:
        """Reconstruct the CLI command string from a run record."""
        record = self.get_run(run_id)
        if not record:
            return None
        parts = ["cola-coder", record["command"]]
        if record.get("subcommand"):
            parts.append(record["subcommand"])
        for key, val in record.get("args", {}).items():
            if isinstance(val, bool):
                if val:
                    parts.append(f"--{key.replace('_', '-')}")
            elif val is not None:
                parts.append(f"--{key.replace('_', '-')} {val}")
        return " ".join(parts)

    def prune(self, keep_n: int = 1000):
        """Delete oldest records beyond keep_n."""
        self._conn.execute(
            """DELETE FROM runs WHERE id NOT IN (
               SELECT id FROM runs ORDER BY started_at DESC LIMIT ?
            )""",
            (keep_n,),
        )
        self._conn.commit()

    def clear_older_than(self, days: int):
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        self._conn.execute("DELETE FROM runs WHERE started_at < ?", (cutoff,))
        self._conn.commit()
```

### Step 2: Main Menu Widget (`cli/main_menu.py`)

```python
from rich.table import Table
from rich.console import Console

def _format_duration(sec: float) -> str:
    if sec is None:
        return "—"
    if sec < 60:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{sec/60:.1f}m"
    return f"{sec/3600:.1f}h"

def show_recent_runs_widget(history: RunHistoryLogger, n: int = 10):
    """Display last N runs as a compact table for the main menu."""
    recent = history.get_recent(n)
    if not recent:
        return

    console = Console()
    table = Table(
        title="[dim]Recent Runs[/]",
        show_header=True,
        box=None,
        padding=(0, 1),
        header_style="dim",
    )
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Command", style="cyan", width=22)
    table.add_column("When", style="dim", width=12)
    table.add_column("Duration", justify="right", width=8)
    table.add_column("Outcome", width=12)
    table.add_column("Key Metric", style="dim")

    for run in recent:
        outcome = run.get("outcome", "unknown")
        color = {"success": "green", "failed": "red", "interrupted": "yellow"}.get(
            outcome, "dim"
        )
        cmd_display = run["command"]
        if run.get("subcommand"):
            cmd_display += f" {run['subcommand']}"

        started = run.get("started_at", "")
        when_str = started[5:16].replace("T", " ") if len(started) >= 16 else "—"

        metrics = run.get("key_metrics", {})
        metric_str = ""
        for preferred in ["final_loss", "type_correctness_rate", "total_tokens", "passed"]:
            if preferred in metrics:
                v = metrics[preferred]
                metric_str = f"{preferred}={v:.3f}" if isinstance(v, float) else f"{preferred}={v}"
                break

        table.add_row(
            str(run["id"]),
            cmd_display[:22],
            when_str,
            _format_duration(run.get("duration_sec")),
            f"[{color}]{outcome}[/]",
            metric_str,
        )

    console.print(table)
```

### Step 3: CLI Commands (`cli/history_cmd.py`)

```python
import click
from pathlib import Path

@click.group("history")
def history_cmd():
    """View and manage CLI run history."""
    pass

@history_cmd.command("list")
@click.option("--command", default=None, help="Filter by command name")
@click.option("--n", default=10, help="Number of runs to show")
def cmd_list(command, n):
    """Show recent run history."""
    history = RunHistoryLogger()
    show_recent_runs_widget(history, n)

@history_cmd.command("show")
@click.argument("run_id", type=int)
def cmd_show(run_id):
    """Show details for a specific run."""
    from rich.console import Console
    history = RunHistoryLogger()
    record = history.get_run(run_id)
    if not record:
        click.echo(f"Run #{run_id} not found.")
        return

    console = Console()
    console.rule(f"[bold]Run #{run_id}[/]")
    console.print(f"  Command:  [cyan]{record['command']} {record.get('subcommand', '')}[/]")
    console.print(f"  Started:  {record['started_at'][:16]}")
    console.print(f"  Duration: {_format_duration(record.get('duration_sec'))}")
    outcome = record['outcome']
    color = {"success": "green", "failed": "red", "interrupted": "yellow"}.get(outcome, "dim")
    console.print(f"  Outcome:  [{color}]{outcome}[/]")
    console.print(f"  Git hash: {record.get('git_hash', '—')}")
    console.print(f"  CWD:      {record.get('cwd', '—')}")

    if record.get("args"):
        console.print("\n  [bold]Arguments:[/]")
        for k, v in record["args"].items():
            console.print(f"    {k}: [dim]{v}[/]")

    if record.get("key_metrics"):
        console.print("\n  [bold]Key Metrics:[/]")
        for k, v in record["key_metrics"].items():
            console.print(f"    {k}: [cyan]{v}[/]")

    if record.get("notes"):
        console.print(f"\n  [bold]Notes:[/] {record['notes']}")

    reconstructed = history.rebuild_command(run_id)
    if reconstructed:
        console.print(f"\n  [dim]Re-run command:[/]")
        console.print(f"  [green]{reconstructed}[/]")

@history_cmd.command("rerun")
@click.argument("run_id", type=int)
@click.option("--dry-run", is_flag=True, help="Print command without executing")
def cmd_rerun(run_id, dry_run):
    """Re-run a previous command with the same arguments."""
    history = RunHistoryLogger()
    cmd = history.rebuild_command(run_id)
    if not cmd:
        click.echo(f"Run #{run_id} not found.")
        return
    click.echo(f"Command:\n  {cmd}")
    if not dry_run:
        import subprocess
        result = subprocess.run(cmd.split())
        click.echo(f"Exit code: {result.returncode}")

@history_cmd.command("clear")
@click.option("--older-than-days", default=30, type=int)
@click.confirmation_option(prompt="Delete old runs from history?")
def cmd_clear(older_than_days):
    """Delete runs older than N days."""
    history = RunHistoryLogger()
    history.clear_older_than(older_than_days)
    click.echo(f"Deleted runs older than {older_than_days} days.")

@history_cmd.command("stats")
def cmd_stats():
    """Show summary stats for run history."""
    from rich.table import Table
    from rich.console import Console
    history = RunHistoryLogger()
    recent = history.get_recent(n=1000)
    if not recent:
        Console().print("[dim]No history.[/]")
        return

    from collections import Counter
    by_command = Counter(r["command"] for r in recent)
    by_outcome = Counter(r["outcome"] for r in recent)

    console = Console()
    table = Table(title="Run History Stats")
    table.add_column("Command")
    table.add_column("Count", justify="right")
    for cmd, count in by_command.most_common():
        table.add_row(cmd, str(count))
    console.print(table)

    console.print(
        f"\n  Total runs: {len(recent)}\n"
        f"  Success: [green]{by_outcome.get('success', 0)}[/]\n"
        f"  Failed:  [red]{by_outcome.get('failed', 0)}[/]\n"
        f"  Interrupted: [yellow]{by_outcome.get('interrupted', 0)}[/]"
    )
```

### Step 4: Auto-wrapping in CLI Dispatch (`cli/main.py`)

```python
# Wrap every command dispatch with track_run:

import click
from tracking.run_history import RunHistoryLogger

@click.group()
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)
    if not getattr(ctx, "_history_started", False):
        history = RunHistoryLogger()
        ctx.obj["history"] = history

# Alternatively, use a Click result callback:
@cli.result_callback()
def process_result(result, **kwargs):
    pass  # history recording happens in context managers per command

# Per-command example:
@cli.command("train")
@click.option("--config", required=True)
@click.pass_context
def train_cmd(ctx, config):
    history = ctx.obj["history"]
    with history.track_run(
        command="train",
        args={"config": config},
        config_snapshot=load_config(config),
    ) as run_metrics:
        result = run_training(config)
        run_metrics["final_loss"] = result["final_loss"]
        run_metrics["steps"] = result["total_steps"]
```

### Step 5: Config

```yaml
run_history:
  enabled: true
  db_path: run_history.db
  max_records: 1000
  capture_config_snapshot: true
  show_in_main_menu: true
  main_menu_count: 10
```

---

## Key Files to Modify

- `tracking/run_history.py` - New file: logger and context manager
- `cli/history_cmd.py` - New file: CLI commands
- `cli/main_menu.py` - Add recent runs widget
- `cli/main.py` - Wrap commands with `track_run`
- `config/training.yaml` - Add `run_history` section

---

## Testing Strategy

1. **Track run success test**: use `track_run`, assert run is recorded with `outcome="success"`.
2. **Failed run test**: raise exception inside `track_run`, assert `outcome="failed"`, exception re-raised.
3. **Interrupted test**: raise `KeyboardInterrupt`, assert `outcome="interrupted"`.
4. **Duration test**: sleep 0.1s inside `track_run`, assert `duration_sec >= 0.1`.
5. **Get recent test**: insert 15 runs, call `get_recent(n=10)`, assert 10 most recent returned.
6. **Rebuild command test**: insert run with known args dict, call `rebuild_command`, assert all flags in output.
7. **Prune test**: insert 20 runs, call `prune(keep_n=10)`, assert exactly 10 rows remain.
8. **WAL mode test**: open two connections simultaneously, assert no "database is locked" errors.

---

## Performance Considerations

- One SQLite write at run start, one at run end: negligible overhead for any CLI command.
- `get_recent(n=10)` with `ORDER BY started_at DESC LIMIT 10`: indexed, <1ms.
- Config snapshot: `json.dumps(config)` on a typical ~50-key config takes <1ms.
- WAL mode (`PRAGMA journal_mode=WAL`) allows concurrent reads while writing. Prevents "database locked" errors if a background process reads history while training writes to it.
- `prune(keep_n=1000)` runs as a maintenance step at startup. 1000 records × ~2KB = 2MB max DB size.

---

## Dependencies

No new dependencies. Uses `sqlite3`, `json`, `subprocess`, `contextlib` (all stdlib), `rich` and `click` (already required).

---

## Estimated Complexity

**Low.** Standard CRUD with a context manager wrapper. The main effort is integrating the `track_run` wrapper into each CLI command without adding boilerplate. Estimated implementation time: 1-2 days.

---

## 2026 Best Practices

- **WAL mode for SQLite**: always use `PRAGMA journal_mode=WAL`. It enables concurrent reads and significantly improves write performance for append-heavy workloads like a run log.
- **Context manager wrapping, not manual calls**: wrap commands automatically rather than requiring each command to call `history.start_run()`. Manual calls will be forgotten for new commands.
- **Config snapshot is the key feature**: the raw command string is in shell history. The value here is the structured config snapshot—knowing exactly what settings were active. Always capture it.
- **Git hash ties to code version**: combined with the config snapshot, a git hash lets you recreate the exact conditions of any run by checking out that commit.
- **Re-run is a power feature**: the `rerun` command reconstructs and re-executes a previous invocation. This makes it trivial to "repeat that training run on the new dataset" without remembering all the flags.
