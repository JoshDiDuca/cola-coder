# 76 - Checkpoint Leaderboard

## Overview

A local SQLite database that tracks pass@1, perplexity, syntax validity, and generation quality for every evaluated checkpoint. Auto-populated after each training run. CLI shows a ranked leaderboard table with Rich formatting. Supports filtering by model size, training data version, and date. Provides colored rankings and side-by-side comparison of any two entries.

**Feature flag:** `config.leaderboard.enabled` (default: `true`)

---

## Motivation

After training 20+ checkpoints across multiple experiments, answering "which checkpoint is actually the best?" requires manually running eval on each one or trusting that the last checkpoint is the best. Neither is reliable.

A leaderboard provides:
- **Single source of truth** for which checkpoint performs best on which metric
- **Regression detection**: immediately see if a new training run produced worse results than a previous one
- **Model selection**: quickly identify the best checkpoint for a specific deployment
- **Progress narrative**: view the history of improvements across the entire project

---

## Architecture / Design

### Database Schema

```sql
CREATE TABLE checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,            -- e.g., "step-5000"
    path TEXT NOT NULL,
    model_size_params INTEGER,
    dataset_version TEXT,
    training_steps INTEGER,
    training_loss REAL,
    training_date TEXT,
    added_at TEXT NOT NULL
);

CREATE TABLE evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id INTEGER NOT NULL REFERENCES checkpoints(id),
    eval_date TEXT NOT NULL,
    eval_suite TEXT NOT NULL DEFAULT 'default',
    pass_at_1 REAL,
    perplexity REAL,
    syntax_validity_rate REAL,
    type_correctness_rate REAL,
    token_efficiency REAL,
    nano_benchmark_score REAL,
    notes TEXT
);

CREATE TABLE eval_details (
    eval_id INTEGER NOT NULL REFERENCES evaluations(id),
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (eval_id, metric)
);

CREATE INDEX idx_evals_checkpoint ON evaluations(checkpoint_id);
```

---

## Implementation Steps

### Step 1: Leaderboard DB (`tracking/leaderboard.py`)

```python
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL,
    model_size_params INTEGER,
    dataset_version TEXT,
    training_steps INTEGER,
    training_loss REAL,
    training_date TEXT,
    added_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id INTEGER NOT NULL REFERENCES checkpoints(id),
    eval_date TEXT NOT NULL,
    eval_suite TEXT NOT NULL DEFAULT 'default',
    pass_at_1 REAL,
    perplexity REAL,
    syntax_validity_rate REAL,
    type_correctness_rate REAL,
    token_efficiency REAL,
    nano_benchmark_score REAL,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS eval_details (
    eval_id INTEGER NOT NULL REFERENCES evaluations(id),
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (eval_id, metric)
);
CREATE INDEX IF NOT EXISTS idx_evals_checkpoint ON evaluations(checkpoint_id);
"""

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

@dataclass
class LeaderboardEntry:
    checkpoint_name: str
    checkpoint_path: str
    eval_date: str
    pass_at_1: float | None
    perplexity: float | None
    syntax_validity_rate: float | None
    type_correctness_rate: float | None
    token_efficiency: float | None
    nano_benchmark_score: float | None
    model_size_params: int | None
    dataset_version: str | None
    training_steps: int | None
    training_loss: float | None

class CheckpointLeaderboard:
    def __init__(self, db_path: str = "leaderboard.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def register_checkpoint(
        self,
        name: str,
        path: str,
        model_size_params: int = None,
        dataset_version: str = None,
        training_steps: int = None,
        training_loss: float = None,
        training_date: str = None,
    ) -> int:
        """Add or update a checkpoint entry."""
        existing = self.conn.execute(
            "SELECT id FROM checkpoints WHERE name=?", (name,)
        ).fetchone()

        if existing:
            self.conn.execute(
                """UPDATE checkpoints
                   SET path=?, model_size_params=?, dataset_version=?,
                       training_steps=?, training_loss=?, training_date=?
                   WHERE name=?""",
                (path, model_size_params, dataset_version,
                 training_steps, training_loss, training_date, name),
            )
            self.conn.commit()
            return existing["id"]

        cur = self.conn.execute(
            """INSERT INTO checkpoints
               (name, path, model_size_params, dataset_version,
                training_steps, training_loss, training_date, added_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (name, path, model_size_params, dataset_version,
             training_steps, training_loss, training_date, _now()),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_evaluation(
        self,
        checkpoint_name: str,
        metrics: dict,
        eval_suite: str = "default",
        notes: str = "",
    ) -> int:
        ckpt = self.conn.execute(
            "SELECT id FROM checkpoints WHERE name=?", (checkpoint_name,)
        ).fetchone()
        if not ckpt:
            raise ValueError(f"Checkpoint '{checkpoint_name}' not registered.")

        cur = self.conn.execute(
            """INSERT INTO evaluations
               (checkpoint_id, eval_date, eval_suite,
                pass_at_1, perplexity, syntax_validity_rate,
                type_correctness_rate, token_efficiency, nano_benchmark_score, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                ckpt["id"], _now(), eval_suite,
                metrics.get("pass_at_1"),
                metrics.get("perplexity"),
                metrics.get("syntax_validity_rate"),
                metrics.get("type_correctness_rate"),
                metrics.get("token_efficiency"),
                metrics.get("nano_benchmark_score"),
                notes,
            ),
        )
        eval_id = cur.lastrowid

        # Store additional arbitrary metrics in eval_details
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and k not in {
                "pass_at_1", "perplexity", "syntax_validity_rate",
                "type_correctness_rate", "token_efficiency", "nano_benchmark_score"
            }:
                self.conn.execute(
                    "INSERT OR REPLACE INTO eval_details(eval_id, metric, value) VALUES (?,?,?)",
                    (eval_id, k, float(v)),
                )

        self.conn.commit()
        return eval_id

    def get_leaderboard(
        self,
        sort_by: str = "pass_at_1",
        sort_desc: bool = True,
        min_date: str = None,
        dataset_version: str = None,
        max_model_size: int = None,
    ) -> list[LeaderboardEntry]:
        """Return leaderboard entries sorted by metric."""
        query = """
            SELECT c.name, c.path, e.eval_date,
                   e.pass_at_1, e.perplexity, e.syntax_validity_rate,
                   e.type_correctness_rate, e.token_efficiency, e.nano_benchmark_score,
                   c.model_size_params, c.dataset_version, c.training_steps, c.training_loss
            FROM checkpoints c
            JOIN evaluations e ON c.id = e.checkpoint_id
            WHERE 1=1
        """
        params = []

        if min_date:
            query += " AND e.eval_date >= ?"
            params.append(min_date)
        if dataset_version:
            query += " AND c.dataset_version = ?"
            params.append(dataset_version)
        if max_model_size:
            query += " AND (c.model_size_params IS NULL OR c.model_size_params <= ?)"
            params.append(max_model_size)

        valid_sort_cols = {
            "pass_at_1", "perplexity", "syntax_validity_rate",
            "type_correctness_rate", "token_efficiency",
            "nano_benchmark_score", "training_steps",
        }
        if sort_by in valid_sort_cols:
            order = "DESC" if sort_desc else "ASC"
            # NULL values last
            query += f" ORDER BY {sort_by} IS NULL, {sort_by} {order}"

        rows = self.conn.execute(query, params).fetchall()
        return [
            LeaderboardEntry(
                checkpoint_name=r["name"],
                checkpoint_path=r["path"],
                eval_date=r["eval_date"][:10],
                pass_at_1=r["pass_at_1"],
                perplexity=r["perplexity"],
                syntax_validity_rate=r["syntax_validity_rate"],
                type_correctness_rate=r["type_correctness_rate"],
                token_efficiency=r["token_efficiency"],
                nano_benchmark_score=r["nano_benchmark_score"],
                model_size_params=r["model_size_params"],
                dataset_version=r["dataset_version"],
                training_steps=r["training_steps"],
                training_loss=r["training_loss"],
            )
            for r in rows
        ]

    def compare_two(self, name_a: str, name_b: str) -> dict:
        """Return side-by-side comparison of two checkpoint evaluations."""
        entries = self.get_leaderboard()
        a = next((e for e in entries if e.checkpoint_name == name_a), None)
        b = next((e for e in entries if e.checkpoint_name == name_b), None)
        if not a or not b:
            raise ValueError("One or both checkpoint names not found.")

        metrics = ["pass_at_1", "perplexity", "syntax_validity_rate",
                   "type_correctness_rate", "token_efficiency"]
        comparison = {}
        for m in metrics:
            va = getattr(a, m)
            vb = getattr(b, m)
            comparison[m] = {"a": va, "b": vb}
        return comparison
```

### Step 2: CLI Commands (`cli/leaderboard_cmd.py`)

```python
from rich.table import Table
from rich.console import Console
from rich.text import Text

METRIC_CONFIG = {
    "pass_at_1":             {"display": "pass@1",       "higher_is_better": True,  "fmt": ".1%"},
    "perplexity":            {"display": "Perplexity",   "higher_is_better": False, "fmt": ".2f"},
    "syntax_validity_rate":  {"display": "Syntax%",      "higher_is_better": True,  "fmt": ".1%"},
    "type_correctness_rate": {"display": "TypeCheck%",   "higher_is_better": True,  "fmt": ".1%"},
    "token_efficiency":      {"display": "Efficiency",   "higher_is_better": True,  "fmt": ".1%"},
    "nano_benchmark_score":  {"display": "NanoBench",    "higher_is_better": True,  "fmt": ".4f"},
}

def cmd_leaderboard(
    lb: CheckpointLeaderboard,
    sort_by: str = "pass_at_1",
    top_n: int = 20,
    dataset_version: str = None,
):
    entries = lb.get_leaderboard(sort_by=sort_by, dataset_version=dataset_version)[:top_n]
    if not entries:
        Console().print("[yellow]No entries in leaderboard.[/]")
        return

    console = Console()
    table = Table(title="[bold]Checkpoint Leaderboard[/]", show_header=True)
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Checkpoint", style="cyan")
    table.add_column("Steps", justify="right")
    table.add_column("Train Loss", justify="right")
    table.add_column("Date", style="dim")

    for mc in METRIC_CONFIG.values():
        table.add_column(mc["display"], justify="right")

    # Find best value for each metric (for coloring)
    metric_attrs = list(METRIC_CONFIG.keys())
    best_vals = {}
    for attr, mc in METRIC_CONFIG.items():
        vals = [getattr(e, attr) for e in entries if getattr(e, attr) is not None]
        if vals:
            best_vals[attr] = max(vals) if mc["higher_is_better"] else min(vals)

    for rank, entry in enumerate(entries, 1):
        metric_cells = []
        for attr, mc in METRIC_CONFIG.items():
            val = getattr(entry, attr)
            if val is None:
                metric_cells.append("[dim]—[/]")
                continue
            formatted = format(val, mc["fmt"])
            is_best = best_vals.get(attr) == val
            color = "bold green" if is_best else (
                "green" if (mc["higher_is_better"] and val >= 0.5)
                or (not mc["higher_is_better"] and val <= 20)
                else "yellow" if (mc["higher_is_better"] and val >= 0.3)
                else "red"
            )
            medal = " 🥇" if rank == 1 and is_best else ""
            metric_cells.append(f"[{color}]{formatted}{medal}[/]")

        table.add_row(
            str(rank),
            entry.checkpoint_name,
            str(entry.training_steps or "—"),
            f"{entry.training_loss:.4f}" if entry.training_loss else "—",
            entry.eval_date,
            *metric_cells,
        )

    console.print(table)

def cmd_compare(lb: CheckpointLeaderboard, name_a: str, name_b: str):
    comparison = lb.compare_two(name_a, name_b)
    console = Console()
    table = Table(title=f"Comparison: {name_a} vs {name_b}")
    table.add_column("Metric")
    table.add_column(name_a, justify="right")
    table.add_column(name_b, justify="right")
    table.add_column("Winner", justify="center")

    for metric, vals in comparison.items():
        mc = METRIC_CONFIG.get(metric, {"higher_is_better": True, "fmt": ".4f"})
        va, vb = vals["a"], vals["b"]
        if va is None and vb is None:
            winner = "—"
        elif va is None:
            winner = f"[green]{name_b}[/]"
        elif vb is None:
            winner = f"[green]{name_a}[/]"
        else:
            if mc["higher_is_better"]:
                winner = f"[green]{name_a if va > vb else name_b}[/]"
            else:
                winner = f"[green]{name_a if va < vb else name_b}[/]"
        table.add_row(
            metric,
            f"{va:{mc.get('fmt', '.4f')}}" if va is not None else "—",
            f"{vb:{mc.get('fmt', '.4f')}}" if vb is not None else "—",
            winner,
        )
    console.print(table)
```

### Step 3: Auto-population from Continuous Eval

In `eval/continuous_eval.py`, after each eval run:

```python
if config.leaderboard.enabled:
    lb = CheckpointLeaderboard(db_path=config.leaderboard.db_path)
    checkpoint_name = Path(checkpoint_path).stem
    lb.register_checkpoint(
        name=checkpoint_name,
        path=str(checkpoint_path),
        training_steps=step,
        training_loss=training_loss,
        dataset_version=config.dataset_version,
    )
    lb.add_evaluation(
        checkpoint_name=checkpoint_name,
        metrics=eval_entry["metrics"],
    )
```

---

## Key Files to Modify

- `tracking/leaderboard.py` - New file: leaderboard DB
- `cli/leaderboard_cmd.py` - New file: CLI
- `cli/main.py` - Register `leaderboard` subcommand
- `eval/continuous_eval.py` - Add auto-populate hook
- `config/training.yaml` - Add `leaderboard` section

---

## Testing Strategy

1. **Register + evaluate test**: register a checkpoint, add evaluation, query leaderboard, assert entry appears.
2. **Sort test**: add 3 evaluations with known `pass_at_1` values [0.1, 0.5, 0.3], assert leaderboard order is [0.5, 0.3, 0.1].
3. **Filter by dataset version**: add entries for two different dataset versions, filter by one, assert only correct entries returned.
4. **Compare test**: add two evaluations for different checkpoints, call `compare_two`, assert all metrics in output.
5. **Duplicate registration test**: register same checkpoint twice with different training loss, assert only one row in `checkpoints` table with updated value.
6. **NULL handling test**: add evaluation with `pass_at_1=None`, assert it appears at bottom of leaderboard (NULL last).

---

## Performance Considerations

- SQLite handles thousands of rows efficiently. Even with 1000 checkpoints and 5 eval runs each = 5000 evaluation rows, queries are fast (<10ms).
- The leaderboard query uses a JOIN between `checkpoints` and `evaluations`. Add an index on `evaluations(checkpoint_id)` for performance at scale.
- Rich table rendering for 50 rows takes ~50ms. For very long leaderboards, add `--top N` option (already implemented).

---

## Dependencies

No new dependencies. Uses `sqlite3` (stdlib) and `rich` (already required).

---

## Estimated Complexity

**Low.** Standard SQLite CRUD with Rich table display. The auto-population hook in continuous eval is one function call. Estimated implementation time: 1-2 days.

---

## 2026 Best Practices

- **Separate leaderboard from experiment tracker (plan 71)**: the experiment tracker (71) tracks training runs with arbitrary metrics; the leaderboard (76) is specifically for ranking *evaluated* checkpoints by standard metrics. They serve different purposes and have different schemas. Share the SQLite infrastructure but keep them as separate tables/databases.
- **Normalize metric names**: always use the same metric key names across all evaluators (`pass_at_1`, `perplexity`, etc.). Inconsistent naming (e.g., `pass@1` vs `pass_at_1`) silently creates null columns in the leaderboard.
- **One evaluation per checkpoint per suite**: prevent duplicate evaluations by inserting only when a (checkpoint_id, eval_suite) pair doesn't already exist, or by using `INSERT OR REPLACE`. Multiple evaluations for the same checkpoint inflate the leaderboard with duplicates.
- **Colored rankings tell a story**: use green/yellow/red coloring based on absolute thresholds (e.g., syntax validity > 70% is green), not just relative to best. This prevents everything looking green when all checkpoints are mediocre.
- **Export for sharing**: add `leaderboard export --output leaderboard.csv` to make it easy to share results. A CSV is the universal format for sharing tabular data.
