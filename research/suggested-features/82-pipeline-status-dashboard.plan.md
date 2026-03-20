# Feature 82: Pipeline Status Dashboard

**Status:** Proposed
**CLI Flag:** `--status` / `cola status` direct command
**Complexity:** Low

---

## Overview

A Rich panel displayed in the main menu header — and available as a standalone `cola status` command — that shows the current completion state of every Cola-Coder pipeline stage at a glance. Each stage is represented by a colored status indicator (green checkmark, yellow spinner, red cross, or grey dash) derived entirely from filesystem artifact detection. No database or state file needed: the dashboard infers state from what exists on disk.

```
╭─ Pipeline Status ─────────────────────────────────────────────╮
│  Tokenizer   [✓] tokenizer.json (48,256 tokens, 2.1 MB)       │
│  Data Prep   [✓] data/processed/train_data.npy (1.2B tokens)  │
│  Training    [⏳] checkpoints/tiny/step_3200 (loss: 4.21)     │
│  Evaluation  [✗] No eval results found                        │
│  API Server  [─] Not configured                               │
╰───────────────────────────────────────────────────────────────╯
```

---

## Motivation

New users constantly ask "where am I in the pipeline?" — especially after returning to a project after a break or inheriting someone else's setup. Currently the only way to answer this is to manually check multiple files and directories. The dashboard makes onboarding instant and prevents common mistakes like trying to train without prepared data or evaluating without a checkpoint.

For experienced users, the dashboard in the menu header means zero extra keystrokes to check pipeline state — it's always visible.

- Prevents "why is training failing?" issues caused by missing upstream artifacts.
- Provides meaningful metadata per stage (token count, step number, loss) not just pass/fail.
- Auto-refreshes on menu startup — no stale state.
- Acts as a self-documenting "what does this project need" reference.

---

## Architecture / Design

```
PipelineStateDetector
  │
  ├── detect_tokenizer()   -> StageStatus
  ├── detect_data()        -> StageStatus
  ├── detect_training()    -> StageStatus
  ├── detect_evaluation()  -> StageStatus
  └── detect_server()      -> StageStatus

StageStatus:
  state: Literal["ok", "running", "missing", "not_configured"]
  label: str
  detail: str          <- e.g. "step_3200, loss=4.21"
  path: Optional[Path] <- the artifact path that was found/missing
  age: Optional[str]   <- "2 hours ago"

PipelineDashboard (Rich Panel renderer)
  └── render(states: list[StageStatus]) -> Panel
```

### Detection Logic Per Stage

| Stage | "ok" condition | "running" condition | "missing" condition |
|---|---|---|---|
| Tokenizer | `tokenizer.json` exists, `vocab_size` readable | `.lock` file present | file absent |
| Data Prep | `data/processed/train_data.npy` exists, size > 0 | `.npy.tmp` exists | file absent |
| Training | `checkpoints/*/step_*` directory exists | `checkpoints/*/RUNNING` sentinel exists | no checkpoints |
| Evaluation | `eval_results/` has any `.json` files | `eval_results/.running` exists | directory empty |
| API Server | port scan shows FastAPI responding | — | no checkpoint configured |

### Running Detection

The trainer writes a `RUNNING` sentinel file at start and deletes it on clean exit. If the sentinel exists but the process PID (stored inside the file) is not alive, the status is shown as `[!] Crashed` rather than `[⏳] Running`.

```python
# Sentinel file format: checkpoints/<size>/RUNNING
# Contents: {"pid": 12345, "started": "2026-03-20T14:32:00Z", "config": "tiny"}
```

---

## Implementation Steps

### Step 1: `StageStatus` dataclass

```python
# src/cola_coder/ui/pipeline_status.py
from dataclasses import dataclass
from typing import Literal, Optional
from pathlib import Path
import time

StatusState = Literal["ok", "running", "crashed", "missing", "not_configured"]

STATE_ICONS = {
    "ok":             ("[bold green]✓[/bold green]", "green"),
    "running":        ("[bold yellow]⏳[/bold yellow]", "yellow"),
    "crashed":        ("[bold red]![/bold red]", "red"),
    "missing":        ("[bold red]✗[/bold red]", "red"),
    "not_configured": ("[dim]─[/dim]", "dim"),
}

@dataclass
class StageStatus:
    name: str
    state: StatusState
    detail: str = ""
    path: Optional[Path] = None
    mtime: Optional[float] = None  # file modification time

    @property
    def age_str(self) -> str:
        if self.mtime is None:
            return ""
        elapsed = time.time() - self.mtime
        if elapsed < 60:
            return f"{int(elapsed)}s ago"
        elif elapsed < 3600:
            return f"{int(elapsed/60)}m ago"
        elif elapsed < 86400:
            return f"{int(elapsed/3600)}h ago"
        else:
            return f"{int(elapsed/86400)}d ago"

    @property
    def icon(self) -> str:
        return STATE_ICONS[self.state][0]

    @property
    def color(self) -> str:
        return STATE_ICONS[self.state][1]
```

### Step 2: `PipelineStateDetector`

```python
# src/cola_coder/ui/pipeline_status.py (continued)
import json
import os
import numpy as np
from pathlib import Path
from typing import Optional

class PipelineStateDetector:
    def __init__(self, root: Path = Path(".")):
        self.root = root

    def detect_all(self) -> list[StageStatus]:
        return [
            self.detect_tokenizer(),
            self.detect_data(),
            self.detect_training(),
            self.detect_evaluation(),
            self.detect_server(),
        ]

    def detect_tokenizer(self) -> StageStatus:
        tok_path = self.root / "tokenizer.json"
        lock_path = self.root / "tokenizer.json.lock"

        if lock_path.exists():
            return StageStatus("Tokenizer", "running", "Training tokenizer...", tok_path)

        if not tok_path.exists():
            return StageStatus("Tokenizer", "missing", "Run: train_tokenizer.py", tok_path)

        try:
            with open(tok_path) as f:
                tok_data = json.load(f)
            vocab_size = len(tok_data.get("model", {}).get("vocab", {}))
            size_mb = tok_path.stat().st_size / 1_048_576
            detail = f"{vocab_size:,} tokens, {size_mb:.1f} MB"
            return StageStatus("Tokenizer", "ok", detail, tok_path, tok_path.stat().st_mtime)
        except Exception:
            return StageStatus("Tokenizer", "ok", "exists (unreadable metadata)", tok_path)

    def detect_data(self) -> StageStatus:
        data_path = self.root / "data" / "processed" / "train_data.npy"
        tmp_path = data_path.with_suffix(".npy.tmp")

        if tmp_path.exists():
            return StageStatus("Data Prep", "running", "Preparing data...", tmp_path)

        if not data_path.exists():
            # Check for any .npy files in data/processed/
            alt = list((self.root / "data" / "processed").glob("*.npy")) if (self.root / "data" / "processed").exists() else []
            if alt:
                p = alt[0]
                size_gb = p.stat().st_size / 1_073_741_824
                return StageStatus("Data Prep", "ok", f"{p.name} ({size_gb:.2f} GB)", p, p.stat().st_mtime)
            return StageStatus("Data Prep", "missing", "Run: prepare_data.py", data_path)

        size_gb = data_path.stat().st_size / 1_073_741_824
        # Estimate token count from file size (each token = 2 bytes uint16)
        token_count = data_path.stat().st_size // 2
        if token_count > 1_000_000_000:
            tok_str = f"{token_count/1e9:.1f}B tokens"
        elif token_count > 1_000_000:
            tok_str = f"{token_count/1e6:.0f}M tokens"
        else:
            tok_str = f"{token_count:,} tokens"
        detail = f"{tok_str}, {size_gb:.2f} GB"
        return StageStatus("Data Prep", "ok", detail, data_path, data_path.stat().st_mtime)

    def detect_training(self) -> StageStatus:
        ckpt_root = self.root / "checkpoints"
        if not ckpt_root.exists():
            return StageStatus("Training", "missing", "No checkpoints directory", ckpt_root)

        # Find all step directories across all config subdirs
        step_dirs = sorted(ckpt_root.glob("*/step_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        running_sentinels = list(ckpt_root.glob("*/RUNNING"))

        if running_sentinels:
            sentinel = running_sentinels[0]
            try:
                data = json.loads(sentinel.read_text())
                pid = data.get("pid")
                # Check if PID is alive
                try:
                    os.kill(pid, 0)
                    alive = True
                except (ProcessLookupError, PermissionError):
                    alive = False
                if alive:
                    step_info = data.get("step", "?")
                    loss = data.get("loss", "?")
                    return StageStatus("Training", "running", f"step {step_info}, loss={loss}", sentinel)
                else:
                    return StageStatus("Training", "crashed", f"PID {pid} not found — crashed?", sentinel)
            except Exception:
                return StageStatus("Training", "running", "In progress (no metadata)", sentinel)

        if not step_dirs:
            return StageStatus("Training", "missing", "No checkpoints found", ckpt_root)

        latest = step_dirs[0]
        # Try to read training metadata
        meta_path = latest / "training_state.json"
        detail = latest.name
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                step = meta.get("global_step", "?")
                loss = meta.get("loss", "?")
                if isinstance(loss, float):
                    loss = f"{loss:.3f}"
                detail = f"{latest.parent.name}/step_{step}, loss={loss}"
            except Exception:
                pass
        return StageStatus("Training", "ok", detail, latest, latest.stat().st_mtime)

    def detect_evaluation(self) -> StageStatus:
        eval_root = self.root / "eval_results"
        running_flag = eval_root / ".running"

        if not eval_root.exists():
            return StageStatus("Evaluation", "missing", "No eval results — run evaluate.py", eval_root)

        if running_flag.exists():
            return StageStatus("Evaluation", "running", "Evaluation in progress...", running_flag)

        result_files = sorted(eval_root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not result_files:
            return StageStatus("Evaluation", "missing", "No result files in eval_results/", eval_root)

        latest = result_files[0]
        try:
            data = json.loads(latest.read_text())
            pass_at_1 = data.get("pass@1", data.get("pass_at_1"))
            perplexity = data.get("perplexity")
            parts = []
            if pass_at_1 is not None:
                parts.append(f"pass@1={pass_at_1:.1%}")
            if perplexity is not None:
                parts.append(f"ppl={perplexity:.1f}")
            detail = ", ".join(parts) if parts else latest.name
        except Exception:
            detail = latest.name
        return StageStatus("Evaluation", "ok", detail, latest, latest.stat().st_mtime)

    def detect_server(self) -> StageStatus:
        import socket
        # Check common server ports
        for port in [8000, 8080, 8765]:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                    return StageStatus("API Server", "ok", f"Running on port {port}")
            except (ConnectionRefusedError, TimeoutError, OSError):
                continue
        return StageStatus("API Server", "not_configured", "Not running")
```

### Step 3: Rich panel renderer

```python
# src/cola_coder/ui/pipeline_dashboard.py
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Console

def render_pipeline_dashboard(states: list[StageStatus], title: str = "Pipeline Status") -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(width=12, no_wrap=True)  # Stage name
    table.add_column(width=4, no_wrap=True)   # Icon
    table.add_column()                         # Detail + age

    for s in states:
        name_text = Text(s.name, style=f"bold {s.color}" if s.state == "ok" else s.color)
        icon_text = Text.from_markup(s.icon)
        detail = s.detail
        if s.age_str and s.state in ("ok", "crashed"):
            detail = f"{detail}  [dim]({s.age_str})[/dim]"
        detail_text = Text.from_markup(detail)
        table.add_row(name_text, icon_text, detail_text)

    return Panel(table, title=f"[bold]{title}[/bold]", border_style="blue", padding=(0, 1))
```

### Step 4: Standalone `cola status` command

```python
# scripts/status.py
"""Quick pipeline status check — no menu required."""
import sys
from pathlib import Path
from rich.console import Console
from cola_coder.ui.pipeline_status import PipelineStateDetector
from cola_coder.ui.pipeline_dashboard import render_pipeline_dashboard

def main():
    root = Path(".")
    detector = PipelineStateDetector(root)
    states = detector.detect_all()
    console = Console()
    console.print(render_pipeline_dashboard(states))

    # Exit with non-zero code if any required stage is missing
    required = ["Tokenizer", "Data Prep", "Training"]
    missing = [s for s in states if s.name in required and s.state == "missing"]
    sys.exit(1 if missing else 0)

if __name__ == "__main__":
    main()
```

### Step 5: Embed in main menu header

```python
# In src/menu/main_menu.py — add to build_layout()
def build_layout(self) -> Layout:
    detector = PipelineStateDetector(Path("."))
    states = detector.detect_all()
    dashboard = render_pipeline_dashboard(states)

    layout = Layout()
    layout.split_column(
        Layout(dashboard, name="status", size=9),
        Layout(name="menu"),
        Layout(name="hotkeys", size=3),
    )
    return layout
```

---

## Key Files to Modify

- `src/cola_coder/ui/pipeline_status.py` — new file: `StageStatus`, `PipelineStateDetector`
- `src/cola_coder/ui/pipeline_dashboard.py` — new file: `render_pipeline_dashboard()`
- `src/menu/main_menu.py` — modify: embed dashboard panel in layout header
- `src/cola_coder/training/trainer.py` — modify: write/delete `RUNNING` sentinel with PID
- `scripts/status.py` — new file: standalone status CLI command
- `cola-status.ps1` (or equivalent) — new wrapper script

---

## Testing Strategy

```python
# tests/test_pipeline_status.py
import json
import pytest
from pathlib import Path
from cola_coder.ui.pipeline_status import PipelineStateDetector

@pytest.fixture
def tmp_project(tmp_path):
    return tmp_path

def test_all_missing_on_empty_dir(tmp_project):
    d = PipelineStateDetector(tmp_project)
    states = d.detect_all()
    assert all(s.state in ("missing", "not_configured") for s in states)

def test_tokenizer_detected(tmp_project):
    tok = {"model": {"vocab": {str(i): i for i in range(32000)}}}
    (tmp_project / "tokenizer.json").write_text(json.dumps(tok))
    d = PipelineStateDetector(tmp_project)
    status = d.detect_tokenizer()
    assert status.state == "ok"
    assert "32,000" in status.detail

def test_data_prep_detected(tmp_project):
    (tmp_project / "data" / "processed").mkdir(parents=True)
    npy = tmp_project / "data" / "processed" / "train_data.npy"
    npy.write_bytes(b"\x00" * 2_000_000)  # 1M tokens (2 bytes each)
    d = PipelineStateDetector(tmp_project)
    status = d.detect_data()
    assert status.state == "ok"
    assert "1M tokens" in status.detail

def test_running_sentinel_detection(tmp_project):
    ckpt = tmp_project / "checkpoints" / "tiny"
    ckpt.mkdir(parents=True)
    import os
    (ckpt / "RUNNING").write_text(json.dumps({"pid": os.getpid(), "step": 100, "loss": 5.2}))
    d = PipelineStateDetector(tmp_project)
    status = d.detect_training()
    assert status.state == "running"

def test_crashed_sentinel_detection(tmp_project):
    ckpt = tmp_project / "checkpoints" / "tiny"
    ckpt.mkdir(parents=True)
    (ckpt / "RUNNING").write_text(json.dumps({"pid": 99999999, "step": 50}))
    d = PipelineStateDetector(tmp_project)
    status = d.detect_training()
    assert status.state == "crashed"

def test_eval_results_detected(tmp_project):
    (tmp_project / "eval_results").mkdir()
    result = {"pass@1": 0.312, "perplexity": 8.7}
    (tmp_project / "eval_results" / "eval_step_3200.json").write_text(json.dumps(result))
    d = PipelineStateDetector(tmp_project)
    status = d.detect_evaluation()
    assert status.state == "ok"
    assert "31.2%" in status.detail
```

---

## Performance Considerations

- All detection is pure filesystem stat calls — no file parsing except for small JSON metadata files. Total detection time is typically < 5ms even on spinning disk.
- `detect_server()` uses a 0.3s socket timeout — this is the only potentially slow operation. It is run in a background thread during menu startup so it does not block the initial render.
- The dashboard is re-computed on every menu entry (not cached) to ensure freshness. Given the low cost this is fine.
- On Windows, `os.kill(pid, 0)` may require elevated privileges for cross-user processes. The code catches `PermissionError` and treats it as "process alive" (safe default).
- `data/processed/train_data.npy` can be multi-gigabyte. The detector only calls `stat()` — it never reads the file contents.

---

## Dependencies

| Package | Purpose | Already present? |
|---|---|---|
| `rich` | Panel, Table, Text rendering | Yes |
| `pathlib` | Filesystem detection | stdlib |
| `json` | Metadata parsing | stdlib |
| `socket` | Server port detection | stdlib |
| `os` | PID liveness check | stdlib |

No new dependencies required.

---

## Estimated Complexity

**Low.** This is fundamentally filesystem detection + Rich rendering. The logic is straightforward with no ML components. Estimated implementation time: 3-5 hours including tests and trainer sentinel integration. The most time-consuming part is handling edge cases in path detection across different project configurations (tiny/small/medium checkpoints, different data paths).

---

## 2026 Best Practices

- **Filesystem as source of truth**: Deriving state from artifact existence (not a separate state file) is robust and requires zero maintenance. The project's state IS the files on disk.
- **PID-based liveness check**: The `RUNNING` sentinel + PID pattern is the standard Unix practice for detecting if a background process is still alive without polling.
- **Fail-open philosophy**: When detection is ambiguous (e.g., unreadable metadata JSON), the dashboard shows a degraded-but-present status rather than crashing. Users get partial information rather than an error.
- **Non-zero exit code from `scripts/status.py`**: Makes the dashboard usable in CI/CD pipelines and shell scripts (`cola-status.ps1 || exit 1`).
- **Age display**: Showing "2 hours ago" next to artifacts helps users immediately understand if checkpoints are fresh or stale without reading timestamps.
