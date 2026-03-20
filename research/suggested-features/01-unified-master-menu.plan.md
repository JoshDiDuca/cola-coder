# 01 - Unified Master Menu (cola.ps1)

## Overview

Replace the current collection of 12+ separate PowerShell scripts (`cola-train.ps1`, `cola-generate.ps1`, `cola-prepare.ps1`, etc.) with a single entry point `cola.ps1` that launches a Rich-based interactive terminal menu system. The menu delegates to Python subcommands while providing a unified, discoverable interface for all Cola-Coder operations.

This is the **foundational feature** — once in place, every other feature in this research plan becomes a menu item rather than a new script file.

---

## Motivation

Currently, new users must discover scripts by browsing the directory or reading documentation. Experienced users must remember exact script names and argument flags. As Cola-Coder grows, the number of scripts proliferates, flags diverge across scripts, and there is no single place to see what the project can do.

A unified menu solves:
- **Discoverability**: All capabilities visible in one place
- **Consistency**: Shared argument handling, shared config loading, shared GPU detection
- **Extensibility**: Adding a new feature means adding a menu item, not a new script
- **Status at a glance**: The menu can show GPU state, last run, active checkpoint

---

## Architecture / Design

### Entry Point Flow

```
cola.ps1
  └─> python src/menu/main_menu.py [--direct <command> [args]]
        ├─> MainMenu (Rich interactive)
        │     ├─> TrainMenu
        │     ├─> PrepareMenu
        │     ├─> GenerateMenu
        │     ├─> EvaluateMenu
        │     ├─> ServeMenu
        │     ├─> ReasonMenu
        │     └─> ToolsMenu
        └─> DirectDispatch (non-interactive, for scripting)
```

### Menu Hierarchy

```
[Cola-Coder]
├── [T] Train
│     ├── [1] Start New Training Run
│     ├── [2] Resume Latest Checkpoint       <- feature 09
│     ├── [3] Resume Specific Checkpoint
│     ├── [4] Configure Training Params
│     ├── [5] Estimate VRAM Usage            <- feature 07
│     └── [6] Validate Config               <- feature 08
├── [P] Prepare
│     ├── [1] Prepare Data (with auto-split) <- feature 04
│     ├── [2] Inspect Dataset Stats
│     └── [3] Rebuild Tokenizer
├── [G] Generate
│     ├── [1] Interactive Prompt
│     ├── [2] Batch Generate from File
│     └── [3] Quick Smoke Test              <- feature 03
├── [E] Evaluate
│     ├── [1] Nano Benchmark                <- feature 02
│     ├── [2] Perplexity on Val Set         <- feature 12
│     └── [3] A/B Checkpoint Compare        <- feature 15
├── [S] Serve
│     ├── [1] Start API Server
│     └── [2] Stop Server
├── [R] Reason
│     └── [1] Reasoning Mode
└── [X] Tools
      ├── [1] View Loss Curve               <- feature 11
      ├── [2] Training Monitor              <- feature 06
      ├── [3] Dead Neuron Detection         <- feature 14
      ├── [4] Gradient Norm Check           <- feature 13
      └── [5] Crash Recovery               <- feature 10
```

### Status Bar

The menu always shows a persistent status bar at the bottom:

```
GPU: RTX 3080 10GB | VRAM: 4.2/10GB | Last Run: train @ step 8400 | Config: small.yaml
```

---

## Implementation Steps

### Step 1: Create the menu package structure

```
src/
  menu/
    __init__.py
    main_menu.py          # Top-level entry, renders root menu
    menus/
      __init__.py
      train_menu.py
      prepare_menu.py
      generate_menu.py
      evaluate_menu.py
      serve_menu.py
      reason_menu.py
      tools_menu.py
    components/
      __init__.py
      status_bar.py       # GPU/training state panel
      breadcrumb.py       # Navigation trail
      action_runner.py    # Runs Python subcommands, streams output
```

### Step 2: Base menu class

```python
# src/menu/base_menu.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from src.cli import header, choose, confirm  # reuse existing utilities

console = Console()

@dataclass
class MenuItem:
    key: str                    # Single char shortcut, e.g. "1", "T"
    label: str                  # Display text
    description: str            # Shown in menu
    action: Callable[[], None]  # What to do when selected
    enabled: bool = True        # Can be toggled

@dataclass
class BaseMenu:
    title: str
    subtitle: str = ""
    items: list[MenuItem] = field(default_factory=list)
    parent: Optional["BaseMenu"] = None

    def render(self) -> None:
        """Render menu items as a Rich table and prompt for selection."""
        while True:
            console.clear()
            header(self.title, self.subtitle)
            self._render_breadcrumb()
            self._render_items_table()
            self._render_status_bar()

            choice = choose(
                "Select option",
                choices=[item.key for item in self.items if item.enabled] + (["B"] if self.parent else ["Q"]),
            )

            if choice == "Q":
                break
            if choice == "B" and self.parent:
                break

            for item in self.items:
                if item.key == choice and item.enabled:
                    item.action()
                    break

    def _render_breadcrumb(self) -> None:
        crumbs = self._collect_breadcrumbs()
        crumb_text = " > ".join(crumbs)
        console.print(f"[dim]{crumb_text}[/dim]")

    def _collect_breadcrumbs(self) -> list[str]:
        crumbs = [self.title]
        menu = self.parent
        while menu:
            crumbs.insert(0, menu.title)
            menu = menu.parent
        return crumbs

    def _render_items_table(self) -> None:
        table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
        table.add_column("Key", style="bold cyan", width=4)
        table.add_column("Label", style="bold white")
        table.add_column("Description", style="dim")
        for item in self.items:
            style = "" if item.enabled else "dim"
            table.add_row(f"[{item.key}]", item.label, item.description, style=style)
        if self.parent:
            table.add_row("[B]", "Back", "Return to previous menu", style="dim")
        else:
            table.add_row("[Q]", "Quit", "Exit Cola-Coder", style="dim")
        console.print(table)

    def _render_status_bar(self) -> None:
        from src.menu.components.status_bar import get_status_bar_text
        console.print(Panel(get_status_bar_text(), style="dim", height=3))
```

### Step 3: Status bar component

```python
# src/menu/components/status_bar.py
import subprocess
import torch
from pathlib import Path
import json

def get_gpu_info() -> str:
    """Query nvidia-smi for VRAM usage."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            name, used, total = parts[0], parts[1], parts[2]
            return f"GPU: {name} | VRAM: {used}/{total}MB"
    except Exception:
        pass
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return f"GPU: {props.name} | VRAM: {props.total_memory // 1024**2}MB total"
    return "GPU: Not available"

def get_last_run_info() -> str:
    """Read last run metadata from training manifest."""
    manifest_path = Path("runs/manifest.json")
    if not manifest_path.exists():
        return "Last Run: None"
    try:
        data = json.loads(manifest_path.read_text())
        runs = data.get("runs", [])
        if not runs:
            return "Last Run: None"
        last = runs[-1]
        step = last.get("last_step", "?")
        name = last.get("name", "unknown")
        return f"Last Run: {name} @ step {step}"
    except Exception:
        return "Last Run: unknown"

def get_active_config() -> str:
    """Detect which config YAML was last used."""
    config_path = Path(".cola_active_config")
    if config_path.exists():
        return f"Config: {config_path.read_text().strip()}"
    return "Config: none"

def get_status_bar_text() -> str:
    parts = [get_gpu_info(), get_last_run_info(), get_active_config()]
    return "  |  ".join(parts)
```

### Step 4: Action runner (streams subprocess output)

```python
# src/menu/components/action_runner.py
import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

def run_python_command(module: str, args: list[str], title: str = "Running") -> int:
    """
    Run a Cola-Coder Python module as a subprocess, streaming output to the terminal.
    Returns exit code.
    """
    cmd = [sys.executable, "-m", module] + args
    console.print(Panel(f"[bold]{title}[/bold]\n[dim]{' '.join(cmd)}[/dim]"))

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            console.print(line, end="")
        proc.wait()
        return proc.returncode
    except KeyboardInterrupt:
        proc.terminate()
        console.print("\n[yellow]Interrupted.[/yellow]")
        return 130
```

### Step 5: cola.ps1 entry point

```powershell
# cola.ps1
param(
    [string]$Command = "",
    [string[]]$Args = @()
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonSrc = Join-Path $ScriptDir "src"
$Env:PYTHONPATH = $ScriptDir

if ($Command -ne "") {
    # Non-interactive direct dispatch for scripting
    python -m src.menu.main_menu --direct $Command @Args
} else {
    # Interactive menu
    python -m src.menu.main_menu
}
```

### Step 6: Direct dispatch for scripting

```python
# src/menu/main_menu.py (excerpt)
import typer
from src.menu.menus.train_menu import TrainMenu
from src.menu.menus.prepare_menu import PrepareMenu
# ... etc

app = typer.Typer(help="Cola-Coder: Code Generation Transformer")

DIRECT_COMMANDS = {
    "train": "src.train",
    "prepare": "src.prepare_data",
    "generate": "src.generate",
    "evaluate": "src.evaluate",
    "benchmark": "src.evaluation.nano_benchmark",
    "smoke": "src.evaluation.smoke_test",
}

@app.command()
def main(direct: str = typer.Option("", help="Run command directly without menu")):
    if direct:
        if direct not in DIRECT_COMMANDS:
            typer.echo(f"Unknown command: {direct}. Valid: {list(DIRECT_COMMANDS.keys())}")
            raise typer.Exit(1)
        import importlib
        mod = importlib.import_module(DIRECT_COMMANDS[direct])
        mod.main()
    else:
        root_menu = build_root_menu()
        root_menu.render()

if __name__ == "__main__":
    app()
```

### Step 7: Wire keyboard shortcuts

Use single-character keys for top-level menus and numeric keys for sub-items. Add `?` globally to show help, and `/` to jump to a search-filtered item list:

```python
# In BaseMenu.render(), extend the choice handling:
if choice == "?":
    self._show_help()
    continue
if choice == "/":
    query = Prompt.ask("Search")
    matches = [i for i in self.items if query.lower() in i.label.lower()]
    if matches:
        matches[0].action()
    continue
```

---

## Key Files to Modify

| File | Change |
|------|--------|
| `cola.ps1` | Replace with thin wrapper calling `src/menu/main_menu.py` |
| `src/cli.py` | Ensure `header()`, `choose()`, `confirm()` are importable as utilities |
| `src/train.py` | Add `main()` function callable from menu |
| `src/prepare_data.py` | Add `main()` function callable from menu |
| `src/generate.py` | Add `main()` function callable from menu |
| `runs/manifest.json` | Add `last_step`, `name` fields for status bar |

New files to create:
- `src/menu/__init__.py`
- `src/menu/main_menu.py`
- `src/menu/base_menu.py`
- `src/menu/components/status_bar.py`
- `src/menu/components/action_runner.py`
- `src/menu/menus/train_menu.py`
- `src/menu/menus/prepare_menu.py`
- `src/menu/menus/generate_menu.py`
- `src/menu/menus/evaluate_menu.py`
- `src/menu/menus/tools_menu.py`

---

## Testing Strategy

- **Smoke test**: Run `python -m src.menu.main_menu --direct train --help` — should print help without entering interactive mode
- **Navigation test**: Manually navigate all menu paths, verify breadcrumb updates correctly
- **Status bar test**: Mock `nvidia-smi` output, verify status bar parses correctly
- **Action runner test**: Run a trivial Python script via `run_python_command`, verify output streams and exit code returns
- **Keyboard shortcut test**: Verify `?`, `/`, `B`, `Q` behave correctly at each level
- **Direct dispatch test**: `cola.ps1 train` should call `src.train.main()` directly

---

## Performance Considerations

- Status bar GPU query uses a 2-second timeout — non-blocking with `subprocess.run(timeout=2)`
- Menu rendering is synchronous; no background threads needed at this layer
- For the training monitor (feature 06), a separate thread is used — that thread is started from within the train action, not the menu
- `console.clear()` on each menu render avoids Rich panel accumulation

---

## Dependencies

| Package | Use | Already installed? |
|---------|-----|-------------------|
| `rich` | All UI rendering | Yes |
| `typer` | CLI argument parsing for direct dispatch | Add to requirements.txt |
| `torch` | GPU detection fallback | Yes |

Install: `pip install typer`

---

## Estimated Complexity

**Medium** — 2-3 days of focused implementation.

- Base menu class and navigation: 4 hours
- Status bar with real GPU info: 2 hours
- All 7 sub-menus wired up: 4 hours
- cola.ps1 integration and direct dispatch: 2 hours
- Testing and polish: 4 hours

Total: ~16 hours

---

## 2026 Best Practices

- **Typer over argparse**: Typer provides automatic `--help`, type validation, and shell completion with minimal boilerplate. Pair with Rich for styled output.
- **Separation of concerns**: Menu code never contains business logic — it only calls into existing modules. This makes menus testable in isolation.
- **Progressive disclosure**: Top-level menu shows 7 options. Sub-menus show 3-6. Never more than 9 items at one level (Miller's Law).
- **Graceful degradation**: If `nvidia-smi` is unavailable, status bar falls back to torch CUDA properties, then to "GPU: Not available". Never crash on missing tools.
- **Non-interactive mode**: Every menu path must be reachable via `cola.ps1 <command>` for CI/scripting use. Interactive menus should never be the only path.
- **Accessibility**: Single-character shortcuts, clear labels, and screen-reader-compatible Rich output (avoid purely decorative Unicode).
