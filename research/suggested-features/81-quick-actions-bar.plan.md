# Feature 81: Quick Actions Bar

**Status:** Proposed
**CLI Flag:** `--quick-actions` / config key `ui.quick_actions: true`
**Complexity:** Low-Medium

---

## Overview

A persistent hotkey bar displayed at the bottom of every Rich menu screen, allowing users to jump directly to common actions without navigating menu hierarchies. Single keystrokes (`g` = generate, `t` = train, `e` = evaluate, `q` = quit, etc.) are captured non-blocking via `readchar` and dispatched immediately from any menu context.

Key bindings are configurable via `configs/ui.yaml` and all bindings are shown as a hint bar so users never need to memorize them.

---

## Motivation

Currently, reaching the generate screen from the main menu requires 2-3 keystrokes. Power users running frequent generate/evaluate cycles during development waste time on repeated navigation. For a developer iteration loop (tweak prompt → generate → evaluate → repeat), having single-key shortcuts cuts the friction dramatically — similar to how Vim-style keymaps work in terminal tools like `lazygit` or `k9s`.

- Reduces menu navigation from 2-3 keys to 1 for the most common operations.
- Makes the CLI feel snappy and professional rather than menu-heavy.
- Customizable bindings respect the fact that power users have preferences.
- Hint bar means zero memorization burden for new users.

---

## Architecture / Design

```
RichMenuBase (base class for all menus)
  │
  ├── render_hotkey_bar()          <- renders hint bar at bottom
  ├── start_hotkey_listener()      <- spawns non-blocking reader thread
  ├── stop_hotkey_listener()       <- cleanup on menu exit
  │
  └── HotkeyDispatcher
        ├── bindings: dict[str, Action]   <- loaded from ui.yaml
        ├── dispatch(key: str) -> bool    <- True if handled
        └── Action: dataclass(label, fn, context_filter)
```

### Layered Binding Resolution

```
Global bindings (always active)          <- g, t, e, q, h
  └── Context bindings (menu-specific)   <- overrides globals for this menu
        └── Modal bindings (dialog open) <- only these active while modal open
```

Context filtering prevents `g` (generate) from triggering while inside the Generate menu's own sub-options, where it would be ambiguous. Each menu declares its active binding contexts.

### Key Detection Strategy

Uses `readchar` (cross-platform, handles Windows `msvcrt` and Unix `termios` transparently). A daemon thread reads keys and puts them on a `queue.Queue`. The main render loop drains the queue between Rich `Live` refresh cycles.

```
Thread A (readchar)      Thread B (Rich Live render)
   │                          │
   │── key event ──> Queue ──>│── dispatch(key)
   │                          │── re-render if needed
```

No busy-waiting: the reader thread blocks on `readchar.readchar()` which is a blocking syscall. The queue is drained every render tick (~100ms).

---

## Implementation Steps

### Step 1: Install and wrap `readchar`

```python
# src/cola_coder/ui/hotkeys.py
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional
import readchar

@dataclass
class HotkeyAction:
    key: str
    label: str
    callback: Callable[[], None]
    contexts: list[str] = field(default_factory=lambda: ["*"])  # "*" = all contexts
    enabled: bool = True

class HotkeyDispatcher:
    def __init__(self):
        self._bindings: dict[str, HotkeyAction] = {}
        self._current_context: str = "global"
        self._key_queue: queue.Queue[str] = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False

    def register(self, action: HotkeyAction) -> None:
        self._bindings[action.key.lower()] = action

    def set_context(self, context: str) -> None:
        self._current_context = context

    def start(self) -> None:
        self._running = True
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="hotkey-reader"
        )
        self._reader_thread.start()

    def stop(self) -> None:
        self._running = False

    def _reader_loop(self) -> None:
        while self._running:
            try:
                key = readchar.readchar()
                self._key_queue.put(key)
            except Exception:
                break

    def drain(self) -> list[str]:
        """Called each render tick to process pending keys."""
        keys = []
        while not self._key_queue.empty():
            try:
                keys.append(self._key_queue.get_nowait())
            except queue.Empty:
                break
        return keys

    def dispatch(self, key: str) -> bool:
        action = self._bindings.get(key.lower())
        if not action or not action.enabled:
            return False
        if "*" not in action.contexts and self._current_context not in action.contexts:
            return False
        action.callback()
        return True
```

### Step 2: Default bindings registry

```python
# src/cola_coder/ui/default_bindings.py
DEFAULT_BINDINGS = [
    {"key": "g", "label": "[g] Generate",  "action": "navigate:generate",  "contexts": ["*"]},
    {"key": "t", "label": "[t] Train",     "action": "navigate:train",     "contexts": ["*"]},
    {"key": "e", "label": "[e] Evaluate",  "action": "navigate:evaluate",  "contexts": ["*"]},
    {"key": "s", "label": "[s] Serve",     "action": "navigate:serve",     "contexts": ["*"]},
    {"key": "p", "label": "[p] Prepare",   "action": "navigate:prepare",   "contexts": ["*"]},
    {"key": "h", "label": "[h] Help",      "action": "show:help",          "contexts": ["*"]},
    {"key": "q", "label": "[q] Quit",      "action": "quit",               "contexts": ["*"]},
    # Context-specific — only active when not already in that menu
    {"key": "r", "label": "[r] Resume",    "action": "train:resume",       "contexts": ["train"]},
    {"key": "b", "label": "[b] Benchmark", "action": "evaluate:benchmark", "contexts": ["evaluate"]},
]
```

### Step 3: Hotkey hint bar renderer

```python
# src/cola_coder/ui/hotkey_bar.py
from rich.text import Text
from rich.panel import Panel
from rich.console import Console

def render_hotkey_bar(
    dispatcher: HotkeyDispatcher,
    context: str,
    width: int = 120,
) -> Panel:
    parts = []
    for key, action in dispatcher._bindings.items():
        if not action.enabled:
            continue
        if "*" not in action.contexts and context not in action.contexts:
            continue
        parts.append(f"[bold cyan]{action.key}[/bold cyan][dim]={action.label.split(']')[1].strip()}[/dim]")

    bar_text = "  ".join(parts)
    return Panel(
        Text.from_markup(bar_text),
        title="[dim]Quick Actions[/dim]",
        border_style="dim",
        height=3,
        padding=(0, 1),
    )
```

### Step 4: Base menu class integration

```python
# src/cola_coder/ui/base_menu.py
from rich.live import Live
from rich.layout import Layout
from contextlib import contextmanager

class RichMenuBase:
    def __init__(self, dispatcher: HotkeyDispatcher):
        self.dispatcher = dispatcher
        self.context = "global"
        self._live: Optional[Live] = None

    def build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="hotkeys", size=3),
        )
        layout["hotkeys"].update(render_hotkey_bar(self.dispatcher, self.context))
        return layout

    @contextmanager
    def run_with_hotkeys(self):
        self.dispatcher.set_context(self.context)
        self.dispatcher.start()
        try:
            with Live(self.build_layout(), refresh_per_second=10) as live:
                self._live = live
                yield live
                # Drain and dispatch keys each tick
                while True:
                    for key in self.dispatcher.drain():
                        handled = self.dispatcher.dispatch(key)
                        if handled:
                            live.update(self.build_layout())
        finally:
            self.dispatcher.stop()
```

### Step 5: Config loading from `configs/ui.yaml`

```yaml
# configs/ui.yaml
ui:
  quick_actions: true
  hotkeys:
    # Override or disable defaults
    g: generate          # default
    t: train             # default
    e: evaluate          # default
    # Disable a default binding:
    # s: null
    # Add a custom binding:
    # c: "navigate:checkpoint_browser"
```

```python
# src/cola_coder/ui/hotkey_config.py
import yaml
from pathlib import Path

def load_hotkey_config(config_path: Path = Path("configs/ui.yaml")) -> dict:
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("ui", {}).get("hotkeys", {})

def build_dispatcher_from_config(config_path: Path) -> HotkeyDispatcher:
    dispatcher = HotkeyDispatcher()
    overrides = load_hotkey_config(config_path)
    for binding in DEFAULT_BINDINGS:
        key = binding["key"]
        # Apply user override
        if key in overrides:
            if overrides[key] is None:
                continue  # User disabled this binding
            binding = {**binding, "action": overrides[key]}
        action = HotkeyAction(
            key=key,
            label=binding["label"],
            callback=lambda a=binding["action"]: dispatch_action(a),
            contexts=binding.get("contexts", ["*"]),
        )
        dispatcher.register(action)
    return dispatcher
```

### Step 6: Integration with cola.ps1 / main_menu.py

```python
# In src/menu/main_menu.py — add to startup
if cfg.get("ui", {}).get("quick_actions", True):
    dispatcher = build_dispatcher_from_config(Path("configs/ui.yaml"))
    dispatcher.register(HotkeyAction(
        key="g",
        label="[g] Generate",
        callback=lambda: navigate_to("generate"),
        contexts=["*"],
    ))
    menu = MainMenu(dispatcher=dispatcher)
else:
    menu = MainMenu(dispatcher=NullDispatcher())
```

---

## Key Files to Modify

- `src/cola_coder/ui/hotkeys.py` — new file: `HotkeyDispatcher`, `HotkeyAction`
- `src/cola_coder/ui/hotkey_bar.py` — new file: `render_hotkey_bar()`
- `src/cola_coder/ui/default_bindings.py` — new file: `DEFAULT_BINDINGS`
- `src/cola_coder/ui/hotkey_config.py` — new file: config loading
- `src/cola_coder/ui/base_menu.py` — modify: integrate dispatcher into base class
- `src/menu/main_menu.py` — modify: pass dispatcher, call `build_layout()`
- `configs/ui.yaml` — new file: UI configuration including hotkey overrides
- `pyproject.toml` — add `readchar>=4.0` to dependencies

---

## Testing Strategy

```python
# tests/test_hotkeys.py
def test_dispatch_registered_key():
    called = []
    dispatcher = HotkeyDispatcher()
    dispatcher.register(HotkeyAction(key="g", label="Generate", callback=lambda: called.append("g")))
    dispatcher.set_context("global")
    result = dispatcher.dispatch("g")
    assert result is True
    assert called == ["g"]

def test_context_filter_blocks_out_of_context_key():
    called = []
    dispatcher = HotkeyDispatcher()
    dispatcher.register(HotkeyAction(
        key="r", label="Resume", callback=lambda: called.append("r"), contexts=["train"]
    ))
    dispatcher.set_context("generate")
    result = dispatcher.dispatch("r")
    assert result is False
    assert called == []

def test_null_override_disables_binding():
    config = {"g": None}
    # Build dispatcher with override
    dispatcher = build_dispatcher_from_config_dict(config)
    assert "g" not in dispatcher._bindings

def test_case_insensitive_dispatch():
    called = []
    dispatcher = HotkeyDispatcher()
    dispatcher.register(HotkeyAction(key="g", label="Generate", callback=lambda: called.append("G")))
    dispatcher.dispatch("G")  # uppercase
    assert called == ["G"]

def test_hotkey_bar_renders_active_context_only():
    dispatcher = HotkeyDispatcher()
    dispatcher.register(HotkeyAction(key="r", label="[r] Resume", callback=lambda: None, contexts=["train"]))
    dispatcher.register(HotkeyAction(key="g", label="[g] Generate", callback=lambda: None, contexts=["*"]))
    dispatcher.set_context("generate")
    panel = render_hotkey_bar(dispatcher, "generate")
    rendered = str(panel.renderable)
    assert "g" in rendered
    assert "r" not in rendered
```

Manual test: run `cola.ps1`, verify hotkey bar visible at bottom of every screen. Press `g` from main menu — should jump to Generate screen directly. Press `q` — should quit cleanly. Edit `configs/ui.yaml` to remap `g` to `null` — `g` should stop working.

---

## Performance Considerations

- The reader thread is a daemon thread — it dies automatically when the main process exits. No cleanup required beyond stopping the `_running` flag.
- `readchar.readchar()` is a blocking syscall with no busy-waiting. CPU overhead is negligible.
- Key queue draining happens at Rich's render tick rate (~10 fps). No key event can pile up faster than the user can type.
- On Windows, `readchar` uses `msvcrt.getwch()` under the hood. This works correctly in Windows Terminal and PowerShell 7 but may behave oddly in legacy cmd.exe (acceptable limitation).
- The hint bar is 3 rows tall — small enough to not crowd the terminal at standard 80-row height.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `readchar` | `>=4.0` | Cross-platform single-key non-blocking reads |
| `rich` | `>=13.0` | Already a dependency — Layout, Panel, Text |
| `pyyaml` | `>=6.0` | Already a dependency — `configs/ui.yaml` loading |
| `threading` | stdlib | Daemon reader thread |
| `queue` | stdlib | Thread-safe key event queue |

No new heavy dependencies. `readchar` is a tiny pure-Python package (~200 lines).

---

## Estimated Complexity

**Low-Medium.** The core dispatcher and reader thread are ~150 lines. The main complexity is integrating cleanly with the existing Rich `Live` render loop without causing flicker or thread-safety issues. Estimated implementation time: 4-6 hours including tests.

Risk: Windows terminal input handling can be quirky for raw key reads. Mitigation: `readchar` 4.x has been tested on Windows Terminal and handles edge cases like arrow keys (multi-byte sequences) gracefully.

---

## 2026 Best Practices

- **Non-blocking I/O threading**: Using `queue.Queue` as the IPC primitive between the reader thread and render thread is the standard Python pattern. Avoids the need for asyncio in a synchronous Rich app.
- **Null object pattern**: `NullDispatcher` for when `--quick-actions` is disabled means zero conditional branches in menu code — clean feature flag.
- **Declarative bindings**: Bindings defined as data (list of dicts in `DEFAULT_BINDINGS`) rather than imperative registrations — easier to override, serialize, and document.
- **Daemon threads**: Reader thread is `daemon=True` so it never blocks process exit — correct 2026 practice for background I/O threads.
- **TOML/YAML config over CLI flags**: Feature is toggled via `configs/ui.yaml` rather than yet another CLI argument, consistent with the project's config-driven approach.
- **Context-aware binding resolution**: Layered context system (global → menu-specific → modal) mirrors modern TUI frameworks like `textual` and prevents binding conflicts as features grow.
