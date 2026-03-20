# 10 - Crash Recovery

## Overview

Ensure that if training crashes (Ctrl+C, OOM error, SIGTERM, power loss simulation, unhandled exception), an emergency checkpoint is saved before the process exits. Uses signal handlers, `atexit`, and `try/except` around the training loop. On next launch, detect the crash checkpoint and offer recovery. Also implements periodic async checkpoint saving in a background thread so that crashes between scheduled saves lose minimal progress.

---

## Motivation

Training a 50M model for 10 hours and then having it crash at step 9500/10000 — without saving — is devastating. It also creates a risk-averse behavior where users save too frequently, slowing training.

The crash recovery system eliminates this risk:
- **Signal handlers** catch Ctrl+C and SIGTERM before the process dies
- **atexit hook** fires even on unhandled exceptions
- **Async periodic saves** keep the loss of progress to at most `async_save_interval` steps
- **Crash markers** make the next launch aware that a crash occurred

---

## Architecture / Design

### Layers of Protection

```
Layer 1: Async periodic save (every N steps, background thread)
  → If crash occurs between steps 1000-1100 and async_save_interval=50,
    lose at most 50 steps of training.

Layer 2: Signal handler (SIGINT, SIGTERM)
  → Intercepts Ctrl+C. Saves emergency checkpoint before exiting.
  → Allows graceful shutdown vs. abrupt kill.

Layer 3: atexit handler
  → Registered at process start. Fires on any exit, including unhandled exceptions.
  → Saves emergency checkpoint if one hasn't been saved yet.

Layer 4: try/except around training loop
  → Catches exceptions, saves checkpoint, re-raises with context.
  → Captures the Python traceback in the checkpoint metadata.
```

### Crash Checkpoint Structure

```
runs/run_001/
  ckpt_002000.pt          <- last scheduled save
  ckpt_CRASH_002437.pt    <- crash checkpoint
  crash_report.json       <- traceback, step, timestamp
```

### Recovery Detection on Next Launch

On the next `cola-train.ps1`:
1. `offer_resume()` (feature 09) scans checkpoints
2. `scan_checkpoints()` identifies files matching `ckpt_CRASH_*.pt`
3. Shows a special "Crash detected" panel with crash info
4. Offers: Resume from crash point / Resume from last clean save / Start Fresh

---

## Implementation Steps

### Step 1: Emergency checkpoint saver

```python
# src/training/crash_recovery.py
import signal
import atexit
import traceback
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
from rich.console import Console

console = Console()

class CrashRecovery:
    """
    Registers signal handlers and atexit to save an emergency checkpoint.
    Usage:
        recovery = CrashRecovery(trainer, run_dir)
        recovery.register()
        # ... training ...
        recovery.deregister()  # Call on clean exit
    """

    def __init__(self, trainer_ref, run_dir: Path):
        self._trainer = trainer_ref
        self._run_dir = run_dir
        self._saved = threading.Event()
        self._atexit_registered = False
        self._original_sigint = None
        self._original_sigterm = None

    def register(self) -> None:
        """Register all crash handlers."""
        # SIGINT (Ctrl+C)
        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_signal)

        # SIGTERM (kill, Docker stop, etc.)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # atexit: fires on any exit including exceptions
        atexit.register(self._atexit_handler)
        self._atexit_registered = True

    def deregister(self) -> None:
        """Call on clean exit to prevent unnecessary crash save."""
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm:
            signal.signal(signal.SIGTERM, self._original_sigterm)
        self._saved.set()  # Prevent atexit from saving

    def _handle_signal(self, signum, frame) -> None:
        sig_name = "SIGINT (Ctrl+C)" if signum == signal.SIGINT else "SIGTERM"
        console.print(f"\n[yellow]Received {sig_name}. Saving emergency checkpoint...[/yellow]")
        self._save_crash_checkpoint(reason=f"signal_{sig_name}")
        # Restore original handler and re-raise
        if signum == signal.SIGINT and self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)
        raise KeyboardInterrupt

    def _atexit_handler(self) -> None:
        """Fire on any exit. Only saves if crash save hasn't happened yet."""
        if not self._saved.is_set():
            self._save_crash_checkpoint(reason="atexit_unhandled_exit")

    def _save_crash_checkpoint(self, reason: str, exc_info=None) -> None:
        if self._saved.is_set():
            return
        self._saved.set()

        try:
            trainer = self._trainer
            step = getattr(trainer, "_current_step", 0)
            loss = getattr(trainer, "_last_loss", None)

            ckpt_name = f"ckpt_CRASH_{step:06d}.pt"
            ckpt_path = self._run_dir / ckpt_name

            import torch
            torch.save({
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "step": step,
                "loss": loss,
                "config": trainer.config,
                "saved_at": datetime.utcnow().isoformat(),
                "crash": True,
                "crash_reason": reason,
                "traceback": traceback.format_exc() if exc_info else None,
            }, ckpt_path)

            # Save crash report
            crash_report = {
                "crash_reason": reason,
                "step": step,
                "loss": loss,
                "timestamp": datetime.utcnow().isoformat(),
                "checkpoint": str(ckpt_path),
                "traceback": traceback.format_exc() if exc_info else None,
            }
            report_path = self._run_dir / "crash_report.json"
            report_path.write_text(json.dumps(crash_report, indent=2))

            console.print(f"[yellow]Emergency checkpoint saved: {ckpt_path}[/yellow]")
        except Exception as e:
            console.print(f"[red]Failed to save emergency checkpoint: {e}[/red]")
```

### Step 2: Training loop integration

```python
# src/trainer.py (additions)
from src.training.crash_recovery import CrashRecovery

class Trainer:
    def train(self, ...):
        recovery = CrashRecovery(self, self._run_dir)
        recovery.register()

        try:
            for step in range(self._start_step, total_steps):
                self._current_step = step  # Keep crash recovery updated

                # ... training step ...
                loss = self._train_step(batch)
                self._last_loss = loss.item()

                # Check for async save
                if self._async_saver.should_save(step):
                    self._async_saver.schedule_save(step, loss.item())

        except KeyboardInterrupt:
            console.print("[yellow]Training interrupted by user.[/yellow]")
            # Emergency checkpoint already saved by signal handler
            raise
        except Exception as e:
            console.print(f"[red]Training crashed: {e}[/red]")
            recovery._save_crash_checkpoint(reason="exception", exc_info=True)
            raise
        else:
            # Clean exit
            recovery.deregister()
            self._save_checkpoint(total_steps, tag="final")
```

### Step 3: Async periodic checkpoint saver

```python
# src/training/async_saver.py
import threading
import queue
from pathlib import Path
from typing import Optional
from rich.console import Console

console = Console()

class AsyncCheckpointSaver:
    """
    Saves checkpoints in a background thread so the training loop is not blocked.
    Uses a queue to pass the model state (a copy) to the save thread.
    """

    def __init__(self, run_dir: Path, save_interval: int = 50):
        self._run_dir = run_dir
        self._interval = save_interval
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._save_loop, daemon=True)
        self._last_save_step = 0
        self._thread.start()

    def should_save(self, step: int) -> bool:
        return step > 0 and step - self._last_save_step >= self._interval

    def schedule_save(self, step: int, loss: float, model_state: dict, optimizer_state: dict, config: dict) -> None:
        """
        Enqueue a save operation. Makes a copy of state dicts to avoid
        modifying them while the training loop continues.
        """
        import copy
        try:
            self._queue.put_nowait({
                "step": step,
                "loss": loss,
                "model_state_dict": copy.deepcopy(model_state),
                "optimizer_state_dict": copy.deepcopy(optimizer_state),
                "config": config,
            })
            self._last_save_step = step
        except queue.Full:
            # Previous save still in progress; skip this one
            console.print(f"[dim]Async save at step {step} skipped (saver busy)[/dim]")

    def _save_loop(self) -> None:
        import torch
        from datetime import datetime
        while not self._stop.wait(0.1):
            try:
                data = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                path = self._run_dir / f"ckpt_{data['step']:06d}.pt"
                data["saved_at"] = datetime.utcnow().isoformat()
                torch.save(data, path)
                console.print(f"[dim]Checkpoint saved: {path.name}[/dim]")
            except Exception as e:
                console.print(f"[red]Async save failed at step {data['step']}: {e}[/red]")

    def stop(self) -> None:
        """Wait for pending saves to complete, then stop."""
        self._stop.set()
        self._thread.join(timeout=30)
```

### Step 4: Crash detection in resume menu

```python
# src/training/resume/scanner.py (addition)
def find_crash_checkpoints(runs_dir: Path = Path("runs")) -> list[CheckpointInfo]:
    """Return checkpoints matching the CRASH naming pattern."""
    infos = []
    for path in sorted(runs_dir.rglob("ckpt_CRASH_*.pt")):
        info = _load_checkpoint_info(path)
        if info:
            infos.append(info)
    return infos

def load_crash_report(run_dir: Path) -> Optional[dict]:
    report_path = run_dir / "crash_report.json"
    if report_path.exists():
        import json
        return json.loads(report_path.read_text())
    return None
```

```python
# src/training/resume/menu.py (additions to offer_resume)
from rich.panel import Panel

def show_crash_banner(crash_info: CheckpointInfo) -> None:
    report = load_crash_report(crash_info.path.parent)
    reason = report.get("crash_reason", "unknown") if report else "unknown"
    tb_preview = ""
    if report and report.get("traceback"):
        tb_lines = report["traceback"].strip().split("\n")
        tb_preview = "\n" + "\n".join(tb_lines[-3:])  # Last 3 lines

    console.print(Panel(
        f"[bold yellow]Previous training run crashed![/bold yellow]\n\n"
        f"  Checkpoint: {crash_info.path.name}\n"
        f"  Step:       {crash_info.step:,}\n"
        f"  Reason:     {reason}\n"
        f"  Time:       {crash_info.saved_at or '?'}"
        + tb_preview,
        title="Crash Detected",
        border_style="yellow",
    ))
```

### Step 5: config.yaml additions

```yaml
training:
  # Crash recovery
  async_save_interval: 50        # Save checkpoint every N steps (background thread)
  crash_recovery_enabled: true   # Register signal handlers
```

---

## Key Files to Modify

| File | Change |
|------|--------|
| `src/training/crash_recovery.py` | New: CrashRecovery with signal handlers + atexit |
| `src/training/async_saver.py` | New: AsyncCheckpointSaver background thread |
| `src/trainer.py` | Register CrashRecovery, integrate AsyncCheckpointSaver |
| `src/training/resume/scanner.py` | Add `find_crash_checkpoints()` |
| `src/training/resume/menu.py` | Show crash banner when crash checkpoint found |
| `src/checkpoint.py` | Add `crash=True` field to checkpoint metadata |
| `configs/*.yaml` | Add `training.async_save_interval` |

---

## Testing Strategy

- **SIGINT test**: Start a training loop in a subprocess, send SIGINT, verify crash checkpoint saved
- **SIGTERM test**: Same but with SIGTERM
- **Exception test**: Raise a RuntimeError inside the training loop, verify emergency checkpoint saved and crash_report.json written
- **Async saver test**: Enqueue 3 save operations, verify all 3 files appear in run_dir
- **Async saver queue full test**: Enqueue faster than save completes, verify "skipped" message, no hang
- **Crash detection test**: Create a checkpoint named `ckpt_CRASH_001234.pt`, verify `find_crash_checkpoints()` finds it
- **Clean exit test**: Normal training completion → `deregister()` called → atexit does NOT save crash checkpoint

---

## Performance Considerations

- Signal handlers execute on the main thread — they must be fast. The actual checkpoint save is done synchronously in the handler (can't use the async saver here since we're about to exit)
- `copy.deepcopy(model_state_dict)` for async saves copies ~200MB for a 50M model. At bf16, this is ~100MB, taking ~50ms on a fast machine. Schedule this copy only every `async_save_interval` steps (default: every 50 steps)
- The async saver thread uses `torch.save()` which writes sequentially. A 200MB checkpoint takes 1-2 seconds. The thread can keep up with the default 50-step interval (assuming steps take >2 seconds total, which is true for batch_size >= 4)
- atexit handlers can be slow — if the main process receives SIGKILL (not SIGTERM), atexit does not fire. This is unavoidable. The signal handler covers SIGINT and SIGTERM.

---

## Dependencies

| Module | Use | Source |
|--------|-----|--------|
| `signal` | SIGINT/SIGTERM handlers | Python stdlib |
| `atexit` | Exit hook | Python stdlib |
| `threading` | Async saver thread | Python stdlib |
| `queue` | Producer-consumer for saves | Python stdlib |
| `torch` | Checkpoint serialization | Already installed |
| `rich` | Console output | Already installed |

No new packages required.

---

## Estimated Complexity

**Medium** — 1.5 days.

- CrashRecovery signal/atexit handlers: 3 hours
- AsyncCheckpointSaver thread: 3 hours
- Trainer integration: 2 hours
- Scanner + menu crash detection: 1.5 hours
- Testing (especially concurrency): 4 hours

Total: ~13.5 hours

---

## 2026 Best Practices

- **atexit + signal handlers, not just one**: Signal handlers catch keyboard interrupts and kill signals. `atexit` catches unhandled Python exceptions. Both are needed for full coverage.
- **deepcopy for async saves**: The training loop mutates model parameters continuously. If you pass the model reference to the save thread, the saved weights will be corrupted. Always deep-copy the state dict before handing off to another thread.
- **Atomic writes**: Write checkpoint to `.pt.tmp` first, then `os.rename()` to `.pt`. `rename()` is atomic on most filesystems — a crash during write leaves a `.tmp` file, not a corrupted `.pt` file.
- **SIGKILL is not catchable**: Document clearly that `kill -9 <pid>` or a power loss cannot be gracefully handled. The async periodic saves are the safety net for these cases.
- **Crash report with full traceback**: When saving a crash checkpoint, always capture `traceback.format_exc()`. The user needs to know why the crash happened to decide whether to resume.
