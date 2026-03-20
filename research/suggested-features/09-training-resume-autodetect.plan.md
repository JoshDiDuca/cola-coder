# 09 - Training Resume Autodetect

## Overview

When launching a training run, automatically scan the checkpoint directories to find the latest saved checkpoint and offer to resume from it. Shows a formatted summary of available checkpoints (step, loss, timestamp, config), offers three options (Resume / Start Fresh / Pick Checkpoint), and validates that the checkpoint's model architecture matches the current config before loading. Handles optimizer state loading gracefully.

---

## Motivation

Currently, resuming training requires manually specifying a checkpoint path via a command-line flag. This means:
- Users forget to check if a prior run exists and accidentally start fresh, overwriting progress
- The correct path must be looked up by browsing the `runs/` directory
- There's no warning if the model config has changed since the checkpoint

The autodetect feature makes resuming the default, safe behavior. The user sees what exists and makes an informed choice before any training begins.

---

## Architecture / Design

### Checkpoint Discovery

```
runs/
  run_001_small_20260320/
    ckpt_000500.pt         <- step 500
    ckpt_001000.pt         <- step 1000
    ckpt_002000.pt         <- LATEST
    manifest.json
  run_002_medium_20260321/
    ckpt_000200.pt
    manifest.json
```

The autodetect scans all `runs/*/ckpt_*.pt` files, reads their embedded metadata, and presents the latest checkpoint from the most recent run.

### Resume Decision Flow

```
cola-train.ps1 invoked
       │
       ▼
Scan runs/ for checkpoints
       │
  Found?
  ├── No  → Start Fresh
  └── Yes → Show resume menu:
              [R] Resume latest (run_002 @ step 200, loss 2.341)
              [F] Start Fresh
              [P] Pick checkpoint (show full list)
              │
              └─ Selected checkpoint
                     │
                     ▼
              Validate architecture compatibility
                     │
                  OK?
                  ├── Yes → Load checkpoint, continue training
                  └── No  → Show incompatibility detail, offer:
                              [S] Start Fresh (keep old run)
                              [Q] Quit
```

### Architecture Compatibility Check

The checkpoint stores the config used to create the model. Before resuming, compare:
- `dim`
- `n_heads`
- `n_kv_heads`
- `n_layers`
- `vocab_size`
- `max_seq_len`

These fields must match exactly. Mismatches in `learning_rate`, `batch_size`, or `dropout` are allowed (these are training hyperparameters, not model architecture).

---

## Implementation Steps

### Step 1: Checkpoint metadata embedded at save time

```python
# src/checkpoint.py (additions to existing save logic)
import torch
import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

def save_checkpoint(
    model,
    optimizer,
    step: int,
    loss: float,
    config: dict,
    run_dir: Path,
    tag: str = None,
) -> Path:
    """Save checkpoint with embedded metadata."""
    if tag:
        name = f"ckpt_{tag}.pt"
    else:
        name = f"ckpt_{step:06d}.pt"

    checkpoint_path = run_dir / name
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss,
        "config": config,                       # Full config dict embedded
        "saved_at": datetime.utcnow().isoformat(),
        "cola_version": "0.1.0",
    }, checkpoint_path)
    return checkpoint_path
```

### Step 2: Checkpoint scanner

```python
# src/training/resume/scanner.py
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import torch
import re

@dataclass
class CheckpointInfo:
    path: Path
    step: int
    loss: Optional[float]
    saved_at: Optional[str]
    run_name: str
    model_config: dict    # Architecture-relevant fields only
    has_optimizer_state: bool

    @property
    def display_name(self) -> str:
        loss_str = f"loss={self.loss:.4f}" if self.loss is not None else "loss=?"
        time_str = self.saved_at[:16] if self.saved_at else "?"
        return f"{self.run_name} @ step {self.step:,}  |  {loss_str}  |  {time_str}"


def scan_checkpoints(runs_dir: Path = Path("runs")) -> list[CheckpointInfo]:
    """Scan all runs/ subdirectories for checkpoints, return sorted by step (newest last)."""
    if not runs_dir.exists():
        return []

    infos = []
    for ckpt_path in sorted(runs_dir.rglob("ckpt_*.pt")):
        info = _load_checkpoint_info(ckpt_path)
        if info:
            infos.append(info)

    # Sort by (run modification time, step) so newest run's latest step is last
    infos.sort(key=lambda c: (c.path.parent.stat().st_mtime, c.step))
    return infos


def _load_checkpoint_info(path: Path) -> Optional[CheckpointInfo]:
    """Load just the metadata from a checkpoint (no model weights loaded into memory)."""
    try:
        # Use map_location="cpu" and weights_only=False to get metadata
        # weights_only=True would be safer but doesn't load our custom metadata
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None

    # Extract step from filename as fallback
    step = ckpt.get("step", 0)
    if step == 0:
        match = re.search(r"ckpt_(\d+)", path.name)
        if match:
            step = int(match.group(1))

    config = ckpt.get("config", {})
    arch_keys = ["dim", "n_heads", "n_kv_heads", "n_layers", "vocab_size", "max_seq_len"]
    model_config = {k: config.get("model", {}).get(k) for k in arch_keys}

    return CheckpointInfo(
        path=path,
        step=step,
        loss=ckpt.get("loss"),
        saved_at=ckpt.get("saved_at"),
        run_name=path.parent.name,
        model_config=model_config,
        has_optimizer_state="optimizer_state_dict" in ckpt,
    )
```

### Step 3: Architecture compatibility check

```python
# src/training/resume/compatibility.py
from dataclasses import dataclass
from src.training.resume.scanner import CheckpointInfo

ARCHITECTURE_KEYS = ["dim", "n_heads", "n_kv_heads", "n_layers", "vocab_size", "max_seq_len"]

@dataclass
class CompatibilityResult:
    compatible: bool
    mismatches: list[tuple[str, any, any]]  # (field, checkpoint_val, config_val)

    def format_mismatches(self) -> str:
        lines = []
        for field, ckpt_val, cfg_val in self.mismatches:
            lines.append(f"  {field}: checkpoint={ckpt_val}, config={cfg_val}")
        return "\n".join(lines)

def check_architecture_compatibility(
    checkpoint: CheckpointInfo,
    current_config: dict,
) -> CompatibilityResult:
    """Check if checkpoint model architecture matches current config."""
    mismatches = []
    current_model = current_config.get("model", {})

    for key in ARCHITECTURE_KEYS:
        ckpt_val = checkpoint.model_config.get(key)
        cfg_val = current_model.get(key)
        if ckpt_val is None or cfg_val is None:
            continue  # Can't check if either side lacks the key
        if ckpt_val != cfg_val:
            mismatches.append((key, ckpt_val, cfg_val))

    return CompatibilityResult(compatible=len(mismatches) == 0, mismatches=mismatches)
```

### Step 4: Interactive resume menu

```python
# src/training/resume/menu.py
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from src.training.resume.scanner import scan_checkpoints, CheckpointInfo
from src.training.resume.compatibility import check_architecture_compatibility
from src.cli import choose, confirm, header

console = Console()

def offer_resume(current_config: dict, runs_dir: Path = Path("runs")) -> Optional[CheckpointInfo]:
    """
    Scan for checkpoints, offer resume choice to user.
    Returns selected CheckpointInfo if user wants to resume, None to start fresh.
    """
    checkpoints = scan_checkpoints(runs_dir)
    if not checkpoints:
        return None  # No checkpoints found, start fresh

    latest = checkpoints[-1]  # Most recent

    header("Resume Training?")
    console.print(Panel(
        f"[cyan]Latest checkpoint found:[/cyan]\n\n"
        f"  Run:    {latest.run_name}\n"
        f"  Step:   {latest.step:,}\n"
        f"  Loss:   {latest.loss:.4f if latest.loss else '?'}\n"
        f"  Saved:  {latest.saved_at or '?'}\n"
        f"  Optimizer state: {'yes' if latest.has_optimizer_state else 'no (will restart optimizer)'}",
        title="Checkpoint Found",
        border_style="cyan",
    ))

    choice = choose(
        "What would you like to do?",
        choices={
            "R": f"Resume from step {latest.step:,}",
            "F": "Start Fresh (new run)",
            "P": "Pick a specific checkpoint",
            "Q": "Cancel",
        }
    )

    if choice == "Q":
        raise SystemExit(0)
    if choice == "F":
        return None
    if choice == "P":
        return _pick_checkpoint_menu(checkpoints, current_config)
    # choice == "R"
    return _validate_and_return(latest, current_config)

def _pick_checkpoint_menu(
    checkpoints: list[CheckpointInfo],
    current_config: dict,
) -> Optional[CheckpointInfo]:
    """Show all checkpoints in a table, let user pick one."""
    table = Table(title="All Checkpoints", box=box.ROUNDED)
    table.add_column("#", width=4)
    table.add_column("Run", style="cyan")
    table.add_column("Step", justify="right")
    table.add_column("Loss", justify="right")
    table.add_column("Saved At")
    table.add_column("Compat", justify="center")

    for i, ckpt in enumerate(reversed(checkpoints), 1):
        compat = check_architecture_compatibility(ckpt, current_config)
        compat_str = "[green]OK[/green]" if compat.compatible else "[red]MISMATCH[/red]"
        table.add_row(
            str(i),
            ckpt.run_name[:25],
            f"{ckpt.step:,}",
            f"{ckpt.loss:.4f}" if ckpt.loss else "?",
            ckpt.saved_at[:16] if ckpt.saved_at else "?",
            compat_str,
        )
    console.print(table)

    index_str = choose("Enter checkpoint number (or B to go back)", choices=[str(i) for i in range(1, len(checkpoints)+1)] + ["B"])
    if index_str == "B":
        return None
    selected = list(reversed(checkpoints))[int(index_str) - 1]
    return _validate_and_return(selected, current_config)

def _validate_and_return(
    ckpt: CheckpointInfo,
    current_config: dict,
) -> Optional[CheckpointInfo]:
    compat = check_architecture_compatibility(ckpt, current_config)
    if not compat.compatible:
        console.print(Panel(
            f"[bold red]Architecture Mismatch[/bold red]\n\n"
            f"Cannot resume: checkpoint and current config differ on:\n"
            f"{compat.format_mismatches()}\n\n"
            f"You must either:\n"
            f"  - Use the same config as when you started training\n"
            f"  - Start a new run with the current config",
            style="red"
        ))
        if not confirm("Start fresh instead?", default=True):
            raise SystemExit(0)
        return None
    return ckpt
```

### Step 5: Load checkpoint into model

```python
# src/checkpoint.py (additions)

def load_checkpoint(
    checkpoint_info,    # CheckpointInfo or Path
    model,
    optimizer,
    device: str = "cuda",
    strict: bool = True,
) -> int:
    """
    Load model (and optionally optimizer) state from checkpoint.
    Returns the step number to resume from.
    """
    path = checkpoint_info.path if hasattr(checkpoint_info, "path") else checkpoint_info
    console = Console()

    console.print(f"[cyan]Loading checkpoint: {path}[/cyan]")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            console.print("[green]Optimizer state loaded.[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not load optimizer state ({e}). Optimizer will restart from scratch.[/yellow]")

    step = ckpt.get("step", 0)
    console.print(f"[green]Resumed from step {step:,}[/green]")
    return step
```

---

## Key Files to Modify

| File | Change |
|------|--------|
| `src/checkpoint.py` | Add metadata to `save_checkpoint()`, add `load_checkpoint()` |
| `src/training/resume/scanner.py` | New: checkpoint discovery |
| `src/training/resume/compatibility.py` | New: architecture comparison |
| `src/training/resume/menu.py` | New: interactive resume menu |
| `src/train.py` | Call `offer_resume()` at the start of training |
| `src/cli.py` | Ensure `choose()` supports dict-style choices |

---

## Testing Strategy

- **Scanner test**: Create dummy checkpoint files in a temp directory, verify scanner finds and sorts them correctly
- **Metadata test**: Save a checkpoint with known step/loss, load with `_load_checkpoint_info`, verify fields match
- **Compatibility test**: Two configs differing only in `n_layers` → mismatch detected; configs differing only in `learning_rate` → compatible
- **No checkpoints test**: Empty `runs/` directory → `offer_resume()` returns None immediately
- **Optimizer state test**: Checkpoint with no optimizer state → load gracefully with warning, not error
- **Corrupted checkpoint test**: Partially written .pt file → scanner skips it silently

---

## Performance Considerations

- `torch.load()` on a 200MB checkpoint with `weights_only=False` takes ~1-2 seconds. For metadata scanning, we're loading the full checkpoint. A future optimization: save a separate `metadata.json` alongside each checkpoint to avoid loading weights just for metadata.
- Scanning 20 checkpoints would take 20-40 seconds. Mitigate by only loading the latest from each run by default, with `Pick Checkpoint` loading all on demand.
- Immediate optimization: scan only manifest.json files first, fall back to loading .pt files if manifest is missing.

---

## Dependencies

No new packages. Uses `torch`, `pathlib`, `json`, `rich` (all existing).

---

## Estimated Complexity

**Low-Medium** — 1.5 days.

- Scanner + metadata extraction: 3 hours
- Compatibility checker: 1.5 hours
- Interactive menu: 2 hours
- Load checkpoint + optimizer handling: 2 hours
- Integration in train.py: 1 hour
- Testing: 3 hours

Total: ~12.5 hours

---

## 2026 Best Practices

- **Embed metadata in checkpoint**: Never rely on filenames alone for step/loss information. Always embed metadata in the checkpoint dict. This survives renames and copies.
- **Sidecar metadata file**: Optionally save a `ckpt_001000.json` alongside `ckpt_001000.pt` with the same metadata. This allows metadata scanning without loading the full PyTorch file.
- **Separate architecture from hyperparameters in compatibility**: Architecture fields (dim, n_layers, vocab_size) are incompatible across checkpoints. Hyperparameters (lr, batch_size) can safely change. Be explicit about which category each field belongs to.
- **Optimizer state is optional**: If the optimizer state doesn't load (wrong optimizer type, shape mismatch), training can still resume — the model weights are the important part. Log a warning but don't crash.
- **Never overwrite a running checkpoint**: Use atomic writes (write to .tmp, then rename) to prevent corrupted checkpoints if training crashes mid-save.
