# 05 - Overfitting Detector

## Overview

Monitor training and validation loss simultaneously. Raise a Rich warning panel when the model shows signs of overfitting: validation loss plateauing or rising while training loss continues to decrease. Uses exponential moving averages for noise-robust detection, configurable patience (K consecutive evals showing the pattern), and optional automated responses (reduce LR, stop training). Detection events are logged to the training manifest.

---

## Motivation

Overfitting is the most common failure mode when fine-tuning or training from scratch on a small corpus. A 50M parameter model on a 50M token corpus is capacity-matched, but if the corpus is repetitive (many similar boilerplate files) or if the learning rate schedule is wrong, the model can memorize rather than generalize.

Without a validation split and detector:
- You discover overfitting hours later when evaluating the final checkpoint
- The best checkpoint (highest generalization) may have been saved at step 3000 but training ran to step 10000
- You wasted GPU time and overwrite the best weights with worse weights

The overfitting detector lets you catch this early and respond automatically (or manually after a Rich alert).

---

## Architecture / Design

### Detection Algorithm

```
Every val_interval steps:
  1. Compute val_loss (existing feature 04)
  2. Update EMA(val_loss) and EMA(train_loss)
  3. Check pattern:
     - EMA(val_loss) has increased for K consecutive evals
     - EMA(train_loss) has decreased (or stayed flat) for same K evals
  4. If pattern detected:
     - Show Rich warning panel
     - Log to manifest
     - Optionally: reduce LR by factor, or set early_stop flag
```

### EMA Smoothing

Raw loss values are noisy. We use exponential moving average (EMA) with α=0.1 (heavy smoothing):

```
EMA_t = α * loss_t + (1 - α) * EMA_{t-1}
```

A smaller α (e.g. 0.05) gives smoother trends but slower detection. 0.1 is a good default.

### Severity Levels

| Condition | Severity | Response |
|-----------|----------|----------|
| val EMA up for K=3 evals | Warning | Print alert, log |
| val EMA up for K=5 evals | Critical | Print alert, optionally reduce LR |
| val EMA up for K=8 evals | Severe | Print alert, optionally stop training |

---

## Implementation Steps

### Step 1: EMA tracker class

```python
# src/training/overfitting_detector.py
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class OverfitSeverity(Enum):
    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"
    SEVERE = "severe"

@dataclass
class EMATracker:
    alpha: float = 0.1
    value: Optional[float] = None

    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

@dataclass
class OverfittingDetector:
    """
    Tracks train/val EMA losses and detects overfitting patterns.
    """
    alpha: float = 0.1
    patience_warning: int = 3     # K consecutive increases for warning
    patience_critical: int = 5    # K consecutive increases for critical
    patience_severe: int = 8      # K consecutive increases for severe
    min_improvement: float = 0.001  # Minimum delta to count as "not increasing"

    train_ema: EMATracker = field(default_factory=lambda: EMATracker(alpha=0.1))
    val_ema: EMATracker = field(default_factory=lambda: EMATracker(alpha=0.1))

    _val_history: list[float] = field(default_factory=list)
    _train_history: list[float] = field(default_factory=list)
    _consecutive_increases: int = 0
    _best_val_ema: Optional[float] = None

    def update(self, train_loss: float, val_loss: float) -> "DetectionResult":
        """
        Update with new loss values. Returns a DetectionResult.
        """
        smoothed_train = self.train_ema.update(train_loss)
        smoothed_val = self.val_ema.update(val_loss)

        self._val_history.append(smoothed_val)
        self._train_history.append(smoothed_train)

        # Update best val
        if self._best_val_ema is None or smoothed_val < self._best_val_ema:
            self._best_val_ema = smoothed_val
            self._consecutive_increases = 0
        elif smoothed_val > self._best_val_ema + self.min_improvement:
            self._consecutive_increases += 1
        else:
            # Essentially flat — count as neutral, don't increment
            pass

        # Is train still decreasing?
        train_decreasing = (
            len(self._train_history) < 2 or
            self._train_history[-1] < self._train_history[-2]
        )

        severity = self._compute_severity(train_decreasing)
        return DetectionResult(
            step=len(self._val_history),
            train_loss=train_loss,
            val_loss=val_loss,
            smoothed_train=smoothed_train,
            smoothed_val=smoothed_val,
            consecutive_val_increases=self._consecutive_increases,
            train_decreasing=train_decreasing,
            severity=severity,
            best_val_ema=self._best_val_ema,
        )

    def _compute_severity(self, train_decreasing: bool) -> OverfitSeverity:
        if not train_decreasing:
            return OverfitSeverity.NONE  # Train isn't dropping, not classic overfitting
        if self._consecutive_increases >= self.patience_severe:
            return OverfitSeverity.SEVERE
        if self._consecutive_increases >= self.patience_critical:
            return OverfitSeverity.CRITICAL
        if self._consecutive_increases >= self.patience_warning:
            return OverfitSeverity.WARNING
        return OverfitSeverity.NONE

    def reset_patience(self) -> None:
        """Call after responding to an overfitting event (e.g., LR reduction)."""
        self._consecutive_increases = 0

@dataclass
class DetectionResult:
    step: int
    train_loss: float
    val_loss: float
    smoothed_train: float
    smoothed_val: float
    consecutive_val_increases: int
    train_decreasing: bool
    severity: OverfitSeverity
    best_val_ema: float
```

### Step 2: Rich alert display

```python
# src/training/overfitting_detector.py (continued)
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

SEVERITY_STYLES = {
    OverfitSeverity.WARNING:  ("yellow", "Overfitting Warning"),
    OverfitSeverity.CRITICAL: ("red",    "Overfitting Detected"),
    OverfitSeverity.SEVERE:   ("bold red on dark_red", "Severe Overfitting — Action Required"),
}

def render_overfit_alert(result: DetectionResult) -> None:
    if result.severity == OverfitSeverity.NONE:
        return
    style, title = SEVERITY_STYLES[result.severity]
    content = (
        f"Val loss has increased for [bold]{result.consecutive_val_increases}[/bold] consecutive evals\n"
        f"while train loss continues to decrease.\n\n"
        f"Train loss (EMA):  {result.smoothed_train:.4f}\n"
        f"Val loss (EMA):    {result.smoothed_val:.4f}\n"
        f"Best val (EMA):    {result.best_val_ema:.4f}\n"
        f"Gap:               +{result.smoothed_val - result.best_val_ema:.4f}\n\n"
        f"[dim]Consider: reduce learning rate, increase dropout, or stop training.[/dim]"
    )
    console.print(Panel(content, title=title, style=style, border_style=style))
```

### Step 3: Automated response handler

```python
# src/training/overfitting_response.py
from src.training.overfitting_detector import OverfitSeverity, DetectionResult
from src.cli import confirm
from rich.console import Console

console = Console()

class OverfitResponder:
    def __init__(self, config: dict):
        self.auto_reduce_lr = config.get("overfit_auto_reduce_lr", False)
        self.auto_stop = config.get("overfit_auto_stop", False)
        self.lr_reduction_factor = config.get("overfit_lr_factor", 0.5)
        self._stop_flag = False
        self._lr_reductions = 0
        self._max_lr_reductions = 3

    @property
    def should_stop(self) -> bool:
        return self._stop_flag

    def respond(self, result: DetectionResult, optimizer) -> None:
        """Apply automated or interactive response to overfitting event."""
        if result.severity == OverfitSeverity.WARNING:
            # Just alert, no action
            return

        if result.severity == OverfitSeverity.CRITICAL:
            if self.auto_reduce_lr and self._lr_reductions < self._max_lr_reductions:
                self._reduce_lr(optimizer)
            elif not self.auto_reduce_lr:
                if confirm(f"Reduce learning rate by {self.lr_reduction_factor}x?", default=True):
                    self._reduce_lr(optimizer)

        if result.severity == OverfitSeverity.SEVERE:
            if self.auto_stop:
                console.print("[bold red]Auto-stopping training due to severe overfitting.[/bold red]")
                self._stop_flag = True
            else:
                if confirm("Stop training now? (Best checkpoint will be saved)", default=False):
                    self._stop_flag = True
                elif confirm(f"Reduce learning rate by {self.lr_reduction_factor}x?", default=True):
                    self._reduce_lr(optimizer)

    def _reduce_lr(self, optimizer) -> None:
        for group in optimizer.param_groups:
            old_lr = group["lr"]
            group["lr"] = old_lr * self.lr_reduction_factor
        new_lr = optimizer.param_groups[0]["lr"]
        self._lr_reductions += 1
        console.print(f"[yellow]LR reduced to {new_lr:.2e} (reduction #{self._lr_reductions})[/yellow]")
```

### Step 4: Integration into training loop

```python
# src/trainer.py (additions)
from src.training.overfitting_detector import OverfittingDetector, render_overfit_alert
from src.training.overfitting_response import OverfitResponder

class Trainer:
    def __init__(self, config: dict):
        # ... existing init ...
        self.overfit_detector = OverfittingDetector(
            alpha=config.get("overfit_ema_alpha", 0.1),
            patience_warning=config.get("overfit_patience_warning", 3),
            patience_critical=config.get("overfit_patience_critical", 5),
            patience_severe=config.get("overfit_patience_severe", 8),
        )
        self.overfit_responder = OverfitResponder(config)
        self._detection_log = []

    def _eval_step(self, step: int, train_loss: float, val_data) -> None:
        """Called every val_interval steps."""
        if val_data is None:
            return
        val_loss = self.compute_val_loss(val_data)
        result = self.overfit_detector.update(train_loss, val_loss)

        # Log to manifest
        self._detection_log.append({
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "smoothed_train": result.smoothed_train,
            "smoothed_val": result.smoothed_val,
            "severity": result.severity.value,
        })

        if result.severity.value != "none":
            render_overfit_alert(result)
            self.overfit_responder.respond(result, self.optimizer)
            if self.overfit_responder.should_stop:
                self._save_checkpoint(step, tag="overfit_stop")
                raise StopIteration("Overfitting detector triggered early stop")
            self.overfit_detector.reset_patience()

    def save_detection_log(self, run_dir: Path) -> None:
        import json
        log_path = run_dir / "overfit_log.json"
        log_path.write_text(json.dumps(self._detection_log, indent=2))
```

### Step 5: Config additions

```yaml
# configs/small.yaml (additions)
overfitting:
  enabled: true
  ema_alpha: 0.1              # Smoothing factor (0.05=heavy, 0.3=light)
  patience_warning: 3         # Consecutive val increases for warning
  patience_critical: 5        # Consecutive val increases for critical
  patience_severe: 8          # Consecutive val increases for severe
  auto_reduce_lr: false        # Automatically reduce LR on critical
  auto_stop: false             # Automatically stop on severe
  lr_reduction_factor: 0.5    # LR multiplier when reducing
```

---

## Key Files to Modify

| File | Change |
|------|--------|
| `src/training/overfitting_detector.py` | New: EMATracker, OverfittingDetector, alerts |
| `src/training/overfitting_response.py` | New: OverfitResponder with LR/stop actions |
| `src/trainer.py` | Add `_eval_step()`, integrate detector into training loop |
| `configs/*.yaml` | Add `overfitting:` section |
| `runs/<run>/overfit_log.json` | New: detection event log per run |

---

## Testing Strategy

- **EMA unit test**: Feed a known sequence [1.0, 0.9, 0.8, ...], verify EMA converges correctly
- **Consecutive increase counter**: Feed val losses [1.0, 1.1, 1.2, 1.3] with decreasing train losses — verify `consecutive_val_increases` reaches 3 and triggers WARNING
- **No false positive**: Feed val losses that are flat (within `min_improvement`) — verify no warning triggered
- **LR reduction test**: Create a mock optimizer, trigger CRITICAL severity, verify `lr` in `param_groups` is multiplied by `lr_reduction_factor`
- **Auto-stop test**: Set `auto_stop=True`, trigger SEVERE, verify `should_stop` becomes True
- **Reset after response**: After a response, verify `consecutive_increases` resets to 0

---

## Performance Considerations

- EMA computation is O(1) per step — negligible overhead
- `compute_val_loss()` (from feature 04) runs on 20 batches per eval — ~100ms on GPU
- Val eval happens every 500 steps (configurable) — not on every step
- The detection logic itself is pure Python arithmetic with no GPU use

---

## Dependencies

No new packages. Uses:
- `rich` (already installed)
- `dataclasses`, `enum` (stdlib)
- `torch` for model eval (already used)

---

## Estimated Complexity

**Low** — 1 day.

- EMATracker + OverfittingDetector: 2 hours
- Rich alert rendering: 1 hour
- OverfitResponder (LR/stop actions): 2 hours
- Trainer integration: 1.5 hours
- Config additions: 30 minutes
- Testing: 2 hours

Total: ~9 hours

---

## 2026 Best Practices

- **EMA over raw losses for detection**: Raw step-level loss is too noisy for pattern detection. EMA with α=0.1 filters noise while remaining responsive to genuine trends.
- **Patience-based detection**: A single val loss increase is not overfitting — it's noise. Require K consecutive increases before alerting. K=3 is a reasonable default.
- **Decouple detection from response**: The detector reports; the responder acts. This separation allows easy testing of detection logic without side effects, and allows swapping response strategies.
- **Log all events**: Overfitting detection is only useful if you can review it later. Log every detection event with timestamp, step, and losses to `overfit_log.json`.
- **Never auto-stop without confirmation** (unless `auto_stop: true` is explicitly set): Automatic stopping without user awareness can be frustrating. Default to an interactive prompt.
- **Best-checkpoint saving**: When stopping due to overfitting, always save the current checkpoint (even if it's not the best). The user can compare it to earlier checkpoints.
