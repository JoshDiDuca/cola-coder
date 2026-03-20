"""Training Monitor: live dashboard during model training.

Shows real-time training metrics in a Rich Live display:
- Loss curve (ASCII sparkline)
- Current learning rate
- Step progress (current/total)
- Tokens/sec throughput
- VRAM usage
- ETA
- Recent generation samples (optional)

Non-blocking: metrics are collected from the training loop and displayed
in the main thread via Rich Live.
"""

import time
import math
from dataclasses import dataclass, field
from collections import deque

FEATURE_ENABLED = True

def is_enabled() -> bool:
    return FEATURE_ENABLED


def _sparkline(values: list[float], width: int = 40) -> str:
    """Generate a Unicode sparkline from a list of values."""
    if not values:
        return ""

    blocks = " ▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1.0

    # Downsample if too many values
    if len(values) > width:
        step = len(values) / width
        sampled = []
        for i in range(width):
            idx = int(i * step)
            sampled.append(values[idx])
        values = sampled

    result = ""
    for v in values:
        idx = int((v - mn) / rng * (len(blocks) - 1))
        idx = max(0, min(len(blocks) - 1, idx))
        result += blocks[idx]

    return result


@dataclass
class TrainingSnapshot:
    """A snapshot of training state at a point in time."""
    step: int
    total_steps: int
    loss: float
    learning_rate: float
    tokens_per_sec: float
    vram_used_gb: float
    vram_total_gb: float
    elapsed_sec: float
    val_loss: float | None = None


class TrainingMonitor:
    """Collects and displays training metrics in real-time."""

    def __init__(self, total_steps: int, log_interval: int = 10, history_size: int = 500):
        """
        Args:
            total_steps: Total training steps expected.
            log_interval: Update display every N steps.
            history_size: Number of recent loss values to keep for sparkline.
        """
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.loss_history: deque[float] = deque(maxlen=history_size)
        self.val_loss_history: deque[float] = deque(maxlen=100)
        self.throughput_history: deque[float] = deque(maxlen=50)
        self.start_time: float = time.time()
        self.last_step_time: float = time.time()
        self.current_step: int = 0
        self.current_loss: float = 0.0
        self.current_lr: float = 0.0
        self.tokens_this_step: int = 0

    def update(self, step: int, loss: float, lr: float, tokens: int = 0,
               val_loss: float | None = None) -> str | None:
        """Record a training step and optionally return display string.

        Args:
            step: Current step number.
            loss: Training loss for this step.
            lr: Current learning rate.
            tokens: Tokens processed in this step.
            val_loss: Validation loss (if evaluated this step).

        Returns:
            Formatted display string if this is a display interval, else None.
        """
        now = time.time()
        self.current_step = step
        self.current_loss = loss
        self.current_lr = lr
        self.loss_history.append(loss)

        # Calculate throughput
        dt = now - self.last_step_time
        if dt > 0 and tokens > 0:
            tps = tokens / dt
            self.throughput_history.append(tps)
        self.last_step_time = now

        if val_loss is not None:
            self.val_loss_history.append(val_loss)

        # Only display at intervals
        if step % self.log_interval != 0 and step != self.total_steps:
            return None

        return self.format_display(step, loss, lr, val_loss)

    def format_display(self, step: int, loss: float, lr: float,
                       val_loss: float | None = None) -> str:
        """Format the training monitor display."""
        elapsed = time.time() - self.start_time

        # Progress
        pct = step / max(self.total_steps, 1) * 100

        # ETA
        if step > 0:
            secs_per_step = elapsed / step
            remaining_steps = self.total_steps - step
            eta_secs = secs_per_step * remaining_steps
            eta_str = self._format_time(eta_secs)
        else:
            eta_str = "calculating..."

        # Throughput
        avg_tps = sum(self.throughput_history) / max(len(self.throughput_history), 1)

        # VRAM
        vram_str = self._get_vram_str()

        # Loss sparkline
        spark = _sparkline(list(self.loss_history), width=30)

        # Perplexity
        ppl = math.exp(min(loss, 20))  # Cap to avoid overflow

        lines = [
            f"Step {step:,}/{self.total_steps:,} ({pct:.1f}%) | ETA: {eta_str} | Elapsed: {self._format_time(elapsed)}",
            f"Loss: {loss:.4f} | Perplexity: {ppl:.1f} | LR: {lr:.2e}",
            f"Throughput: {avg_tps:,.0f} tok/s | {vram_str}",
            f"Loss curve: {spark} ({list(self.loss_history)[0]:.2f} -> {loss:.2f})" if self.loss_history else "",
        ]

        if val_loss is not None:
            lines.append(f"Val Loss: {val_loss:.4f} | Val PPL: {math.exp(min(val_loss, 20)):.1f}")

        return "\n".join(line for line in lines if line)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m {seconds%60:.0f}s"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h {m}m"

    def _get_vram_str(self) -> str:
        """Get VRAM usage string."""
        try:
            import torch
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1e9
                props = torch.cuda.get_device_properties(0)
                total = (getattr(props, 'total_memory', 0) or getattr(props, 'total_mem', 0)) / 1e9
                pct = used / total * 100 if total > 0 else 0
                return f"VRAM: {used:.1f}/{total:.1f}GB ({pct:.0f}%)"
        except Exception:
            pass
        return "VRAM: N/A"

    def get_summary(self) -> dict:
        """Get a summary of the training session."""
        elapsed = time.time() - self.start_time
        avg_tps = sum(self.throughput_history) / max(len(self.throughput_history), 1)

        return {
            "total_steps": self.current_step,
            "final_loss": self.current_loss,
            "final_perplexity": math.exp(min(self.current_loss, 20)),
            "elapsed_seconds": elapsed,
            "avg_tokens_per_sec": avg_tps,
            "loss_history_len": len(self.loss_history),
        }

    def print_final_summary(self):
        """Print a final summary after training."""
        from cola_coder.cli import cli
        summary = self.get_summary()

        cli.rule("Training Session Summary")
        cli.kv_table({
            "Steps completed": f"{summary['total_steps']:,}",
            "Final loss": f"{summary['final_loss']:.4f}",
            "Final perplexity": f"{summary['final_perplexity']:.1f}",
            "Duration": self._format_time(summary['elapsed_seconds']),
            "Avg throughput": f"{summary['avg_tokens_per_sec']:,.0f} tok/s",
        })

        # Show loss curve
        if self.loss_history:
            spark = _sparkline(list(self.loss_history), width=50)
            cli.print(f"\n  Loss curve: {spark}")
            cli.print(f"  Start: {list(self.loss_history)[0]:.4f} -> End: {list(self.loss_history)[-1]:.4f}")
