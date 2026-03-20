"""Training metrics tracking.

Tracks loss, learning rate, throughput, and other metrics during training.
Optionally logs to Weights & Biases (wandb) for a nice dashboard.

For a TS dev: this is like a logging/monitoring service that records
training progress so you can check how things are going.
"""

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..cli import cli


def _format_duration(seconds: float) -> str:
    """Format seconds into a readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s"


@dataclass
class TrainingMetrics:
    """Accumulates and reports training metrics."""

    # Running accumulators (reset each logging interval)
    loss_sum: float = 0.0
    loss_count: int = 0
    tokens_processed: int = 0
    start_time: float = field(default_factory=time.time)

    # Global timer (from first update to now)
    _global_start: float = 0.0
    _total_steps_seen: int = 0
    _max_steps: int = 0

    # History (keeps all logged values)
    loss_history: list[float] = field(default_factory=list)
    lr_history: list[float] = field(default_factory=list)
    throughput_history: list[float] = field(default_factory=list)

    # wandb run (optional)
    _wandb_run: object = None

    def init_wandb(self, project: str = "cola-coder", config: dict | None = None):
        """Initialize Weights & Biases logging (optional)."""
        try:
            import wandb
            self._wandb_run = wandb.init(project=project, config=config)
            cli.success(f"wandb initialized: {wandb.run.url}")
        except ImportError:
            cli.warn("wandb not installed, skipping. Install with: pip install wandb")

    def set_max_steps(self, max_steps: int):
        """Set total training steps for ETA calculation."""
        self._max_steps = max_steps

    def update(self, loss: float, num_tokens: int):
        """Record a training step's loss and token count."""
        if self._global_start == 0.0:
            self._global_start = time.time()
        self.loss_sum += loss
        self.loss_count += 1
        self.tokens_processed += num_tokens
        self._total_steps_seen += 1

    def log(self, step: int, lr: float, log_interval: int = 100) -> str | None:
        """Log metrics if we've hit the logging interval.

        Returns:
            Formatted log string (rich markup), or None.
        """
        if step % log_interval != 0 or self.loss_count == 0:
            return None

        now = time.time()
        avg_loss = self.loss_sum / self.loss_count
        elapsed = now - self.start_time
        tokens_per_sec = self.tokens_processed / max(elapsed, 1e-6)
        perplexity = math.exp(min(avg_loss, 20))

        # Global elapsed time
        global_elapsed = now - self._global_start if self._global_start else 0.0

        # ETA calculation
        eta_str = ""
        if self._max_steps > 0 and self._total_steps_seen > 0:
            steps_remaining = self._max_steps - step
            sec_per_step = global_elapsed / self._total_steps_seen
            eta_seconds = steps_remaining * sec_per_step
            eta_time = datetime.now() + timedelta(seconds=eta_seconds)
            eta_str = (
                f" | ETA {_format_duration(eta_seconds)} "
                f"({eta_time.strftime('%H:%M')})"
            )

        # Save to history
        self.loss_history.append(avg_loss)
        self.lr_history.append(lr)
        self.throughput_history.append(tokens_per_sec)

        # Timestamp and progress
        ts = datetime.now().strftime("%H:%M:%S")
        pct = step / self._max_steps * 100 if self._max_steps else 0

        # Color-code loss
        if avg_loss < 2.0:
            loss_style = "bold green"
        elif avg_loss < 3.0:
            loss_style = "green"
        elif avg_loss < 4.0:
            loss_style = "yellow"
        elif avg_loss < 6.0:
            loss_style = "dark_orange"
        else:
            loss_style = "red"

        # Color-code throughput
        if tokens_per_sec >= 200_000:
            tps_style = "bold green"
        elif tokens_per_sec >= 100_000:
            tps_style = "green"
        elif tokens_per_sec >= 50_000:
            tps_style = "yellow"
        else:
            tps_style = "red"

        # Build log message using cli.print markup
        msg = (
            f"[dim]{ts}[/dim] "
            f"[bold]step {step:>7,d}[/bold] "
            f"[dim]({pct:4.1f}%)[/dim] "
            f"[{loss_style}]loss {avg_loss:.4f}[/{loss_style}] "
            f"[dim]ppl[/dim] {perplexity:>8.1f} "
            f"[dim]lr[/dim] {lr:.2e} "
            f"[{tps_style}]{tokens_per_sec:>9,.0f} tok/s[/{tps_style}]"
            f"[dim]{eta_str}[/dim]"
        )

        # Log to wandb if available
        if self._wandb_run is not None:
            import wandb
            wandb.log({
                "loss": avg_loss,
                "perplexity": perplexity,
                "learning_rate": lr,
                "tokens_per_sec": tokens_per_sec,
                "step": step,
            })

        # Reset accumulators
        self.loss_sum = 0.0
        self.loss_count = 0
        self.tokens_processed = 0
        self.start_time = time.time()

        return msg

    def finish(self):
        """Print final summary and clean up wandb."""
        if self._global_start:
            total = time.time() - self._global_start
            extras = {
                "Duration": _format_duration(total),
                "Steps": f"{self._total_steps_seen:,}",
            }
            if self.loss_history:
                extras["Final loss"] = f"{self.loss_history[-1]:.4f}"
                extras["Best loss"] = f"{min(self.loss_history):.4f}"
            if self.throughput_history:
                avg_tps = sum(self.throughput_history) / len(self.throughput_history)
                extras["Avg throughput"] = f"{avg_tps:,.0f} tok/s"
            cli.done("Training complete", extras=extras)

        if self._wandb_run is not None:
            import wandb
            wandb.finish()
