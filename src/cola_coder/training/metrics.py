"""Training metrics tracking.

Tracks loss, learning rate, throughput, and other metrics during training.
Optionally logs to Weights & Biases (wandb) for a nice dashboard.

For a TS dev: this is like a logging/monitoring service that records
training progress so you can check how things are going.
"""

import math
import time
from dataclasses import dataclass, field


@dataclass
class TrainingMetrics:
    """Accumulates and reports training metrics."""

    # Running accumulators (reset each logging interval)
    loss_sum: float = 0.0
    loss_count: int = 0
    tokens_processed: int = 0
    start_time: float = field(default_factory=time.time)

    # History (keeps all logged values)
    loss_history: list[float] = field(default_factory=list)
    lr_history: list[float] = field(default_factory=list)
    throughput_history: list[float] = field(default_factory=list)

    # wandb run (optional)
    _wandb_run: object = None

    def init_wandb(self, project: str = "cola-coder", config: dict | None = None):
        """Initialize Weights & Biases logging (optional).

        wandb gives you a web dashboard to monitor training in real-time.
        Free for personal use. Skip this if you just want console output.
        """
        try:
            import wandb
            self._wandb_run = wandb.init(project=project, config=config)
            print(f"wandb initialized: {wandb.run.url}")
        except ImportError:
            print("wandb not installed, skipping. Install with: pip install wandb")

    def update(self, loss: float, num_tokens: int):
        """Record a training step's loss and token count."""
        self.loss_sum += loss
        self.loss_count += 1
        self.tokens_processed += num_tokens

    def log(self, step: int, lr: float, log_interval: int = 100) -> str | None:
        """Log metrics if we've hit the logging interval.

        Args:
            step: Current training step.
            lr: Current learning rate.
            log_interval: Log every N steps.

        Returns:
            Formatted log string, or None if not a logging step.
        """
        if step % log_interval != 0 or self.loss_count == 0:
            return None

        avg_loss = self.loss_sum / self.loss_count
        elapsed = time.time() - self.start_time
        tokens_per_sec = self.tokens_processed / max(elapsed, 1e-6)
        perplexity = math.exp(min(avg_loss, 20))  # Cap to prevent overflow

        # Save to history
        self.loss_history.append(avg_loss)
        self.lr_history.append(lr)
        self.throughput_history.append(tokens_per_sec)

        # Format log message
        msg = (
            f"step {step:>7d} | "
            f"loss {avg_loss:.4f} | "
            f"ppl {perplexity:8.2f} | "
            f"lr {lr:.2e} | "
            f"tok/s {tokens_per_sec:,.0f}"
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
        """Clean up wandb if active."""
        if self._wandb_run is not None:
            import wandb
            wandb.finish()
