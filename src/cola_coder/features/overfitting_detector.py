"""Overfitting Detector: alerts when the model starts overfitting.

Monitors training and validation loss, detects when:
- Val loss increases while train loss decreases
- Val loss plateaus while train loss keeps dropping
- Large gap between train and val loss

Uses exponential moving average (EMA) for smooth detection.
"""

from dataclasses import dataclass, field
from cola_coder.cli import cli
import math

FEATURE_ENABLED = True

def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class OverfitAlert:
    """Record of an overfitting detection event."""
    step: int
    alert_type: str  # "divergence", "val_plateau", "val_increasing"
    train_loss: float
    val_loss: float
    message: str
    severity: str  # "warning", "critical"


class OverfittingDetector:
    """Monitors train/val loss for signs of overfitting."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.01,
        ema_alpha: float = 0.1,
        divergence_threshold: float = 0.5,
    ):
        """
        Args:
            patience: How many consecutive evals of increasing val loss before alerting.
            min_delta: Minimum change to count as improvement.
            ema_alpha: Smoothing factor for exponential moving average (0-1, higher = less smoothing).
            divergence_threshold: Alert when val_loss - train_loss exceeds this.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.ema_alpha = ema_alpha
        self.divergence_threshold = divergence_threshold

        # Tracking state
        self.train_losses: list[tuple[int, float]] = []  # (step, loss)
        self.val_losses: list[tuple[int, float]] = []  # (step, loss)
        self.train_ema: float | None = None
        self.val_ema: float | None = None
        self.best_val_loss: float = float("inf")
        self.best_val_step: int = 0
        self.increasing_count: int = 0  # Consecutive increases in val loss
        self.alerts: list[OverfitAlert] = []

    def _update_ema(self, current_ema: float | None, new_value: float) -> float:
        """Update exponential moving average."""
        if current_ema is None:
            return new_value
        return self.ema_alpha * new_value + (1 - self.ema_alpha) * current_ema

    def record_train_loss(self, step: int, loss: float) -> None:
        """Record a training loss value."""
        self.train_losses.append((step, loss))
        self.train_ema = self._update_ema(self.train_ema, loss)

    def record_val_loss(self, step: int, loss: float) -> list[OverfitAlert]:
        """Record a validation loss value and check for overfitting.

        Args:
            step: Current training step.
            loss: Validation loss.

        Returns:
            List of new alerts (empty if no issues detected).
        """
        self.val_losses.append((step, loss))
        self.val_ema = self._update_ema(self.val_ema, loss)

        new_alerts = []

        # Check: is val loss improving?
        if loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = loss
            self.best_val_step = step
            self.increasing_count = 0
        else:
            self.increasing_count += 1

        # Check: val loss increasing for too long
        if self.increasing_count >= self.patience:
            alert = OverfitAlert(
                step=step,
                alert_type="val_increasing",
                train_loss=self.train_ema or 0,
                val_loss=loss,
                message=(
                    f"Validation loss has not improved for {self.increasing_count} evaluations. "
                    f"Best val loss was {self.best_val_loss:.4f} at step {self.best_val_step}. "
                    f"Current: {loss:.4f}"
                ),
                severity="critical" if self.increasing_count >= self.patience * 2 else "warning",
            )
            new_alerts.append(alert)

        # Check: train/val divergence
        if self.train_ema is not None and self.val_ema is not None:
            gap = self.val_ema - self.train_ema
            if gap > self.divergence_threshold:
                alert = OverfitAlert(
                    step=step,
                    alert_type="divergence",
                    train_loss=self.train_ema,
                    val_loss=self.val_ema,
                    message=(
                        f"Train/val loss gap ({gap:.4f}) exceeds threshold ({self.divergence_threshold}). "
                        f"Train EMA: {self.train_ema:.4f}, Val EMA: {self.val_ema:.4f}"
                    ),
                    severity="warning" if gap < self.divergence_threshold * 2 else "critical",
                )
                new_alerts.append(alert)

        # Check: val plateau (val not changing while train drops)
        if len(self.val_losses) >= 3 and self.train_ema is not None:
            recent_val = [v for _, v in self.val_losses[-3:]]
            val_range = max(recent_val) - min(recent_val)

            if len(self.train_losses) >= 3:
                recent_train = [v for _, v in self.train_losses[-3:]]
                train_drop = recent_train[0] - recent_train[-1]

                if val_range < self.min_delta and train_drop > self.min_delta * 3:
                    alert = OverfitAlert(
                        step=step,
                        alert_type="val_plateau",
                        train_loss=self.train_ema,
                        val_loss=self.val_ema or loss,
                        message=(
                            f"Validation loss plateaued (range: {val_range:.4f}) "
                            f"while train loss dropped by {train_drop:.4f}"
                        ),
                        severity="warning",
                    )
                    new_alerts.append(alert)

        # Display alerts
        for alert in new_alerts:
            self.alerts.append(alert)
            self._display_alert(alert)

        return new_alerts

    def _display_alert(self, alert: OverfitAlert):
        """Display an alert in the CLI."""
        if alert.severity == "critical":
            cli.error(f"[Overfit] {alert.message}")
        else:
            cli.warn(f"[Overfit] {alert.message}")

    def should_stop_early(self) -> bool:
        """Recommend early stopping if overfitting is severe."""
        if self.increasing_count >= self.patience * 3:
            return True

        # Check for consistent divergence
        critical_alerts = [a for a in self.alerts if a.severity == "critical"]
        if len(critical_alerts) >= 3:
            return True

        return False

    def get_summary(self) -> dict:
        """Get a summary of the detector state."""
        return {
            "best_val_loss": self.best_val_loss if self.best_val_loss < float("inf") else None,
            "best_val_step": self.best_val_step,
            "current_train_ema": self.train_ema,
            "current_val_ema": self.val_ema,
            "increasing_count": self.increasing_count,
            "total_alerts": len(self.alerts),
            "should_stop": self.should_stop_early(),
        }

    def print_summary(self):
        """Print a summary to CLI."""
        summary = self.get_summary()
        cli.rule("Overfitting Detector Summary")
        cli.kv_table({
            "Best val loss": f"{summary['best_val_loss']:.4f}" if summary['best_val_loss'] else "N/A",
            "Best val step": str(summary['best_val_step']),
            "Train EMA": f"{summary['current_train_ema']:.4f}" if summary['current_train_ema'] else "N/A",
            "Val EMA": f"{summary['current_val_ema']:.4f}" if summary['current_val_ema'] else "N/A",
            "Stale evals": str(summary['increasing_count']),
            "Total alerts": str(summary['total_alerts']),
            "Recommend stop": "YES" if summary['should_stop'] else "No",
        })
