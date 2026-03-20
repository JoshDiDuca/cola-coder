"""Crash Recovery: save emergency checkpoint on training failure.

Catches training crashes (OOM, CUDA errors, keyboard interrupt, etc.)
and saves an emergency checkpoint so no progress is lost.

Features:
- Signal handlers for SIGINT/SIGTERM
- try/except around training loop
- Emergency checkpoint saves current state
- On next start, detects crash checkpoint and offers recovery
- Periodic async checkpoint saving option
"""

import signal
import sys
import time
import json
from pathlib import Path
from cola_coder.cli import cli

FEATURE_ENABLED = True

def is_enabled() -> bool:
    return FEATURE_ENABLED


class CrashRecovery:
    """Manages crash recovery for training sessions."""

    def __init__(self, checkpoint_dir: str):
        """
        Args:
            checkpoint_dir: Directory where checkpoints are saved.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.crash_dir = self.checkpoint_dir / "_crash_recovery"
        self._original_sigint = None
        self._original_sigterm = None
        self._model = None
        self._optimizer = None
        self._scheduler = None
        self._step = 0
        self._loss = 0.0
        self._active = False

    def register(self, model, optimizer, scheduler=None):
        """Register model and optimizer for crash recovery.

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            scheduler: Optional LR scheduler
        """
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._active = True

        # Install signal handlers (Windows-safe)
        try:
            self._original_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._signal_handler)
        except (OSError, ValueError):
            pass

        # SIGTERM not available on Windows in all contexts
        try:
            self._original_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (OSError, ValueError, AttributeError):
            pass

        cli.dim("Crash recovery registered")

    def update_step(self, step: int, loss: float):
        """Update the current training step (call every step)."""
        self._step = step
        self._loss = loss

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        cli.warn(f"\nReceived {signal_name} at step {self._step}")
        self.save_crash_checkpoint(reason=f"signal_{signal_name}")

        # Restore original handler and re-raise
        if signum == signal.SIGINT and self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)

        sys.exit(1)

    def save_crash_checkpoint(self, reason: str = "unknown"):
        """Save an emergency checkpoint.

        Args:
            reason: Why the crash checkpoint was saved.
        """
        if not self._active or self._model is None:
            return

        self.crash_dir.mkdir(parents=True, exist_ok=True)

        cli.warn(f"Saving crash recovery checkpoint at step {self._step}...")

        try:
            import torch

            # Save model state
            model_path = self.crash_dir / "model_crash.pt"
            torch.save(self._model.state_dict(), str(model_path))

            # Save optimizer state
            optim_path = self.crash_dir / "optimizer_crash.pt"
            torch.save(self._optimizer.state_dict(), str(optim_path))

            # Save scheduler state
            if self._scheduler is not None:
                sched_path = self.crash_dir / "scheduler_crash.pt"
                torch.save(self._scheduler.state_dict(), str(sched_path))

            # Save metadata
            meta = {
                "step": self._step,
                "loss": self._loss,
                "reason": reason,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            meta_path = self.crash_dir / "crash_meta.json"
            meta_path.write_text(json.dumps(meta, indent=2))

            cli.success(f"Crash checkpoint saved to {self.crash_dir}")
            cli.info("Step", self._step)
            cli.info("Loss", f"{self._loss:.4f}")
            cli.info("Reason", reason)

        except Exception as e:
            cli.error(f"Failed to save crash checkpoint: {e}")

    def check_crash_checkpoint(self) -> dict | None:
        """Check if a crash checkpoint exists.

        Returns:
            Crash metadata dict if found, None otherwise.
        """
        meta_path = self.crash_dir / "crash_meta.json"
        if not meta_path.exists():
            return None

        try:
            meta = json.loads(meta_path.read_text())
            model_exists = (self.crash_dir / "model_crash.pt").exists()
            optim_exists = (self.crash_dir / "optimizer_crash.pt").exists()

            if not model_exists:
                return None

            meta["has_optimizer"] = optim_exists
            meta["has_scheduler"] = (self.crash_dir / "scheduler_crash.pt").exists()
            return meta

        except Exception:
            return None

    def recover(self, model, optimizer=None, scheduler=None) -> int:
        """Recover from a crash checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optional optimizer to restore
            scheduler: Optional scheduler to restore

        Returns:
            Step number to resume from.
        """
        import torch

        meta = self.check_crash_checkpoint()
        if meta is None:
            raise FileNotFoundError("No crash checkpoint found")

        cli.step(1, 2, "Loading crash recovery checkpoint")

        model_path = self.crash_dir / "model_crash.pt"
        model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
        cli.info("Model state", "restored")

        if optimizer and meta.get("has_optimizer"):
            optim_path = self.crash_dir / "optimizer_crash.pt"
            optimizer.load_state_dict(torch.load(str(optim_path), map_location="cpu"))
            cli.info("Optimizer state", "restored")

        if scheduler and meta.get("has_scheduler"):
            sched_path = self.crash_dir / "scheduler_crash.pt"
            scheduler.load_state_dict(torch.load(str(sched_path), map_location="cpu"))
            cli.info("Scheduler state", "restored")

        step = meta.get("step", 0)
        cli.success(f"Recovered from step {step} (crash reason: {meta.get('reason', 'unknown')})")

        return step

    def clear_crash_checkpoint(self):
        """Delete crash checkpoint after successful recovery or manual cleanup."""
        if self.crash_dir.exists():
            import shutil
            shutil.rmtree(str(self.crash_dir), ignore_errors=True)
            cli.dim("Crash checkpoint cleared")

    def unregister(self):
        """Restore original signal handlers."""
        self._active = False
        try:
            if self._original_sigint:
                signal.signal(signal.SIGINT, self._original_sigint)
        except (OSError, ValueError):
            pass
        try:
            if self._original_sigterm:
                signal.signal(signal.SIGTERM, self._original_sigterm)
        except (OSError, ValueError, AttributeError):
            pass


def wrap_training_with_recovery(trainer_fn, checkpoint_dir: str, model, optimizer, scheduler=None, **kwargs):
    """Wrap a training function with crash recovery.

    Usage:
        wrap_training_with_recovery(
            trainer.train,
            checkpoint_dir="./checkpoints/tiny",
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            data_path="data/processed/train_data.npy",
        )

    Args:
        trainer_fn: The training function to call.
        checkpoint_dir: Where to save crash checkpoints.
        model: PyTorch model.
        optimizer: PyTorch optimizer.
        scheduler: Optional LR scheduler.
        **kwargs: Arguments to pass to trainer_fn.
    """
    recovery = CrashRecovery(checkpoint_dir)

    # Check for existing crash checkpoint
    crash_meta = recovery.check_crash_checkpoint()
    if crash_meta:
        cli.warn(f"Found crash checkpoint from step {crash_meta['step']}")
        cli.info("Crash reason", crash_meta.get("reason", "unknown"))
        cli.info("Timestamp", crash_meta.get("timestamp", "unknown"))

        if cli.confirm("Recover from crash checkpoint?"):
            recovery.recover(model, optimizer, scheduler)
            recovery.clear_crash_checkpoint()
        else:
            if cli.confirm("Delete crash checkpoint?", default=False):
                recovery.clear_crash_checkpoint()

    # Register for crash recovery
    recovery.register(model, optimizer, scheduler)

    try:
        trainer_fn(**kwargs)
        recovery.clear_crash_checkpoint()
    except KeyboardInterrupt:
        recovery.save_crash_checkpoint(reason="keyboard_interrupt")
        cli.warn("Training interrupted. Crash checkpoint saved.")
        raise
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            recovery.save_crash_checkpoint(reason=f"cuda_error: {str(e)[:100]}")
        else:
            recovery.save_crash_checkpoint(reason=f"runtime_error: {str(e)[:100]}")
        raise
    except Exception as e:
        recovery.save_crash_checkpoint(reason=f"exception: {type(e).__name__}: {str(e)[:100]}")
        raise
    finally:
        recovery.unregister()
