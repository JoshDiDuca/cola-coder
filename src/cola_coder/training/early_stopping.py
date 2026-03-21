"""Early stopping for training.

Early stopping monitors a validation metric (usually validation loss) and halts
training when it stops improving. This prevents overfitting — the point where
the model memorizes training data but generalizes worse.

The "patience" parameter controls how forgiving the stopper is: with patience=5,
training stops only after 5 consecutive non-improving steps.

For a TS dev: think of this like a circuit breaker — it trips after N consecutive
failures (non-improvements), but resets the counter whenever things improve.

Usage:
    stopper = EarlyStopping(patience=5, min_delta=0.001, mode="min")
    for epoch in range(max_epochs):
        val_loss = evaluate(model)
        if stopper.step(val_loss, model=model, step=epoch):
            print("Early stopping triggered!")
            break
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Monitor a metric and stop training when it stops improving.

    Supports both minimization (loss) and maximization (accuracy) modes.
    Optionally saves the best model separately from regular checkpoints.

    State can be serialized/loaded for checkpoint resume support.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "min",
        save_best: bool = True,
        best_model_path: str = "checkpoints/best",
        verbose: bool = True,
    ):
        """
        Args:
            patience: Number of calls to step() with no improvement before stopping.
            min_delta: Minimum change to count as an improvement. Acts as a dead zone
                       so tiny fluctuations don't reset the counter.
            mode: "min" to minimize (e.g., loss), "max" to maximize (e.g., accuracy).
            save_best: If True, save the model when a new best metric is achieved.
            best_model_path: Directory to save the best model state dict.
            verbose: If True, log patience decrements and best-metric updates.
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_best = save_best
        self.best_model_path = best_model_path
        self.verbose = verbose

        # Runtime state (also part of state_dict for resuming)
        self.counter: int = 0
        self.best_score: float | None = None
        self.should_stop: bool = False
        self.best_step: int = 0
        self.num_calls: int = 0

    def _is_improvement(self, score: float) -> bool:
        """Check if score represents an improvement over best_score."""
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

    def step(self, metric: float, model: nn.Module | None = None, step: int = 0) -> bool:
        """Check if training should stop.

        Call this once per validation step, passing the latest validation metric.
        Saves the best model if save_best=True and the metric improved.

        Args:
            metric: The validation metric value (e.g., val_loss or val_accuracy).
            model: The model to save if metric improved and save_best=True.
            step: Current training step (used for logging and best_step tracking).

        Returns:
            True if training should stop, False to continue.
        """
        self.num_calls += 1

        if self._is_improvement(metric):
            prev_best = self.best_score
            self.best_score = metric
            self.best_step = step
            self.counter = 0

            if self.verbose:
                if prev_best is None:
                    logger.info(
                        "EarlyStopping: initial best %s = %.6f at step %d",
                        "loss" if self.mode == "min" else "score",
                        metric,
                        step,
                    )
                else:
                    direction = "decreased" if self.mode == "min" else "increased"
                    logger.info(
                        "EarlyStopping: metric %s from %.6f → %.6f at step %d (counter reset)",
                        direction,
                        prev_best,
                        metric,
                        step,
                    )

            # Save best model
            if self.save_best and model is not None:
                self._save_best_model(model, metric, step)
        else:
            self.counter += 1
            remaining = self.patience - self.counter
            if self.verbose:
                logger.info(
                    "EarlyStopping: no improvement (metric=%.6f, best=%.6f). "
                    "Patience counter: %d/%d (%d remaining)",
                    metric,
                    self.best_score if self.best_score is not None else float("nan"),
                    self.counter,
                    self.patience,
                    remaining,
                )

            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    logger.info(
                        "EarlyStopping: patience (%d) exhausted at step %d. "
                        "Best metric was %.6f at step %d. Stopping.",
                        self.patience,
                        step,
                        self.best_score if self.best_score is not None else float("nan"),
                        self.best_step,
                    )

        return self.should_stop

    def _save_best_model(self, model: nn.Module, metric: float, step: int):
        """Save model state dict to best_model_path.

        Uses the same safetensors-compatible approach as the main checkpoint system:
        strips torch.compile prefix and excludes tied output.weight.
        """
        try:
            path = Path(self.best_model_path)
            path.mkdir(parents=True, exist_ok=True)

            # Try safetensors first (preferred format for this project)
            try:
                from safetensors.torch import save_file

                raw_state = model.state_dict()
                state_dict = {}
                for k, v in raw_state.items():
                    clean_key = k.removeprefix("_orig_mod.")
                    if clean_key == "output.weight":
                        continue  # Weight-tied — skip to avoid duplicates
                    state_dict[clean_key] = v.contiguous()

                save_file(state_dict, str(path / "best_model.safetensors"))
            except ImportError:
                # Fallback to torch.save if safetensors not available
                torch.save(model.state_dict(), str(path / "best_model.pt"))

            # Write metadata
            metadata = {
                "metric": metric,
                "step": step,
                "mode": self.mode,
                "best_score": self.best_score,
            }
            (path / "best_metadata.json").write_text(json.dumps(metadata, indent=2))

            if self.verbose:
                logger.info("EarlyStopping: saved best model to %s", path)

        except Exception as e:
            logger.warning("EarlyStopping: failed to save best model: %s", e)

    def state_dict(self) -> dict:
        """Serialize state for checkpoint saving.

        Include this dict in your training checkpoint so early stopping
        state is preserved across training interruptions.

        Returns:
            Dict containing all state needed to resume.
        """
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "save_best": self.save_best,
            "best_model_path": self.best_model_path,
            "verbose": self.verbose,
            "counter": self.counter,
            "best_score": self.best_score,
            "should_stop": self.should_stop,
            "best_step": self.best_step,
            "num_calls": self.num_calls,
        }

    def load_state_dict(self, state: dict):
        """Restore state from a saved state dict.

        Args:
            state: Dict from a previous call to state_dict().
        """
        # Runtime state
        self.counter = state["counter"]
        self.best_score = state["best_score"]
        self.should_stop = state["should_stop"]
        self.best_step = state.get("best_step", 0)
        self.num_calls = state.get("num_calls", 0)

        # Configuration (restore in case the object was re-created)
        self.patience = state.get("patience", self.patience)
        self.min_delta = state.get("min_delta", self.min_delta)
        self.mode = state.get("mode", self.mode)
        self.save_best = state.get("save_best", self.save_best)
        self.best_model_path = state.get("best_model_path", self.best_model_path)
        self.verbose = state.get("verbose", self.verbose)

    def reset(self):
        """Reset all state (start fresh without changing configuration)."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_step = 0
        self.num_calls = 0

    def __repr__(self) -> str:
        return (
            f"EarlyStopping("
            f"patience={self.patience}, "
            f"min_delta={self.min_delta}, "
            f"mode={self.mode!r}, "
            f"counter={self.counter}/{self.patience}, "
            f"best={self.best_score}, "
            f"should_stop={self.should_stop}"
            f")"
        )
