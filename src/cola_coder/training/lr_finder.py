"""Learning Rate Finder using Smith's LR Range Test.

The LR finder is a diagnostic tool — run it BEFORE training to pick a good
learning rate. It works by training for a few hundred steps while exponentially
increasing the learning rate from very small (1e-7) to very large (10.0).

Plot the loss vs LR: the optimal LR is where the loss decreases most steeply
(steepest negative slope), just before it starts diverging.

For a TS dev: imagine a binary search for the learning rate, except you plot
the whole landscape at once instead of bisecting.

Reference: "Cyclical Learning Rates for Training Neural Networks" by Leslie Smith (2017).
"""

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


@dataclass
class LRFinderResult:
    """Results from an LR range test run."""

    lrs: list[float]
    losses: list[float]
    smoothed_losses: list[float]
    suggested_lr: float
    suggested_min_lr: float
    # Step index of the suggested LR in the lrs list
    suggested_idx: int = 0
    # Whether the run diverged early
    diverged_early: bool = False
    num_steps_run: int = 0

    def summary(self) -> str:
        """Human-readable summary of results."""
        lines = [
            "LR Finder Results",
            f"  Steps run:       {self.num_steps_run}",
            f"  Diverged early:  {self.diverged_early}",
            f"  Suggested LR:    {self.suggested_lr:.2e}",
            f"  Suggested min LR:{self.suggested_min_lr:.2e}",
            f"  LR range tested: {self.lrs[0]:.2e} → {self.lrs[-1]:.2e}",
        ]
        return "\n".join(lines)


class LRFinder:
    """Find optimal learning rate using Smith's LR Range Test.

    Usage:
        finder = LRFinder(model, optimizer, criterion, device)
        result = finder.find(train_loader)
        print(result.summary())
        min_lr, max_lr = LRFinder.suggest_lr(result)
        finder.plot(result, save_path="lr_finder_plot.png")
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module | None = None,
        device: str = "cuda",
    ):
        """
        Args:
            model: The model to find LR for.
            optimizer: Optimizer (learning rate will be overridden during the test).
            criterion: Loss function. If None, model.compute_loss() is used (assumes
                       the model has a built-in loss method like Transformer.compute_loss).
            device: Device to run on.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Saved state for restoration after the test
        self._saved_model_state: dict | None = None
        self._saved_optimizer_state: dict | None = None

    def _save_state(self):
        """Deep-copy model and optimizer state for restoration."""
        self._saved_model_state = copy.deepcopy(self.model.state_dict())
        self._saved_optimizer_state = copy.deepcopy(self.optimizer.state_dict())

    def _restore_state(self):
        """Restore model and optimizer to pre-test state."""
        if self._saved_model_state is not None:
            self.model.load_state_dict(self._saved_model_state)
        if self._saved_optimizer_state is not None:
            self.optimizer.load_state_dict(self._saved_optimizer_state)

    def _compute_loss(self, batch: dict | torch.Tensor) -> torch.Tensor:
        """Compute loss from a batch, supporting both dict batches and raw tensors."""
        if self.criterion is not None:
            # External criterion: batch must be (inputs, targets) or a dict
            if isinstance(batch, dict):
                inputs = batch["input_ids"].to(self.device, non_blocking=True)
                # Assume targets are the shifted inputs (language modeling)
                outputs = self.model(inputs)
                targets = inputs[:, 1:].contiguous()
                logits = outputs[:, :-1, :].contiguous()
                return self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                inputs, targets = batch
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                return self.criterion(outputs, targets)
        else:
            # Model has built-in compute_loss (Transformer.compute_loss)
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            else:
                input_ids = batch.to(self.device, non_blocking=True)
            return self.model.compute_loss(input_ids)

    def find(
        self,
        train_loader,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_steps: int = 200,
        smooth_factor: float = 0.05,
        diverge_threshold: float = 4.0,
        gradient_accumulation: int = 1,
    ) -> LRFinderResult:
        """Run the LR range test.

        Exponentially increases the learning rate from start_lr to end_lr
        over num_steps steps, recording the loss at each step. Stops early
        if the loss diverges (exceeds diverge_threshold * min_loss_seen).

        Args:
            train_loader: DataLoader to draw batches from.
            start_lr: Minimum learning rate to test.
            end_lr: Maximum learning rate to test.
            num_steps: Number of LR steps to run.
            smooth_factor: Exponential smoothing factor for loss (0 = no smooth, 1 = max smooth).
            diverge_threshold: Stop if loss > min_loss * diverge_threshold.
            gradient_accumulation: Micro-steps per optimizer step (matches training config).

        Returns:
            LRFinderResult with recorded LRs, losses, and suggested LR.
        """
        self._save_state()
        self.model.train()

        # Exponential LR schedule: lr_i = start_lr * (end_lr / start_lr) ^ (i / num_steps)
        lr_mult = (end_lr / start_lr) ** (1.0 / num_steps)

        lrs: list[float] = []
        losses: list[float] = []
        smoothed_losses: list[float] = []

        best_loss = float("inf")
        smoothed_loss = 0.0
        diverged_early = False

        # Create an infinite iterator from the dataloader
        data_iter = iter(train_loader)

        def _next_batch():
            nonlocal data_iter
            try:
                return next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                return next(data_iter)

        # Set initial LR on all param groups
        current_lr = start_lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = current_lr

        actual_steps = 0

        try:
            for step in range(num_steps):
                self.optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0

                # Gradient accumulation support
                for micro_step in range(gradient_accumulation):
                    batch = _next_batch()
                    try:
                        loss = self._compute_loss(batch)
                        scaled_loss = loss / gradient_accumulation
                        scaled_loss.backward()
                        step_loss += loss.item()
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            raise RuntimeError(
                                f"OOM during LR finder at lr={current_lr:.2e}. "
                                f"Reduce batch_size or start with a smaller end_lr."
                            ) from e
                        raise

                self.optimizer.step()

                avg_step_loss = step_loss / gradient_accumulation
                actual_steps += 1

                # Exponential smoothing: smooth = alpha * raw + (1 - alpha) * prev_smooth
                if step == 0:
                    smoothed_loss = avg_step_loss
                else:
                    smoothed_loss = smooth_factor * avg_step_loss + (1.0 - smooth_factor) * smoothed_loss

                # Bias correction for the first few steps (same trick as Adam)
                bias_correction = 1.0 - (1.0 - smooth_factor) ** (step + 1)
                corrected_loss = smoothed_loss / bias_correction

                lrs.append(current_lr)
                losses.append(avg_step_loss)
                smoothed_losses.append(corrected_loss)

                if corrected_loss < best_loss:
                    best_loss = corrected_loss

                # Early stopping: divergence check
                if step > 10 and corrected_loss > diverge_threshold * best_loss:
                    diverged_early = True
                    break

                # Advance LR exponentially
                current_lr *= lr_mult
                for pg in self.optimizer.param_groups:
                    pg["lr"] = current_lr

        finally:
            # Always restore state, even if we crash mid-run
            self._restore_state()

        # Find suggested LR: point of steepest decrease on the smoothed curve
        suggested_lr, suggested_idx = self._find_steepest_descent(lrs, smoothed_losses)
        suggested_min_lr = suggested_lr / 10.0

        return LRFinderResult(
            lrs=lrs,
            losses=losses,
            smoothed_losses=smoothed_losses,
            suggested_lr=suggested_lr,
            suggested_min_lr=suggested_min_lr,
            suggested_idx=suggested_idx,
            diverged_early=diverged_early,
            num_steps_run=actual_steps,
        )

    @staticmethod
    def _find_steepest_descent(lrs: list[float], smoothed_losses: list[float]) -> tuple[float, int]:
        """Find the LR at the point of steepest loss decrease.

        Computes the numerical gradient (finite differences) on the smoothed
        loss curve and returns the LR at the minimum gradient.

        Returns:
            Tuple of (suggested_lr, index_in_lrs).
        """
        if len(lrs) < 3:
            # Not enough points — just use the middle
            mid = len(lrs) // 2
            return lrs[mid], mid

        # Compute forward-differences (loss[i+1] - loss[i])
        gradients = [
            smoothed_losses[i + 1] - smoothed_losses[i]
            for i in range(len(smoothed_losses) - 1)
        ]

        # Skip first 10% and last 10% to avoid edge artifacts
        skip = max(1, len(gradients) // 10)
        search_region = gradients[skip:-skip] if len(gradients) > 2 * skip else gradients

        if not search_region:
            mid = len(lrs) // 2
            return lrs[mid], mid

        # Find index of steepest descent (most negative gradient)
        min_grad_idx = min(range(len(search_region)), key=lambda i: search_region[i])
        actual_idx = min_grad_idx + skip  # Adjust back to full gradient array index

        # The LR at that gradient step is lrs[actual_idx + 1] (gradient is between steps)
        lr_idx = min(actual_idx + 1, len(lrs) - 1)
        return lrs[lr_idx], lr_idx

    @staticmethod
    def suggest_lr(result: LRFinderResult) -> tuple[float, float]:
        """Suggest a (min_lr, max_lr) range for training schedules.

        The suggested max_lr is the LR at steepest descent.
        The suggested min_lr is max_lr / 10, suitable as the cosine schedule floor.

        Args:
            result: LRFinderResult from a find() call.

        Returns:
            Tuple of (min_lr, max_lr).
        """
        return result.suggested_min_lr, result.suggested_lr

    def plot(self, result: LRFinderResult, save_path: str | None = None):
        """Plot the loss vs LR curve.

        Uses matplotlib if available, falls back to ASCII art otherwise.

        Args:
            result: LRFinderResult from a find() call.
            save_path: If provided, save the plot to this path (matplotlib only).
        """
        if _HAS_MATPLOTLIB:
            self._plot_matplotlib(result, save_path)
        else:
            self._plot_ascii(result)
            if save_path:
                print(f"  (matplotlib not installed — cannot save plot to {save_path})")

    def _plot_matplotlib(self, result: LRFinderResult, save_path: str | None = None):
        """Render the LR finder plot with matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.semilogx(result.lrs, result.losses, alpha=0.3, color="blue", label="Raw loss")
        ax.semilogx(
            result.lrs, result.smoothed_losses,
            color="blue", linewidth=2, label="Smoothed loss"
        )

        # Mark the suggested LR
        if 0 <= result.suggested_idx < len(result.lrs):
            suggested_loss = result.smoothed_losses[result.suggested_idx]
            ax.axvline(
                x=result.suggested_lr, color="red", linestyle="--", linewidth=1.5,
                label=f"Suggested LR: {result.suggested_lr:.2e}"
            )
            ax.scatter([result.suggested_lr], [suggested_loss], color="red", zorder=5, s=80)

        ax.set_xlabel("Learning Rate (log scale)")
        ax.set_ylabel("Loss")
        ax.set_title("LR Finder — Loss vs Learning Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if result.diverged_early:
            ax.text(
                0.02, 0.98, "Diverged early",
                transform=ax.transAxes, va="top", color="orange",
                fontsize=10,
            )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Plot saved to: {save_path}")

        plt.show()
        plt.close(fig)

    def _plot_ascii(self, result: LRFinderResult):
        """Render a simple ASCII art loss curve."""
        width = 60
        height = 20

        losses = result.smoothed_losses
        if not losses:
            print("  No data to plot.")
            return

        min_loss = min(losses)
        max_loss = max(losses)
        loss_range = max_loss - min_loss or 1.0

        # Build the grid
        grid = [[" "] * width for _ in range(height)]

        # Plot points
        for i, loss in enumerate(losses):
            x = int(i / len(losses) * (width - 1))
            y = int((1.0 - (loss - min_loss) / loss_range) * (height - 1))
            y = max(0, min(height - 1, y))
            x = max(0, min(width - 1, x))
            grid[y][x] = "*"

        # Mark suggested LR
        if 0 <= result.suggested_idx < len(losses):
            x = int(result.suggested_idx / len(losses) * (width - 1))
            x = max(0, min(width - 1, x))
            for row in range(height):
                if grid[row][x] == " ":
                    grid[row][x] = "|"

        # Print
        lr_start = result.lrs[0] if result.lrs else 0
        lr_end = result.lrs[-1] if result.lrs else 0
        print("\nLR Finder — Loss vs Learning Rate (ASCII)")
        print(f"  Loss range: {min_loss:.4f} (bottom) — {max_loss:.4f} (top)")
        print(f"  LR range:   {lr_start:.2e} → {lr_end:.2e}")
        print(f"  '|' marks suggested LR = {result.suggested_lr:.2e}")
        print()
        print("  " + "+" + "-" * width + "+")
        for row in grid:
            print("  |" + "".join(row) + "|")
        print("  " + "+" + "-" * width + "+")
        print(f"  LR: {lr_start:.1e}" + " " * (width - 16) + f"{lr_end:.1e}")
        print()
