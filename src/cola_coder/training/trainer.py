"""Main training loop.

This is the heart of the training pipeline. It orchestrates:
1. Loading data in batches
2. Forward pass (model makes predictions)
3. Loss computation (how wrong were the predictions?)
4. Backward pass (compute gradients — which direction to adjust weights)
5. Optimizer step (actually adjust the weights)
6. Logging and checkpointing

Key techniques for consumer GPUs:
- Mixed precision (bf16/fp16): use half-precision for most operations
- Gradient accumulation: simulate larger batches by accumulating gradients
- Gradient checkpointing: recompute activations instead of storing them
- Gradient clipping: prevent exploding gradients

For a TS dev: this is like the main event loop of a server, except instead
of handling requests, it processes batches of training data and updates
the model's weights based on how wrong its predictions were.
"""

from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from ..model.config import Config
from ..model.transformer import Transformer
from ..data.dataset import create_dataloader
from .checkpoint import save_checkpoint, load_checkpoint
from .metrics import TrainingMetrics
from .optimizer import create_optimizer, create_scheduler


class Trainer:
    """Orchestrates the complete training process."""

    def __init__(self, config: Config, resume_from: str | None = None):
        """
        Args:
            config: Full training configuration.
            resume_from: Path to checkpoint directory to resume from.
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cpu":
            print("WARNING: Training on CPU. This will be extremely slow.")
            print("CUDA GPU is required for practical training.")

        # Build model
        print(f"\n{'='*60}")
        print("Building model...")
        print(config.summary())
        print(f"{'='*60}\n")

        self.model = Transformer(config.model).to(self.device)
        print(f"Model parameters: {self.model.num_parameters:,}")

        # Enable gradient checkpointing if configured
        if config.training.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()
            print("Gradient checkpointing enabled (saves VRAM, slower training)")

        # Create optimizer and scheduler
        self.optimizer = create_optimizer(
            self.model,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        min_lr_ratio = config.training.min_lr / config.training.learning_rate
        self.scheduler = create_scheduler(
            self.optimizer,
            warmup_steps=config.training.warmup_steps,
            max_steps=config.training.max_steps,
            min_lr_ratio=min_lr_ratio,
        )

        # Mixed precision setup
        # bf16: larger dynamic range, no scaler needed (RTX 4080+)
        # fp16: needs GradScaler to prevent underflow (RTX 3080)
        # CPU: disable mixed precision entirely (no amp on CPU)
        if self.device == "cpu":
            self.use_bf16 = False
            self.use_fp16 = False
        else:
            self.use_bf16 = config.training.precision == "bf16"
            self.use_fp16 = config.training.precision == "fp16"
        self.scaler = GradScaler("cuda", enabled=self.use_fp16)

        # Metrics tracker
        self.metrics = TrainingMetrics()

        # Resume from checkpoint if provided
        self.start_step = 0
        if resume_from:
            self.start_step = load_checkpoint(
                resume_from, self.model, self.optimizer, self.scheduler, self.device
            )
            print(f"Resuming from step {self.start_step}")

    def train(self, data_path: str, use_wandb: bool = False):
        """Run the full training loop.

        Args:
            data_path: Path to preprocessed training data (.npy file).
            use_wandb: Whether to log to Weights & Biases.
        """
        cfg = self.config.training

        # Initialize logging
        if use_wandb:
            self.metrics.init_wandb(config=vars(cfg))

        # Create data loader
        # Pass max_seq_len so the dataset truncates chunks if they were
        # prepared with a larger chunk size than the model expects.
        # If a quality-weights file exists next to the data file, weighted
        # training is activated automatically (no flag needed).
        weights_path = str(Path(data_path).with_suffix(".weights.npy"))
        dataloader = create_dataloader(
            data_path,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            max_seq_len=self.config.model.max_seq_len,
            weights_path=weights_path,
        )

        # Create infinite data iterator (loop over dataset forever)
        data_iter = self._infinite_dataloader(dataloader)

        print(f"\nStarting training from step {self.start_step} to {cfg.max_steps}")
        print(f"Batch size: {cfg.batch_size} x {cfg.gradient_accumulation} = "
              f"{cfg.effective_batch_size} effective")
        print(f"Precision: {cfg.precision}")
        print()

        # --- Provenance tracking ---
        # Resolve the data manifest path (sits next to the .npy file)
        data_manifest_path = str(Path(data_path).with_suffix(".manifest.yaml"))
        if not Path(data_manifest_path).exists():
            data_manifest_path = None

        # Track loss at checkpoint intervals for the manifest
        loss_history: dict[str, float] = {}
        total_tokens_seen = 0

        # Try to read total tokens from data manifest for epoch calculation
        total_data_tokens = 0
        if data_manifest_path:
            try:
                from ..manifest import read_manifest
                dm = read_manifest(data_manifest_path)
                total_data_tokens = dm.get("data", {}).get("total_tokens", 0)
            except Exception:
                pass

        self.model.train()  # Set to training mode (enables dropout)

        for step in tqdm(range(self.start_step, cfg.max_steps), initial=self.start_step,
                         total=cfg.max_steps, desc="Training"):

            # Zero gradients at the start of each accumulation cycle
            self.optimizer.zero_grad(set_to_none=True)

            step_loss = 0.0
            step_tokens = 0

            # Gradient accumulation: process multiple micro-batches
            for micro_step in range(cfg.gradient_accumulation):
                batch = next(data_iter)
                input_ids = batch["input_ids"].to(self.device)
                # Quality weights (1.0 when not using weighted training)
                weights = batch.get("weights")

                # Forward pass with mixed precision
                with autocast(
                    device_type=self.device,
                    dtype=torch.bfloat16 if self.use_bf16 else torch.float16,
                    enabled=self.use_bf16 or self.use_fp16,
                ):
                    loss = self.model.compute_loss(input_ids)

                    # Apply per-example quality weights if available.
                    # Higher-quality code contributes more to the loss.
                    if weights is not None:
                        weights = weights.to(self.device)
                        loss = loss * weights.mean()

                    # Divide by accumulation steps so the total gradient is averaged
                    scaled_loss = loss / cfg.gradient_accumulation

                # Backward pass (compute gradients)
                self.scaler.scale(scaled_loss).backward()

                step_loss += loss.item()
                step_tokens += input_ids.numel()

            # Gradient clipping (prevents gradient explosion)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

            # Optimizer step (update weights)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Learning rate schedule step
            self.scheduler.step()

            # Track metrics
            avg_loss = step_loss / cfg.gradient_accumulation
            self.metrics.update(avg_loss, step_tokens)
            total_tokens_seen += step_tokens

            # Log metrics
            current_lr = self.scheduler.get_last_lr()[0]
            log_msg = self.metrics.log(step, current_lr, log_interval=100)
            if log_msg:
                tqdm.write(log_msg)

            # Save checkpoint
            if step > 0 and step % self.config.checkpoint.save_every == 0:
                loss_history[f"step_{step}"] = round(avg_loss, 4)
                epochs = (
                    total_tokens_seen / total_data_tokens
                    if total_data_tokens > 0 else 0.0
                )
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    step=step,
                    loss=avg_loss,
                    config={"model": vars(self.config.model),
                            "training": vars(self.config.training)},
                    output_dir=self.config.checkpoint.output_dir,
                    max_checkpoints=self.config.checkpoint.max_checkpoints,
                    manifest_info={
                        "model_config": vars(self.config.model),
                        "training_config": vars(self.config.training),
                        "data_path": data_path,
                        "data_manifest_path": data_manifest_path,
                        "tokens_seen": total_tokens_seen,
                        "epochs_completed": epochs,
                        "loss_history": loss_history,
                        "max_steps": cfg.max_steps,
                    },
                )

        # Final checkpoint
        loss_history[f"step_{cfg.max_steps}"] = round(avg_loss, 4)
        epochs = (
            total_tokens_seen / total_data_tokens
            if total_data_tokens > 0 else 0.0
        )
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=cfg.max_steps,
            loss=avg_loss,
            config={"model": vars(self.config.model),
                    "training": vars(self.config.training)},
            output_dir=self.config.checkpoint.output_dir,
            max_checkpoints=self.config.checkpoint.max_checkpoints,
            manifest_info={
                "model_config": vars(self.config.model),
                "training_config": vars(self.config.training),
                "data_path": data_path,
                "data_manifest_path": data_manifest_path,
                "tokens_seen": total_tokens_seen,
                "epochs_completed": epochs,
                "loss_history": loss_history,
                "max_steps": cfg.max_steps,
            },
        )

        self.metrics.finish()
        print("\nTraining complete!")

    def _infinite_dataloader(self, dataloader):
        """Create an infinite iterator that loops over the dataloader.

        When we reach the end of the dataset, start over (new epoch).
        This is simpler than tracking epochs — we just train for N steps.
        """
        while True:
            for batch in dataloader:
                yield batch
