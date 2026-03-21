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
- torch.compile: fuses operations and eliminates Python overhead (~20-40% speedup)
- CUDA optimizations: TF32, cuDNN benchmark, non-blocking transfers

For a TS dev: this is like the main event loop of a server, except instead
of handling requests, it processes batches of training data and updates
the model's weights based on how wrong its predictions were.
"""

import os
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from ..cli import cli
from ..model.config import Config
from ..model.transformer import Transformer
from ..data.dataset import create_dataloader
from .checkpoint import save_checkpoint, load_checkpoint
from .metrics import TrainingMetrics
from .optimizer import create_optimizer, create_scheduler
from .early_stopping import EarlyStopping
from .auto_eval import AutoEvaluator


def _setup_cuda_optimizations():
    """Enable all CUDA performance knobs for maximum training speed."""
    if not torch.cuda.is_available():
        return

    # TF32: uses Tensor Cores for float32 matmuls at ~7x the throughput
    # of pure fp32, with slightly less precision (fine for training).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # cuDNN benchmark: auto-tunes convolution algorithms for your exact
    # input shapes. First batch is slower, then every batch is faster.
    torch.backends.cudnn.benchmark = True

    # Enable flash attention if available (PyTorch 2.0+)
    # This is much faster than standard attention for long sequences.
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


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

        # Enable all CUDA performance knobs before building anything
        _setup_cuda_optimizations()

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

        # torch.compile: fuses ops, eliminates Python overhead (~20-40% speedup)
        # Only available in PyTorch 2.0+ and requires CUDA
        self._compiled = False
        if (
            self.device == "cuda"
            and hasattr(torch, "compile")
            and os.environ.get("COLA_NO_COMPILE") != "1"
        ):
            try:
                # "default" mode: good speedup without extra VRAM overhead.
                # "reduce-overhead" uses CUDA graphs which pre-allocate memory
                # and can cause OOM on 16GB cards.
                self.model = torch.compile(self.model, mode="default")
                self._compiled = True
                print("torch.compile enabled (default mode)")
            except Exception as e:
                print(f"torch.compile not available: {e}")

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
        self.metrics.set_max_steps(config.training.max_steps)

        # Early stopping (disabled by default; enabled via train() kwarg)
        self.early_stopping: EarlyStopping | None = None

        # Auto-evaluator (optional; pass via train() kwarg)
        self.auto_evaluator: AutoEvaluator | None = None

        # Resume from checkpoint if provided
        self.start_step = 0
        if resume_from:
            self.start_step = load_checkpoint(
                resume_from, self.model, self.optimizer, self.scheduler, self.device
            )
            print(f"Resuming from step {self.start_step}")

    def train(
        self,
        data_path: str,
        use_wandb: bool = False,
        early_stopping: EarlyStopping | None = None,
        val_loader=None,
        val_check_every: int = 500,
        auto_evaluator: AutoEvaluator | None = None,
        tokenizer=None,
    ):
        """Run the full training loop.

        Args:
            data_path: Path to preprocessed training data (.npy file).
            use_wandb: Whether to log to Weights & Biases.
            early_stopping: Optional EarlyStopping instance. When provided, the
                trainer evaluates on val_loader every val_check_every steps and
                calls early_stopping.step(val_loss). Training stops if the stopper
                signals should_stop. Pass None (default) to disable.
            val_loader: DataLoader for validation set. Required if early_stopping
                is provided. If None and early_stopping is set, training loss is
                used as the monitored metric instead.
            val_check_every: How often (in optimizer steps) to evaluate on the
                validation set. Only used when early_stopping is set.
        """
        cfg = self.config.training
        self.early_stopping = early_stopping
        if auto_evaluator is not None:
            self.auto_evaluator = auto_evaluator

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

        # Pre-compute constants for the inner loop
        use_amp = self.use_bf16 or self.use_fp16
        amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        accum_steps = cfg.gradient_accumulation
        inv_accum = 1.0 / accum_steps
        is_cuda = self.device == "cuda"

        # Print performance info and VRAM check
        if is_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_vram_gb = props.total_mem / 1e9 if hasattr(props, 'total_mem') else 0
            if not total_vram_gb:
                total_vram_gb = getattr(props, 'total_memory', 0) / 1e9
            print(f"GPU: {gpu_name} ({total_vram_gb:.1f} GB VRAM)")
            print(f"TF32: {torch.backends.cuda.matmul.allow_tf32}")
            print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
            if self._compiled:
                print("torch.compile: ON (first few steps will be slower due to compilation)")
            print(f"Workers: {self.config.data.num_workers}")
            print(f"Batch: {cfg.batch_size} x seq {self.config.model.max_seq_len} "
                  f"= {cfg.batch_size * self.config.model.max_seq_len:,} tok/step")

        for step in tqdm(range(self.start_step, cfg.max_steps), initial=self.start_step,
                         total=cfg.max_steps, desc="Training"):

            # Zero gradients at the start of each accumulation cycle
            self.optimizer.zero_grad(set_to_none=True)

            step_loss = 0.0
            step_tokens = 0

            # Gradient accumulation: process multiple micro-batches
            for micro_step in range(accum_steps):
                batch = next(data_iter)
                # non_blocking=True overlaps CPU→GPU transfer with compute
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                weights = batch.get("weights")

                try:
                    # Forward pass with mixed precision
                    with autocast(
                        device_type=self.device,
                        dtype=amp_dtype,
                        enabled=use_amp,
                    ):
                        loss = self.model.compute_loss(input_ids)

                        if weights is not None:
                            weights = weights.to(self.device, non_blocking=True)
                            loss = loss * weights.mean()

                        scaled_loss = loss * inv_accum

                    # Backward pass (compute gradients)
                    self.scaler.scale(scaled_loss).backward()
                except RuntimeError as e:
                    if "CUBLAS" in str(e) or "out of memory" in str(e).lower():
                        if is_cuda:
                            alloc = torch.cuda.memory_allocated() / 1e9
                            reserved = torch.cuda.memory_reserved() / 1e9
                        else:
                            alloc = reserved = 0
                        raise RuntimeError(
                            f"CUDA OOM during forward/backward pass.\n"
                            f"  VRAM allocated: {alloc:.1f} GB, reserved: {reserved:.1f} GB\n"
                            f"  batch_size={cfg.batch_size}, seq_len={self.config.model.max_seq_len}\n"
                            f"  Fix: reduce batch_size in your config YAML "
                            f"(try {max(1, cfg.batch_size // 2)})"
                        ) from e
                    raise

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
            avg_loss = step_loss * inv_accum
            self.metrics.update(avg_loss, step_tokens)
            total_tokens_seen += step_tokens

            # Log metrics
            current_lr = self.scheduler.get_last_lr()[0]
            log_msg = self.metrics.log(step, current_lr, log_interval=100)
            if log_msg:
                # Use cli.print so Rich markup renders (tqdm.write strips it)
                cli.print(log_msg)

            # Early stopping check
            if (
                self.early_stopping is not None
                and step > 0
                and step % val_check_every == 0
            ):
                if val_loader is not None:
                    val_loss = self._evaluate(val_loader, use_amp, amp_dtype)
                    metric_label = "val_loss"
                    metric_value = val_loss
                else:
                    # Fall back to training loss as the monitored metric
                    metric_value = avg_loss
                    metric_label = "train_loss (no val_loader)"

                cli.print(
                    f"[dim]EarlyStopping check — {metric_label}: "
                    f"[bold]{metric_value:.4f}[/bold][/dim]"
                )
                if self.early_stopping.step(metric_value, model=self.model, step=step):
                    cli.print(
                        f"[bold yellow]Early stopping triggered at step {step}. "
                        f"Best metric: {self.early_stopping.best_score:.4f} "
                        f"at step {self.early_stopping.best_step}.[/bold yellow]"
                    )
                    # Save final checkpoint before stopping
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
                        manifest_info=None,
                    )
                    self.metrics.finish()
                    return

            # Auto-eval check (optional — only runs at configured intervals)
            if (
                self.auto_evaluator is not None
                and tokenizer is not None
                and self.auto_evaluator.should_eval(step)
            ):
                self.auto_evaluator.evaluate(
                    self.model, tokenizer, step, device=self.device
                )
                cli.print(self.auto_evaluator.format_report())
                if self.auto_evaluator.check_regression(self.auto_evaluator.history[-1]):
                    cli.warn(
                        f"Auto-eval regression detected at step {step}! "
                        f"pass@1 {self.auto_evaluator.history[-1].pass_at_1:.1%} vs "
                        f"best {self.auto_evaluator.best_score:.1%}"
                    )

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

    def _evaluate(self, val_loader, use_amp: bool, amp_dtype) -> float:
        """Compute average loss on a validation DataLoader.

        Runs in eval mode (no gradients, no dropout) and averages the loss
        over all batches in val_loader.

        Args:
            val_loader: DataLoader with validation data.
            use_amp: Whether mixed precision is enabled.
            amp_dtype: Mixed precision dtype (bfloat16 or float16).

        Returns:
            Average validation loss (scalar float).
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                with autocast(
                    device_type=self.device,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    loss = self.model.compute_loss(input_ids)
                total_loss += loss.item()
                num_batches += 1

        self.model.train()
        return total_loss / max(num_batches, 1)

    def _infinite_dataloader(self, dataloader):
        """Create an infinite iterator that loops over the dataloader.

        When we reach the end of the dataset, start over (new epoch).
        This is simpler than tracking epochs — we just train for N steps.
        """
        while True:
            for batch in dataloader:
                yield batch
