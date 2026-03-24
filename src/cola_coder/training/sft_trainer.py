"""Supervised Fine-Tuning (SFT) trainer for instruction tuning.

Standard SFT pipeline:
1. Load pretrained checkpoint
2. Add ChatML tokens to vocabulary
3. Train on instruction data with loss masking
4. Only compute loss on assistant-generated tokens

Key differences from base ``Trainer``:
- Uses ``InstructionDataset`` with per-token labels (not raw autoregressive next-token)
- Loss masking: ``labels == -100`` positions are ignored by ``F.cross_entropy``
- Lower default LR (2e-5 vs ~1e-4 for pretraining)
- Fewer epochs (2-3 vs many pretraining epochs)

Research backing:
- Loss masking is standard in Llama-2-chat, Qwen, Mistral-Instruct SFT recipes
- Short cosine warmup prevents large early gradient steps on a pretrained model
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast  # type: ignore[attr-defined]
from torch.utils.data import DataLoader

from cola_coder.data.instruction_dataset import (
    InstructionCollator,
    InstructionDataset,
    PackedInstructionDataset,
)
from cola_coder.training.checkpoint import save_checkpoint


class SFTTrainer:
    """Supervised Fine-Tuning trainer for instruction tuning.

    Trains a pretrained model to follow instructions using ChatML format.
    Loss is computed exclusively on assistant tokens (``labels != -100``).

    Typical usage::

        from cola_coder.training.sft_trainer import SFTTrainer

        trainer = SFTTrainer(model, tokenizer, config, device="cuda")
        metrics = trainer.train("data/sft_train.jsonl", output_dir="checkpoints/sft")
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        config,
        device: str = "cuda",
    ):
        """
        Args:
            model: Pretrained ``Transformer`` model (already on ``device``).
            tokenizer: ``CodeTokenizer`` instance with ChatML tokens added.
            config: ``Config`` object with ``training`` and ``model`` sub-configs.
            device: ``"cuda"`` or ``"cpu"``.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        # SFT-specific defaults — fall back to config values when present
        self.lr = getattr(config.training, "sft_lr", 2e-5)
        self.epochs = getattr(config.training, "sft_epochs", 3)
        self.warmup_steps = getattr(config.training, "sft_warmup_steps", 100)

    def train(
        self,
        data_path: str | Path,
        output_dir: str | Path = "checkpoints/sft",
        epochs: int | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        gradient_accumulation: int | None = None,
        use_packing: bool = False,
        max_seq_len: int | None = None,
        save_every: int = 500,
        eval_data_path: str | Path | None = None,
    ) -> dict:
        """Run SFT training.

        Args:
            data_path: Path to ChatML JSONL training data.
            output_dir: Directory to save checkpoints.
            epochs: Number of training epochs (default from config / ``sft_epochs``).
            learning_rate: Learning rate (default: ``sft_lr`` → 2e-5).
            batch_size: Micro-batch size (default from ``config.training.batch_size``).
            gradient_accumulation: Gradient accumulation steps.
            use_packing: Pack multiple conversations into one sequence for
                better GPU utilisation.
            max_seq_len: Max sequence length (default from ``config.model.max_seq_len``).
            save_every: Save a checkpoint every N optimiser steps.
            eval_data_path: Optional validation JSONL path (reserved for future use).

        Returns:
            Dict with keys ``losses``, ``learning_rates``, ``final_loss``,
            ``best_loss``, and ``total_steps``.
        """
        from cola_coder.cli import cli

        # ------------------------------------------------------------------
        # Resolve parameters
        # ------------------------------------------------------------------
        epochs = epochs or self.epochs
        learning_rate = learning_rate or self.lr
        batch_size = batch_size or self.config.training.batch_size
        gradient_accumulation = (
            gradient_accumulation or self.config.training.gradient_accumulation
        )
        max_seq_len = max_seq_len or self.config.model.max_seq_len
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Dataset & DataLoader
        # ------------------------------------------------------------------
        cli.info("Dataset", str(data_path))
        if use_packing:
            dataset: InstructionDataset | PackedInstructionDataset = PackedInstructionDataset(
                data_path, self.tokenizer, max_seq_len=max_seq_len
            )
            cli.info("Packing", f"{len(dataset)} packed sequences")
        else:
            dataset = InstructionDataset(
                data_path, self.tokenizer, max_seq_len=max_seq_len
            )
            cli.info("Examples", str(len(dataset)))

        collator = InstructionCollator(pad_id=self.tokenizer.pad_id or 0)

        dataloader: DataLoader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,  # Avoid Windows multiprocessing / pickle issues
            drop_last=True,
        )

        # ------------------------------------------------------------------
        # Optimiser
        # ------------------------------------------------------------------
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.95),
        )

        # ------------------------------------------------------------------
        # LR scheduler — cosine decay with linear warmup
        # ------------------------------------------------------------------
        warmup_steps = self.warmup_steps
        total_steps = len(dataloader) * epochs // gradient_accumulation

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.1 + 0.9 * (1 + math.cos(progress * math.pi)) / 2

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # ------------------------------------------------------------------
        # Mixed precision
        # ------------------------------------------------------------------
        precision = self.config.training.precision
        use_amp = precision in ("bf16", "fp16")
        amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        scaler: torch.amp.GradScaler | None = None
        if precision == "fp16":
            scaler = torch.amp.GradScaler("cuda")

        # ------------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------------
        self.model.train()
        global_step = 0
        total_loss = 0.0
        best_loss = float("inf")
        metrics: dict = {"losses": [], "learning_rates": []}

        cli.header("SFT Training", f"{epochs} epochs · lr={learning_rate:.2e}")

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                device_type = "cuda" if "cuda" in self.device else "cpu"
                with autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                    logits = self.model(input_ids)

                    # Shift for next-token prediction
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()

                    # F.cross_entropy ignores positions where label == -100
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                    loss = loss / gradient_accumulation

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                step_loss = loss.item() * gradient_accumulation
                epoch_loss += step_loss

                # Gradient accumulation step
                if (batch_idx + 1) % gradient_accumulation == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.training.grad_clip
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.training.grad_clip
                        )
                        optimizer.step()

                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                    total_loss += step_loss
                    metrics["losses"].append(step_loss)
                    metrics["learning_rates"].append(scheduler.get_last_lr()[0])

                    if global_step % 10 == 0:
                        avg_loss = total_loss / global_step
                        lr_now = scheduler.get_last_lr()[0]
                        cli.step(
                            global_step,
                            total_steps,
                            f"loss={step_loss:.4f} avg={avg_loss:.4f} lr={lr_now:.2e}",
                        )

                    if global_step % save_every == 0:
                        avg = total_loss / global_step
                        ckpt_dir = output_dir / f"step_{global_step:06d}"
                        save_checkpoint(
                            self.model,
                            optimizer,
                            scheduler,
                            global_step,
                            avg,
                            vars(self.config) if hasattr(self.config, "__dict__") else {},
                            str(ckpt_dir),
                        )
                        if avg < best_loss:
                            best_loss = avg
                            cli.success(f"New best loss: {best_loss:.4f}")

            avg_epoch_loss = epoch_loss / max(len(dataloader), 1)
            cli.info(f"Epoch {epoch + 1}/{epochs}", f"loss={avg_epoch_loss:.4f}")

        # ------------------------------------------------------------------
        # Save final checkpoint
        # ------------------------------------------------------------------
        final_dir = output_dir / "final"
        avg_final = total_loss / max(global_step, 1)
        save_checkpoint(
            self.model,
            optimizer,
            scheduler,
            global_step,
            avg_final,
            vars(self.config) if hasattr(self.config, "__dict__") else {},
            str(final_dir),
        )

        cli.done(
            "SFT training complete",
            {
                "Steps": str(global_step),
                "Final loss": f"{avg_final:.4f}",
                "Best loss": f"{best_loss:.4f}",
                "Checkpoint": str(final_dir),
            },
        )

        metrics["final_loss"] = avg_final
        metrics["best_loss"] = best_loss
        metrics["total_steps"] = global_step

        return metrics
