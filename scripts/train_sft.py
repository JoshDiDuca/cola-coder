"""Supervised fine-tuning (SFT) training script.

Loads a pre-trained cola-coder checkpoint, adds ChatML tokens, and
fine-tunes on instruction-following data in JSONL format.  The JSONL
file should contain one JSON object per line with a ``"messages"`` key
(see ``scripts/generate_sft_data.py`` for generating training data).

Usage:
    python scripts/train_sft.py \
        --data data/sft_train.jsonl \
        --config configs/small.yaml \
        --checkpoint checkpoints/small/latest

    python scripts/train_sft.py \
        --data data/sft_train.jsonl \
        --config configs/4080_max.yaml \
        --checkpoint checkpoints/4080_max/latest \
        --epochs 5 --lr 1e-5 --wandb
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from cola_coder.cli import cli
from cola_coder.data.sft_dataset import SFTCollator, SFTDataset
from cola_coder.model.config import Config
from cola_coder.model.transformer import Transformer
from cola_coder.tokenizer.chat_template import add_chat_tokens
from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
from cola_coder.training.checkpoint import load_model_only, save_checkpoint
from cola_coder.training.optimizer import create_optimizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supervised fine-tuning on instruction-following data."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the SFT JSONL training file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config YAML (e.g. configs/small.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pre-trained checkpoint directory to fine-tune.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output checkpoint directory. "
            "Defaults to checkpoints/<size>_sft/ derived from config."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs (default: 3).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Micro batch size (default: 4).",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4).",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of steps used for LR warmup (default: 0.1).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help=(
            "Weight decay (default: 0.01 — standard SFT value, lower than "
            "pretraining to limit drift from the base model)."
        ),
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Max sequence length (default: from config).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log metrics to Weights & Biases.",
    )

    args = parser.parse_args()

    cli.header("Cola-Coder", "Supervised Fine-Tuning (SFT)")

    # ---- Validate inputs ----
    if not Path(args.data).exists():
        cli.fatal(
            f"Training data not found: {args.data}",
            hint="Generate data with: python scripts/generate_sft_data.py",
        )

    if not Path(args.config).exists():
        cli.fatal(f"Config file not found: {args.config}")

    if not Path(args.checkpoint).exists():
        cli.fatal(
            f"Checkpoint not found: {args.checkpoint}",
            hint="Train a base model first with scripts/train.py.",
        )

    # ---- Device ----
    device = cli.gpu_info()

    # ---- Step 1: Load config, tokenizer, model ----
    cli.step(1, 5, "Loading config and model")

    try:
        config = Config.from_yaml(args.config)
    except Exception as e:
        cli.fatal(f"Loading config: {e}")

    max_seq_len = args.max_seq_len or config.model.max_seq_len
    cli.info("Config", args.config)
    cli.info("Model", f"{config.model.total_params_human} parameters")
    cli.info("Max seq len", max_seq_len)

    # Resolve tokenizer path from config or default
    tokenizer_path = "tokenizer.json"
    if hasattr(config, "tokenizer"):
        tp = getattr(config.tokenizer, "path", None)
        if tp:
            tokenizer_path = tp
    if not Path(tokenizer_path).exists():
        # Try storage config
        from cola_coder.model.config import get_storage_config
        storage = get_storage_config()
        try:
            from cola_coder.data.dataset_resolver import DatasetResolver
            tokenizer_path = (
                str(DatasetResolver.get_tokenizer_path())
                if DatasetResolver.tokenizer_exists()
                else storage.tokenizer_path
            )
        except Exception:
            tokenizer_path = storage.tokenizer_path
    if not Path(tokenizer_path).exists():
        cli.fatal(
            f"Tokenizer not found: {tokenizer_path}",
            hint="Run scripts/train_tokenizer.py first.",
        )

    tokenizer = CodeTokenizer(tokenizer_path)
    cli.info("Tokenizer vocab", tokenizer.vocab_size)

    model = Transformer(config.model).to(device)
    load_model_only(args.checkpoint, model, device=device)
    cli.info("Checkpoint", args.checkpoint)

    # ---- Step 2: Add ChatML tokens ----
    cli.step(2, 5, "Adding ChatML tokens")
    add_chat_tokens(tokenizer, model)
    model = model.to(device)

    # ---- Step 3: Create dataset & dataloader ----
    cli.step(3, 5, "Loading SFT dataset")

    dataset = SFTDataset(args.data, tokenizer, max_seq_len=max_seq_len)
    cli.info("Training examples", len(dataset))

    if len(dataset) == 0:
        cli.fatal(
            "Dataset is empty after tokenization.",
            hint="Check that your JSONL file has valid ChatML messages.",
        )

    pad_id = tokenizer.pad_id if tokenizer.pad_id is not None else 0
    collator = SFTCollator(pad_id=pad_id)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # ---- Step 4: Optimizer & scheduler ----
    cli.step(4, 5, "Setting up optimizer and scheduler")

    # Shared optimizer factory: biases and norm weights are excluded from
    # weight decay, same as pretraining.
    optimizer = create_optimizer(
        model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    effective_batch = args.batch_size * args.gradient_accumulation
    steps_per_epoch = math.ceil(len(dataset) / effective_batch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        # (current_step + 1): step/warmup would return 0 at step 0,
        # wasting the first optimizer update at LR=0.
        if current_step < warmup_steps:
            return (current_step + 1) / max(warmup_steps, 1)
        progress = (current_step - warmup_steps) / max(
            total_steps - warmup_steps, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    cli.info("Learning rate", args.lr)
    cli.info("Effective batch", effective_batch)
    cli.info("Steps per epoch", steps_per_epoch)
    cli.info("Total steps", total_steps)
    cli.info("Warmup steps", warmup_steps)

    # Resolve output directory
    output_dir = args.output
    if output_dir is None:
        cfg_name = Path(args.config).stem
        output_dir = f"checkpoints/{cfg_name}_sft"
    cli.info("Output dir", output_dir)

    # Persist the chat-token-expanded tokenizer next to the checkpoint.
    # add_chat_tokens() extended the in-memory tokenizer (and the model's vocab)
    # with <|im_start|>/<|im_end|>. Without saving it, inference reloads the BASE
    # tokenizer.json — which lacks those tokens — fragments the ChatML role
    # markers, and can neither feed nor decode the ids the model trained on: a
    # silent train/inference mismatch that breaks instruction following. The
    # tokenizer_path recorded in metadata.json (via save_checkpoint below) is
    # what inference's resolve_tokenizer_path() reads back first.
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sft_tokenizer_path = str(Path(output_dir) / "tokenizer.json")
    tokenizer.tokenizer.save(sft_tokenizer_path)
    cli.info("SFT tokenizer", sft_tokenizer_path)

    # wandb
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="cola-coder-sft",
                config={
                    "data": args.data,
                    "config": args.config,
                    "checkpoint": args.checkpoint,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "gradient_accumulation": args.gradient_accumulation,
                    "warmup_ratio": args.warmup_ratio,
                    "max_seq_len": max_seq_len,
                    "model_params": config.model.total_params_human,
                },
            )
            cli.info("wandb", wandb_run.url)
        except ImportError:
            cli.warn("wandb not installed. Skipping logging.")
        except Exception as e:
            cli.warn(f"wandb init failed: {e}")

    # ---- Step 5: Training loop ----
    cli.step(5, 5, "Training")

    # Precision from config — bf16 (RTX 4080+, no scaler), fp16 (RTX 3080,
    # needs GradScaler against underflow), or full fp32 on CPU. Previously
    # hardcoded to bf16, which silently ignored `precision: fp16` configs.
    precision = getattr(config.training, "precision", "bf16")
    use_bf16 = device == "cuda" and precision == "bf16"
    use_fp16 = device == "cuda" and precision == "fp16"
    use_amp = use_bf16 or use_fp16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler("cuda", enabled=use_fp16)
    grad_clip = getattr(config.training, "grad_clip", 1.0)
    cli.info("Precision", precision if device == "cuda" else "fp32 (CPU)")
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    global_step = 0
    best_loss = float("inf")

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_loss = 0.0
            epoch_tokens = 0
            optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(
                    device, non_blocking=True
                )
                labels = batch["labels"].to(device, non_blocking=True)

                try:
                    with autocast(
                        device_type=device,
                        dtype=amp_dtype,
                        enabled=use_amp,
                    ):
                        # Forward pass: get logits from model
                        logits = model(input_ids)

                        # Shift for next-token prediction:
                        # logits[:-1] predicts labels[1:]
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = labels[:, 1:].contiguous()

                        loss = loss_fn(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                        )
                        scaled_loss = loss / args.gradient_accumulation

                    scaler.scale(scaled_loss).backward()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        cli.error(f"GPU OOM: {e}")
                        cli.dim(
                            "Reduce --batch-size or --max-seq-len "
                            "and try again."
                        )
                        sys.exit(1)
                    raise

                epoch_loss += loss.item()
                epoch_tokens += (labels != -100).sum().item()

                # Gradient accumulation step
                if (batch_idx + 1) % args.gradient_accumulation == 0 or (
                    batch_idx + 1
                ) == len(dataloader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip
                    )
                    # fp16 only: don't advance the LR schedule when the
                    # GradScaler skips the optimizer step on inf/NaN grads.
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= scale_before:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    # Log progress
                    if global_step % 10 == 0 or global_step == 1:
                        current_lr = scheduler.get_last_lr()[0]
                        avg = epoch_loss / (batch_idx + 1)
                        cli.print(
                            f"  [dim]epoch {epoch}/{args.epochs}  "
                            f"step {global_step}/{total_steps}  "
                            f"loss {avg:.4f}  "
                            f"lr {current_lr:.2e}[/dim]"
                        )

                    if wandb_run is not None:
                        wandb_run.log({
                            "loss": loss.item(),
                            "lr": scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "step": global_step,
                            "tokens": epoch_tokens,
                        })

            # End of epoch
            avg_epoch_loss = epoch_loss / max(len(dataloader), 1)
            cli.success(
                f"Epoch {epoch}/{args.epochs} complete  "
                f"avg_loss={avg_epoch_loss:.4f}  "
                f"tokens={epoch_tokens:,}"
            )

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss

            # Save checkpoint after each epoch
            dummy_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda _: 1.0
            )
            dummy_scheduler.load_state_dict(scheduler.state_dict())

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=dummy_scheduler,
                step=global_step,
                loss=avg_epoch_loss,
                config={
                    "model": vars(config.model),
                    "sft": True,
                    "epoch": epoch,
                    "lr": args.lr,
                },
                output_dir=output_dir,
                tokenizer_path=sft_tokenizer_path,
            )

    except KeyboardInterrupt:
        cli.warn("Training interrupted by user.")
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    cli.done("SFT training complete", extras={
        "Epochs": str(args.epochs),
        "Best loss": f"{best_loss:.4f}",
        "Checkpoint": output_dir,
        "Total steps": str(global_step),
    })


if __name__ == "__main__":
    main()
