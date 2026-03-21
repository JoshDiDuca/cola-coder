"""Train the domain router model.

Trains a lightweight classifier (<5M params) that routes code prompts
to specialist domains (react, nextjs, graphql, prisma, zod, testing, general).

Usage:
    python scripts/train_router.py --data data/router_training_data.jsonl
    python scripts/train_router.py --data data/router_training_data.jsonl --arch transformer
    python scripts/train_router.py --generate-data --source data/processed/train_data.npy
    python scripts/train_router.py --generate-data --source-dir ./repos/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from cola_coder.cli import cli
from cola_coder.model.config import get_storage_config
from cola_coder.features.router_model import (
    RouterConfig, create_router, DEFAULT_DOMAINS,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RouterDataset(Dataset):
    """Dataset for router training from JSONL data."""

    def __init__(
        self,
        data_path: str,
        domains: list[str] | None = None,
        max_seq_len: int = 256,
        vocab_size: int = 32768,
    ):
        self.domains = domains or DEFAULT_DOMAINS
        self.domain_to_idx = {d: i for i, d in enumerate(self.domains)}
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.samples: list[dict] = []
        self._load(data_path)

    def _load(self, path: str) -> None:
        """Load JSONL data."""
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                domain = sample.get("domain", "general")
                if domain not in self.domain_to_idx:
                    continue
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        domain_idx = self.domain_to_idx[sample["domain"]]

        # If we have token_indices, use them directly
        if "token_indices" in sample:
            tokens = sample["token_indices"][:self.max_seq_len]
        else:
            # Simple character-level encoding as fallback
            code = sample.get("code", "")
            tokens = [ord(c) % self.vocab_size for c in code[:self.max_seq_len]]

        # Pad to max_seq_len
        while len(tokens) < self.max_seq_len:
            tokens.append(0)

        return torch.tensor(tokens, dtype=torch.long), domain_idx


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_router(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
    save_dir: str = "checkpoints/router",
) -> dict:
    """Train the router model.

    Returns:
        Training summary dict with final metrics.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # --- Validate ---
        val_acc = 0.0
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            val_total_loss = 0.0

            with torch.no_grad():
                for input_ids, labels in val_loader:
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                    logits = model(input_ids)
                    loss = F.cross_entropy(logits, labels)
                    val_total_loss += loss.item() * labels.size(0)
                    preds = logits.argmax(dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss = val_total_loss / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

        scheduler.step()

        # Log
        epoch_info = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": round(scheduler.get_last_lr()[0], 8),
        }
        history.append(epoch_info)

        val_str = f"  val_loss={val_loss:.4f}  val_acc={val_acc:.1%}" if val_loader else ""
        cli.substep(
            f"Epoch {epoch:3d}/{epochs}"
            f"  loss={train_loss:.4f}  acc={train_acc:.1%}"
            f"{val_str}"
        )

        # Save best
        if val_loader and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path / "best_router.pt")
            cli.success(f"New best val accuracy: {val_acc:.1%}")

    # Always save final
    torch.save(model.state_dict(), save_path / "router_final.pt")

    # Save config alongside
    if hasattr(model, "config"):
        import dataclasses
        config_dict = dataclasses.asdict(model.config)
        with open(save_path / "router_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    summary = {
        "epochs": epochs,
        "best_val_acc": round(best_val_acc, 4),
        "final_train_acc": history[-1]["train_acc"] if history else 0,
        "save_dir": str(save_path),
    }
    return summary


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def generate_router_data(
    source: str | None = None,
    source_dir: str | None = None,
    output: str = "data/router_training_data.jsonl",
    tokenizer_path: str = "tokenizer.json",
    max_samples: int = 50000,
) -> str:
    """Generate router training data from existing data or source files."""
    from cola_coder.features.router_data_generator import RouterDataGenerator

    gen = RouterDataGenerator(
        min_confidence=0.3,
        max_samples_per_domain=max_samples // 7,
    )

    if source and Path(source).suffix == ".npy":
        return gen.generate_from_npy(source, tokenizer_path, output, max_samples)
    elif source_dir:
        return gen.generate_from_files(source_dir, output)
    else:
        # Generate synthetic data for bootstrap
        return gen.generate_synthetic(output, num_per_domain=500)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    storage = get_storage_config()

    parser = argparse.ArgumentParser(description="Train the domain router model")
    parser.add_argument("--data", type=str, help="Path to router training JSONL")
    parser.add_argument(
        "--generate-data", action="store_true",
        help="Generate training data first",
    )
    parser.add_argument(
        "--source", type=str,
        help="Source .npy file for data generation",
    )
    parser.add_argument(
        "--source-dir", type=str,
        help="Source directory for data generation",
    )
    parser.add_argument(
        "--arch", type=str, default="mlp", choices=["mlp", "transformer"],
        help="Router architecture (default: mlp)",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--save-dir", type=str, default="checkpoints/router")
    parser.add_argument(
        "--tokenizer", type=str, default=storage.tokenizer_path,
        help="Path to tokenizer.json",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "Router Training")

    # Step 1: Generate data if needed
    data_path = args.data
    if args.generate_data or data_path is None:
        if data_path is None:
            data_path = "data/router_training_data.jsonl"
        if not Path(data_path).exists() or args.generate_data:
            cli.info("Step 1", "Generating router training data...")
            data_path = generate_router_data(
                source=args.source,
                source_dir=args.source_dir,
                output=data_path,
                tokenizer_path=args.tokenizer,
            )
        else:
            cli.info("Step 1", f"Using existing data: {data_path}")
    else:
        cli.info("Step 1", f"Using provided data: {data_path}")

    if not Path(data_path).exists():
        cli.error(f"Training data not found: {data_path}")
        sys.exit(1)

    # Count samples
    with open(data_path) as f:
        total = sum(1 for _ in f)
    cli.info("Samples", f"{total:,}")

    if total < 10:
        cli.error("Too few samples for training. Generate more data.")
        sys.exit(1)

    # Step 2: Create dataset
    cli.info("Step 2", "Loading dataset...")
    config = RouterConfig(architecture=args.arch)
    dataset = RouterDataset(data_path, max_seq_len=config.max_seq_len)

    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    cli.info("Train samples", f"{train_size:,}")
    cli.info("Val samples", f"{val_size:,}")

    # Step 3: Create and train model
    cli.info("Step 3", "Training router model...")
    model = create_router(config, architecture=args.arch)

    summary = train_router(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
    )

    cli.rule("Results")
    cli.info("Best val accuracy", f"{summary['best_val_acc']:.1%}")
    cli.info("Final train accuracy", f"{summary['final_train_acc']:.1%}")
    cli.info("Saved to", summary["save_dir"])
    cli.success("Router training complete!")


if __name__ == "__main__":
    main()
