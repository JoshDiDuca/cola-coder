"""Multi-turn interactive chat CLI.

Loads a trained model and starts a conversational REPL (InteractiveChat) that
keeps history and packs it into the context window. Unlike generate.py (single
prompt → completion), this maintains a running user/assistant dialogue.

Format parity (INFER-011): an SFT checkpoint is trained on ChatML
(<|im_start|>role…<|im_end|>), while a base/pretrain checkpoint expects the
Alpaca-style ### User:/### Assistant: layout. --chat-format auto picks ChatML
for checkpoints whose path contains an `_sft` directory (where train_sft.py
writes), else Alpaca. Override explicitly with --chat-format {alpaca,chatml}.

Usage:
    python scripts/chat.py --checkpoint checkpoints/small_sft/latest --config configs/small.yaml
    python scripts/chat.py --checkpoint checkpoints/small/latest --config configs/small.yaml --chat-format alpaca
"""

import argparse
from pathlib import Path

from cola_coder.cli import cli
from cola_coder.model.config import get_storage_config


def resolve_chat_format(checkpoint: str | Path, override: str = "auto") -> str:
    """Decide the chat format for a checkpoint.

    Args:
        checkpoint: Path to the checkpoint directory.
        override: "auto", "alpaca", or "chatml". A non-auto value wins.

    Returns:
        "chatml" or "alpaca". In "auto" mode, ChatML is chosen when any path
        component ends with "_sft" (the convention train_sft.py uses:
        checkpoints/<size>_sft/step_XXXXXXXX), matching the SFT training format;
        otherwise Alpaca (base/pretrain checkpoint).
    """
    if override in ("alpaca", "chatml"):
        return override
    if override != "auto":
        raise ValueError(f"chat_format must be auto|alpaca|chatml, got {override!r}")
    parts = Path(checkpoint).resolve().parts
    return "chatml" if any(p.endswith("_sft") for p in parts) else "alpaca"


def _resolve_tokenizer(checkpoint: str, config: str, storage) -> str | None:
    """Mirror generate.py's resolution: checkpoint metadata → DatasetResolver →
    storage default. Returns None only if nothing resolves to an existing file."""
    import json

    meta_path = Path(checkpoint) / "metadata.json"
    if meta_path.exists():
        try:
            saved = json.loads(meta_path.read_text(encoding="utf-8")).get("tokenizer_path", "")
            if saved and Path(saved).exists():
                return saved
        except (json.JSONDecodeError, OSError):
            pass
    try:
        from cola_coder.data.dataset_resolver import DatasetResolver

        if DatasetResolver.tokenizer_exists(config_path=config):
            return str(DatasetResolver.get_tokenizer_path(config_path=config))
    except Exception:
        pass
    return storage.tokenizer_path


def main() -> None:
    storage = get_storage_config()

    parser = argparse.ArgumentParser(
        description="Multi-turn interactive chat with a trained cola-coder model.",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory (required).")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file (required).")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer.json (auto-resolved if omitted).")
    parser.add_argument(
        "--chat-format", choices=["auto", "alpaca", "chatml"], default="auto",
        help="Prompt format. 'auto' picks chatml for _sft checkpoints, else "
             "alpaca (default: auto).",
    )
    parser.add_argument("--system", type=str, default="You are a helpful coding assistant.",
                        help="System prompt.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature per turn (default: 0.7).")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max new tokens per assistant turn (default: 256).")
    parser.add_argument("--max-context", type=int, default=1024,
                        help="Max context tokens for history packing (default: 1024).")
    args = parser.parse_args()

    cli.header("Cola-Coder", "Multi-Turn Chat")

    # ---- Validate inputs ----
    if not Path(args.checkpoint).exists():
        cli.fatal(f"Checkpoint not found: {args.checkpoint}", hint="Check the path")
    if not Path(args.config).exists():
        cli.fatal(f"Config file not found: {args.config}", hint="Check the path")

    tokenizer_path = args.tokenizer or _resolve_tokenizer(args.checkpoint, args.config, storage)
    if not tokenizer_path or not Path(tokenizer_path).exists():
        cli.fatal(
            f"Tokenizer not found: {tokenizer_path}",
            hint="Train one (scripts/train_tokenizer.py) or pass --tokenizer <path>",
        )

    chat_format = resolve_chat_format(args.checkpoint, args.chat_format)

    device = cli.gpu_info()
    cli.print("Loading model...")
    try:
        from cola_coder.inference.loading import load_generator

        generator, _config, _tok = load_generator(
            checkpoint=args.checkpoint,
            config_path=args.config,
            tokenizer_path=tokenizer_path,
            device=device,
        )
    except Exception as e:  # noqa: BLE001 — surface any load failure to the user
        cli.fatal(f"Failed to load model: {e}")

    cli.kv_table("Chat session", {
        "Checkpoint": args.checkpoint,
        "Format": chat_format + (" (auto)" if args.chat_format == "auto" else ""),
        "System": args.system,
        "Temperature": args.temperature,
        "Max tokens/turn": args.max_tokens,
    })
    if chat_format == "alpaca":
        cli.dim("Tip: pass --chat-format chatml after instruction-tuning (SFT).")

    from cola_coder.features.multi_turn_chat import InteractiveChat

    chat = InteractiveChat(
        generator=generator,
        system_prompt=args.system,
        max_context_tokens=args.max_context,
        chat_format=chat_format,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    chat.run()


if __name__ == "__main__":
    main()
