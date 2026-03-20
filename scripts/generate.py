"""Interactive code generation CLI.

Loads a trained model from a checkpoint and provides a REPL for generating
code interactively. Enter a prompt, press Enter twice to submit, and the
model will generate a continuation.

Usage:
    python scripts/generate.py --checkpoint ./checkpoints/step_00010000 --config configs/small.yaml
    python scripts/generate.py --checkpoint ./checkpoints/step_00010000 --config configs/small.yaml --temperature 0.5
"""

import argparse
from pathlib import Path

from cola_coder.cli import cli
from cola_coder.model.config import get_storage_config


def main():
    storage = get_storage_config()

    parser = argparse.ArgumentParser(
        description="Interactive code generation using a trained cola-coder model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (required).",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (required).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=storage.tokenizer_path,
        help=f"Path to tokenizer.json (default: {storage.tokenizer_path}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature: 0 = greedy, higher = more random (default: 0.8).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (default: 256).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling threshold (default: 0.9).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling threshold (default: 50).",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "Code Generation")

    # ---- Validate inputs ----
    if not Path(args.checkpoint).exists():
        cli.fatal(f"Checkpoint not found: {args.checkpoint}", hint="Check the path")

    if not Path(args.config).exists():
        cli.fatal(f"Config file not found: {args.config}", hint="Check the path")

    if not Path(args.tokenizer).exists():
        cli.fatal(f"Tokenizer file not found: {args.tokenizer}", hint="Check the path")

    # ---- Determine device ----
    device = cli.gpu_info()

    # ---- Load model ----
    cli.print("Loading model...")

    try:
        from cola_coder.model.config import Config
        from cola_coder.model.transformer import Transformer
        from cola_coder.training.checkpoint import load_model_only
        from cola_coder.inference.generator import CodeGenerator
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
    except ImportError:
        cli.fatal(
            "Could not import cola_coder. Make sure the package is installed.",
            hint="Try: pip install -e .",
        )

    try:
        config = Config.from_yaml(args.config)
        cli.info("Model", f"{config.model.total_params_human} parameters")

        tokenizer = CodeTokenizer(args.tokenizer)
        cli.info("Tokenizer", f"{tokenizer.vocab_size} tokens")

        model = Transformer(config.model).to(device)
        load_model_only(args.checkpoint, model, device=device)
        cli.info("Checkpoint", args.checkpoint)
        cli.info("Device", device)

        generator = CodeGenerator(model=model, tokenizer=tokenizer, device=device)
    except Exception as e:
        cli.fatal(f"Loading model: {e}")

    # ---- REPL loop ----
    cli.success("Code generation ready!")
    cli.info("Temperature", args.temperature)
    cli.info("Max tokens", args.max_tokens)
    cli.info("Top-p", f"{args.top_p}, Top-k: {args.top_k}")
    cli.print("\nEnter a prompt, then press Enter on an empty line to submit.")
    cli.print("Press Ctrl+C to exit.\n")

    while True:
        try:
            # Read multiline input
            lines = []
            print(">>> ", end="", flush=True)
            while True:
                line = input()
                if line == "" and lines:
                    break
                lines.append(line)

            if not lines:
                continue

            prompt = "\n".join(lines)

            # Generate
            cli.rule("Generating")
            result = generator.generate(
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

            # Print just the generated part (after the prompt)
            print(result)
            cli.rule("End")
            print()

        except KeyboardInterrupt:
            cli.done("Session ended.")
            break
        except EOFError:
            cli.done("Session ended.")
            break
        except Exception as e:
            cli.error(f"Generation error: {e}")
            print()


if __name__ == "__main__":
    main()
