"""Interactive code generation CLI.

Loads a trained model from a checkpoint and provides a REPL for generating
code interactively. Enter a prompt, press Enter twice to submit, and the
model will generate a continuation.

Usage:
    python scripts/generate.py --checkpoint ./checkpoints/step_00010000 --config configs/small.yaml
    python scripts/generate.py --checkpoint ./checkpoints/step_00010000 --config configs/small.yaml --temperature 0.5
"""

import argparse
import sys
from pathlib import Path

import torch


def main():
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
        default="tokenizer.json",
        help="Path to tokenizer.json (default: tokenizer.json).",
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

    # ---- Validate inputs ----
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    if not Path(args.tokenizer).exists():
        print(f"Error: Tokenizer file not found: {args.tokenizer}")
        sys.exit(1)

    # ---- Determine device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Note: No GPU detected. Running inference on CPU (slower).")

    # ---- Load model ----
    print("Loading model...")

    try:
        from cola_coder.model.config import Config
        from cola_coder.model.transformer import Transformer
        from cola_coder.training.checkpoint import load_model_only
        from cola_coder.inference.generator import CodeGenerator
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
    except ImportError:
        print("Error: Could not import cola_coder. Make sure the package is installed.")
        print("  Try: pip install -e .")
        sys.exit(1)

    try:
        config = Config.from_yaml(args.config)
        print(f"  Model: {config.model.total_params_human} parameters")

        tokenizer = CodeTokenizer(args.tokenizer)
        print(f"  Tokenizer: {tokenizer.vocab_size} tokens")

        model = Transformer(config.model).to(device)
        load_model_only(args.checkpoint, model, device=device)
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Device: {device}")

        generator = CodeGenerator(model=model, tokenizer=tokenizer, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # ---- REPL loop ----
    print(f"\nCode generation ready!")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Top-p: {args.top_p}, Top-k: {args.top_k}")
    print(f"\nEnter a prompt, then press Enter on an empty line to submit.")
    print(f"Press Ctrl+C to exit.\n")

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
            print("\n--- Generating ---")
            result = generator.generate(
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

            # Print just the generated part (after the prompt)
            print(result)
            print("--- End ---\n")

        except KeyboardInterrupt:
            print("\n\nExiting.")
            break
        except EOFError:
            print("\n\nExiting.")
            break
        except Exception as e:
            print(f"\nGeneration error: {e}\n")


if __name__ == "__main__":
    main()
