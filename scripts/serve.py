"""Start the FastAPI inference server.

Loads a trained model and serves it via HTTP for code generation requests.
Swagger documentation is available at /docs when the server is running.

Usage:
    python scripts/serve.py --checkpoint ./checkpoints/step_00010000 --config configs/small.yaml
    python scripts/serve.py --checkpoint ./checkpoints/step_00010000 --config configs/small.yaml --port 8080

Then:
    curl -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt": "def fibonacci(n):", "max_new_tokens": 128}'
"""

import argparse
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Start the cola-coder FastAPI inference server."
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
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000).",
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
        print("Note: No GPU detected. Serving on CPU (slower inference).")

    # ---- Load model ----
    print("Loading model...")

    try:
        from cola_coder.model.config import Config
        from cola_coder.model.transformer import Transformer
        from cola_coder.training.checkpoint import load_model_only
        from cola_coder.inference.generator import CodeGenerator
        from cola_coder.inference.server import create_app
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

    # ---- Create and run server ----
    app = create_app(generator)

    print(f"\nStarting server at http://{args.host}:{args.port}")
    print(f"  API docs: http://{args.host}:{args.port}/docs")
    print(f"  Health check: http://{args.host}:{args.port}/health")
    print(f"\nPress Ctrl+C to stop.\n")

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is not installed.")
        print("  Install it with: pip install uvicorn")
        sys.exit(1)

    try:
        uvicorn.run(app, host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
