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
from pathlib import Path

from cola_coder.cli import cli
from cola_coder.model.config import get_storage_config


def main():
    storage = get_storage_config()

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
        default=storage.tokenizer_path,
        help=f"Path to tokenizer.json (default: {storage.tokenizer_path}).",
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

    cli.header("Cola-Coder", "Inference Server")

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
        from cola_coder.inference.server import create_app
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

    # ---- Create and run server ----
    app = create_app(generator)

    cli.success(f"Starting server at http://{args.host}:{args.port}")
    cli.info("API docs", f"http://{args.host}:{args.port}/docs")
    cli.info("Health check", f"http://{args.host}:{args.port}/health")
    cli.print("\nPress Ctrl+C to stop.\n")

    try:
        import uvicorn
    except ImportError:
        cli.fatal("uvicorn is not installed.", hint="Install it with: pip install uvicorn")

    try:
        uvicorn.run(app, host=args.host, port=args.port)
    except KeyboardInterrupt:
        cli.done("Server stopped.")


if __name__ == "__main__":
    main()
