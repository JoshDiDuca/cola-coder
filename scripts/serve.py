"""Start the FastAPI inference server.

Loads a trained model and serves it via HTTP for code generation requests.
Provides OpenAI-compatible endpoints alongside the original cola-coder API.
Swagger documentation is available at /docs when the server is running.

Usage:
    python scripts/serve.py --checkpoint ./checkpoints/small/latest --config configs/small.yaml
    python scripts/serve.py --checkpoint ./checkpoints/small/latest --config configs/small.yaml --cors
    python scripts/serve.py --checkpoint ./checkpoints/small/latest --config configs/small.yaml \
        --repo /path/to/project --enable-thinking --cors

Then:
    curl -X POST http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"def fibonacci(n):"}], "max_tokens":128}'
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
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Path to project root for repo context scanning. "
        "Enables the /v1/context endpoint and context-aware generation.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking tokens (<think>/<\\/think>) for chain-of-thought.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for /v1/models (default: derived from config filename).",
    )
    parser.add_argument(
        "--cors",
        action="store_true",
        help="Enable CORS (allow all origins). Needed for VS Code extension.",
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

    if args.repo and not Path(args.repo).exists():
        cli.fatal(f"Repo root not found: {args.repo}", hint="Check the path")

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

        # ---- Optional: add thinking tokens ----
        if args.enable_thinking:
            from cola_coder.reasoning.thinking_tokens import add_thinking_tokens

            think_open_id, think_close_id = add_thinking_tokens(tokenizer, model)
            cli.info("Thinking tokens", f"<think>={think_open_id} </think>={think_close_id}")

        # ---- Create generator ----
        generator = CodeGenerator(model=model, tokenizer=tokenizer, device=device)

        # ---- Optional: wrap with repo context ----
        if args.repo:
            from cola_coder.inference.context_generator import ContextAwareGenerator

            repo_root = Path(args.repo)
            generator = ContextAwareGenerator(generator, repo_root, eager_scan=True)
            cli.info("Repo context", str(repo_root))
    except Exception as e:
        cli.fatal(f"Loading model: {e}")

    # ---- Derive model name ----
    model_name = args.model_name
    if model_name is None:
        config_stem = Path(args.config).stem
        model_name = f"cola-coder-{config_stem}"

    # ---- Create and run server ----
    app = create_app(
        generator,
        config=config,
        model_name=model_name,
        enable_thinking=args.enable_thinking,
        enable_cors=args.cors,
    )

    cli.success(f"Starting server at http://{args.host}:{args.port}")
    cli.info("Model name", model_name)
    cli.info("API docs", f"http://{args.host}:{args.port}/docs")
    cli.info("OpenAI-compat", f"http://{args.host}:{args.port}/v1/chat/completions")
    cli.info("Health check", f"http://{args.host}:{args.port}/health")
    if args.cors:
        cli.info("CORS", "enabled (all origins)")
    if args.repo:
        cli.info("Context endpoint", f"http://{args.host}:{args.port}/v1/context")
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
