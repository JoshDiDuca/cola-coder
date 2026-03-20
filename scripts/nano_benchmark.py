"""Run the nano benchmark on a checkpoint.

Usage:
    python scripts/nano_benchmark.py --checkpoint checkpoints/tiny/latest
"""
import argparse
from pathlib import Path
from cola_coder.cli import cli
from cola_coder.model.config import get_storage_config

def main():
    parser = argparse.ArgumentParser(description="Run nano benchmark")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--smoke-only", action="store_true", help="Run smoke test only (no nano problems)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        cli.fatal(f"Checkpoint not found: {ckpt_path}")

    # Load model
    cli.step(1, 3, "Loading model")

    from cola_coder.training.checkpoint import load_model_for_inference
    from cola_coder.inference.generator import CodeGenerator
    from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer

    storage = get_storage_config()
    model, config = load_model_for_inference(str(ckpt_path))
    tokenizer = CodeTokenizer(storage.tokenizer_path)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    generator = CodeGenerator(model=model, tokenizer=tokenizer, device=device)

    # Run smoke test
    cli.step(2, 3, "Running smoke test")
    from cola_coder.features.smoke_test import SmokeTest
    smoke = SmokeTest()
    smoke.run(generator)

    if not args.smoke_only:
        # Run nano benchmark
        cli.step(3, 3, "Running nano benchmark")
        from cola_coder.features.nano_benchmark import NanoBenchmark
        nano = NanoBenchmark()
        nano.run(generator)

    cli.success("Evaluation complete!")

if __name__ == "__main__":
    main()
