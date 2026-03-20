"""Estimate VRAM usage for a model config.

Usage:
    python scripts/vram_estimate.py --config configs/tiny.yaml
    python scripts/vram_estimate.py --config configs/medium.yaml
"""
import argparse
from pathlib import Path
from cola_coder.cli import cli

def main():
    parser = argparse.ArgumentParser(description="Estimate VRAM usage")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    if not Path(args.config).exists():
        cli.fatal(f"Config not found: {args.config}")

    from cola_coder.features.vram_estimator import estimate_vram, print_vram_estimate
    estimate = estimate_vram(config_path=args.config)
    print_vram_estimate(estimate)

if __name__ == "__main__":
    main()
