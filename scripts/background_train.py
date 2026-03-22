#!/usr/bin/env python3
"""Background training script — automated GPU-throttled training.

Runs training in the background with GPU clock/power limiting so your
desktop stays responsive. Model output is identical to normal training.

Usage:
    python scripts/background_train.py --config configs/medium.yaml
    python scripts/background_train.py --config configs/medium.yaml --duration 8h
    python scripts/background_train.py --config configs/medium.yaml --stop-at 07:00
    python scripts/background_train.py --config configs/medium.yaml --gpu-clock 1200 --gpu-power 175
    python scripts/background_train.py --config configs/medium.yaml --no-throttle
"""

from __future__ import annotations

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def setup_logging(log_path: str) -> None:
    """Configure file-based logging for headless operation."""
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        str(log_file), maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # Also log to stderr (if running in console for debugging)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    ))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.addHandler(console_handler)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Background training with GPU throttling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML training config (e.g. configs/medium.yaml).",
    )
    parser.add_argument(
        "--data",
        help="Path to training data .npy file. Auto-detects if omitted.",
    )
    parser.add_argument(
        "--duration",
        help="Max training duration (e.g. '8h', '30m', '2h30m').",
    )
    parser.add_argument(
        "--stop-at",
        help="Wall-clock time to stop (e.g. '07:00').",
    )
    parser.add_argument(
        "--gpu-clock", type=int, default=1500,
        help="GPU clock limit in MHz (default: 1500). RTX 4080S boosts to ~2550.",
    )
    parser.add_argument(
        "--gpu-power", type=int, default=200,
        help="GPU power limit in Watts (default: 200). RTX 4080S default is 320W.",
    )
    parser.add_argument(
        "--save-every", type=int, default=1000,
        help="Save checkpoint every N steps (default: 1000).",
    )
    parser.add_argument(
        "--no-throttle", action="store_true",
        help="Disable GPU throttling (run at full speed).",
    )
    parser.add_argument(
        "--log-file", default="logs/background_train.log",
        help="Log file path (default: logs/background_train.log).",
    )

    args = parser.parse_args()

    # Validate config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Setup logging
    setup_logging(args.log_file)
    logger = logging.getLogger("background_trainer")
    logger.info("=" * 60)
    logger.info("Background Training Starting")
    logger.info("=" * 60)
    logger.info("Config: %s", args.config)
    logger.info("GPU clock: %d MHz, Power: %d W", args.gpu_clock, args.gpu_power)
    if args.duration:
        logger.info("Duration: %s", args.duration)
    if args.stop_at:
        logger.info("Stop at: %s", args.stop_at)
    if args.no_throttle:
        logger.info("Throttling: DISABLED")

    # Build config
    from cola_coder.features.background_trainer import (
        BackgroundTrainingConfig, TrainingSession, parse_duration,
    )
    from cola_coder.model.config import Config

    # Resolve checkpoint dir from training config for lock/status files
    training_config = Config.from_yaml(args.config)
    ckpt_dir = Path(training_config.checkpoint.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    bg_config = BackgroundTrainingConfig(
        config_path=args.config,
        data_path=args.data,
        gpu_clock_mhz=args.gpu_clock,
        gpu_power_watts=args.gpu_power,
        save_every_override=args.save_every,
        max_duration_seconds=parse_duration(args.duration) if args.duration else None,
        stop_at_time=args.stop_at,
        no_throttle=args.no_throttle,
        lock_file=str(ckpt_dir / ".background_train.lock"),
        status_file=str(ckpt_dir / ".background_status.json"),
        log_file=args.log_file,
    )

    logger.info("Lock file: %s", bg_config.lock_file)
    logger.info("Status file: %s", bg_config.status_file)

    # Run training session
    session = TrainingSession(bg_config)
    session.run()

    logger.info("Background training script finished.")


if __name__ == "__main__":
    main()
