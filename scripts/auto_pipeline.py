"""Full Auto Pipeline: detect hardware, pick the best config, run all stages.

One command that profiles the machine (GPUs, VRAM, bf16 support), recommends
the largest model config that safely fits, writes a derived auto-config with
hardware-tuned training overrides, and hands off to full_pipeline.py for
end-to-end execution (collect → prepare → train → SFT → router → reasoning
→ evaluate).

Usage:
    .venv/Scripts/python scripts/auto_pipeline.py --profile-only
    .venv/Scripts/python scripts/auto_pipeline.py --dry-run
    .venv/Scripts/python scripts/auto_pipeline.py --smoke --yes   # minutes-long validation
    .venv/Scripts/python scripts/auto_pipeline.py --yes           # real full-scale run
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from cola_coder.cli import cli
from cola_coder.features.hardware_profiler import (
    generate_auto_config,
    print_hardware_profile,
    print_recommendation,
    profile_hardware,
    recommend_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect hardware, pick the best config, and run the full 10-stage pipeline.",
    )
    parser.add_argument(
        "--profile-only", action="store_true",
        help="Print detected hardware and the recommendation, then exit",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke mode: tiny steps (~minutes) to validate pipeline wiring, not train a model",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would run without executing any stage",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip the confirmation prompt (for scripted/overnight runs)",
    )
    parser.add_argument(
        "--base-config", default=None,
        help="Override the recommended base config (e.g. configs/small.yaml)",
    )
    parser.add_argument(
        "--stages", default=None,
        help="Comma-separated stage numbers to run (default: all, optional stages auto-skip)",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "Full Auto Pipeline")

    profile = profile_hardware()
    print_hardware_profile(profile)

    rec = recommend_config(profile)
    if args.base_config:
        override = Path(args.base_config)
        if not override.exists():
            cli.error(f"Config not found: {override}")
            return 1
        rec.config_name = override.stem
        rec.config_path = str(override)
        rec.reasons.append(f"Base config overridden by --base-config {override}")
    print_recommendation(rec)

    if args.profile_only:
        return 0

    auto_config = generate_auto_config(rec, smoke=args.smoke)
    cli.info("Auto config", str(auto_config))
    if args.smoke:
        cli.warn(
            "Smoke mode: 30 training steps on the recommended config. "
            "Validates wiring only — re-run without --smoke for a real model."
        )

    if not args.yes and not args.dry_run:
        if not cli.confirm("Run the full pipeline with this setup?"):
            cli.dim("Cancelled. Re-run with --profile-only to just inspect hardware.")
            return 0

    venv_python = Path(sys.executable)
    cmd = [str(venv_python), "scripts/full_pipeline.py", "--config", str(auto_config)]
    if args.stages:
        cmd.extend(["--stages", args.stages])
    elif args.smoke:
        # Smoke skips GRPO reasoning (stage 9) as well as the optional
        # stages — far too slow for a wiring check (matches the menu path).
        cmd.extend(["--stages", "1,2,3,5,6,8,10"])
    else:
        cmd.append("--skip-optional")
    if args.dry_run:
        cmd.append("--dry-run")

    cli.dim(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
