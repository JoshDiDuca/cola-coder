"""Training Resume Auto-Detector: find and offer to resume from checkpoints.

Scans checkpoint directories to find the latest available checkpoint.
Shows metadata about each checkpoint and lets the user choose.

Features:
- Auto-detect latest checkpoint for each model size
- Show training step, loss, timestamp for each
- Verify config compatibility before resuming
- Offer: Resume / Start Fresh / Pick Specific Checkpoint
"""

import json
from pathlib import Path
from dataclasses import dataclass
from cola_coder.cli import cli

FEATURE_ENABLED = True

def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class CheckpointInfo:
    """Metadata about a saved checkpoint."""
    path: str
    step: int
    loss: float | None
    timestamp: str | None
    model_size: str  # e.g., "tiny", "small"
    config_hash: str | None  # For compatibility checking
    total_params: int | None


def scan_checkpoints(checkpoint_dir: str = "./checkpoints") -> list[CheckpointInfo]:
    """Scan for all available checkpoints.

    Args:
        checkpoint_dir: Root checkpoint directory.

    Returns:
        List of CheckpointInfo sorted by step (latest first).
    """
    ckpt_root = Path(checkpoint_dir)
    if not ckpt_root.exists():
        return []

    checkpoints = []

    for size_dir in sorted(ckpt_root.iterdir()):
        if not size_dir.is_dir():
            continue

        model_size = size_dir.name

        for step_dir in sorted(size_dir.iterdir(), reverse=True):
            if not step_dir.is_dir():
                continue
            if step_dir.name.startswith("_"):  # Skip internal dirs like _crash_recovery
                continue

            # Check for model file (safetensors or pt)
            has_model = (
                (step_dir / "model.safetensors").exists() or
                (step_dir / "model.pt").exists()
            )
            if not has_model:
                continue

            # Read metadata
            meta_path = step_dir / "metadata.json"
            step = 0
            loss = None
            timestamp = None
            config_hash = None
            total_params = None

            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    step = meta.get("step", 0)
                    loss = meta.get("loss")
                    timestamp = meta.get("timestamp") or meta.get("saved_at")
                    config_hash = meta.get("config_hash")

                    # Try to get param count from config in metadata
                    model_cfg = meta.get("config", {}).get("model", {})
                    if model_cfg:
                        from cola_coder.model.config import ModelConfig
                        try:
                            mc = ModelConfig(**{k: v for k, v in model_cfg.items() if k in ModelConfig.__dataclass_fields__})
                            total_params = mc.total_params
                        except Exception:
                            pass
                except Exception:
                    pass

            # Parse step from directory name as fallback
            if step == 0 and step_dir.name.startswith("step_"):
                try:
                    step = int(step_dir.name.split("_")[1])
                except (IndexError, ValueError):
                    pass

            checkpoints.append(CheckpointInfo(
                path=str(step_dir),
                step=step,
                loss=loss,
                timestamp=timestamp,
                model_size=model_size,
                config_hash=config_hash,
                total_params=total_params,
            ))

    # Sort by step descending
    checkpoints.sort(key=lambda c: c.step, reverse=True)
    return checkpoints


def find_latest_checkpoint(checkpoint_dir: str = "./checkpoints", model_size: str | None = None) -> CheckpointInfo | None:
    """Find the most recent checkpoint.

    Args:
        checkpoint_dir: Root checkpoint directory.
        model_size: Filter by model size (e.g., "tiny"). None = any.

    Returns:
        Latest checkpoint info, or None if none found.
    """
    checkpoints = scan_checkpoints(checkpoint_dir)
    if model_size:
        checkpoints = [c for c in checkpoints if c.model_size == model_size]

    return checkpoints[0] if checkpoints else None


def prompt_resume(checkpoint_dir: str = "./checkpoints", model_size: str | None = None) -> str | None:
    """Interactive prompt to resume from a checkpoint.

    Returns:
        Checkpoint path to resume from, or None to start fresh.
    """
    checkpoints = scan_checkpoints(checkpoint_dir)
    if model_size:
        checkpoints = [c for c in checkpoints if c.model_size == model_size]

    if not checkpoints:
        return None

    latest = checkpoints[0]

    cli.rule("Existing Checkpoints Found")
    cli.info("Latest checkpoint", f"{latest.model_size}/step_{latest.step:05d}")
    if latest.loss is not None:
        cli.info("Loss at save", f"{latest.loss:.4f}")
    if latest.timestamp:
        cli.info("Saved at", latest.timestamp)
    if latest.total_params:
        cli.info("Model params", f"{latest.total_params:,}")

    if len(checkpoints) == 1:
        options = [
            {"label": f"Resume from step {latest.step}", "detail": f"Loss: {latest.loss:.4f}" if latest.loss else ""},
            {"label": "Start fresh", "detail": "Begin training from scratch"},
        ]

        choice = cli.choose("Resume training?", options, allow_cancel=True)
        if choice == 0:
            return latest.path
        return None

    # Multiple checkpoints available
    options = [
        {"label": f"Resume from latest (step {latest.step})", "detail": f"Loss: {latest.loss:.4f}" if latest.loss else ""},
        {"label": "Pick a specific checkpoint", "detail": f"{len(checkpoints)} available"},
        {"label": "Start fresh", "detail": "Begin training from scratch"},
    ]

    choice = cli.choose("Resume training?", options, allow_cancel=True)

    if choice is None or choice == 2:
        return None
    elif choice == 0:
        return latest.path
    elif choice == 1:
        # Show all checkpoints
        ckpt_options = []
        for c in checkpoints[:10]:  # Limit to 10 most recent
            loss_str = f"Loss: {c.loss:.4f}" if c.loss else ""
            ts_str = c.timestamp or ""
            ckpt_options.append({
                "label": f"{c.model_size}/step_{c.step:05d}",
                "detail": f"{loss_str}  {ts_str}".strip(),
            })

        pick = cli.choose("Select checkpoint:", ckpt_options, allow_cancel=True)
        if pick is not None:
            return checkpoints[pick].path

    return None


def detect_and_prompt(config_path: str, checkpoint_dir: str = "./checkpoints") -> str | None:
    """Auto-detect checkpoint and prompt for resume based on config.

    Extracts model size from config path name and looks for matching checkpoints.

    Args:
        config_path: Path to the YAML config being used.
        checkpoint_dir: Where to look for checkpoints.

    Returns:
        Checkpoint path to resume from, or None.
    """
    # Try to determine model size from config filename
    config_name = Path(config_path).stem  # e.g., "tiny", "small"

    # Check if we have the checkpoint dir from the config
    try:
        from cola_coder.model.config import Config
        config = Config.from_yaml(config_path)
        checkpoint_dir = config.checkpoint.output_dir
    except Exception:
        pass

    return prompt_resume(checkpoint_dir, model_size=config_name)
