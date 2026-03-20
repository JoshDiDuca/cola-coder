"""Manifest utilities for training provenance and metadata tracking.

Creates YAML manifest files that record data lineage, training configuration,
and environment info. This makes training runs reproducible and comparable.

For a TS dev: think of manifests like package-lock.json — they capture the
exact state of everything that went into producing an artifact.
"""

import hashlib
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _environment_info() -> dict:
    """Collect current environment info (platform, python, torch, hostname)."""
    env = {
        "platform": sys.platform,
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
    }
    try:
        import torch
        env["torch"] = str(torch.__version__)
        if torch.cuda.is_available():
            env["cuda"] = str(torch.version.cuda or "N/A")
    except ImportError:
        pass
    return env


def _iso_now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hash of a file. Useful for verifying tokenizer identity."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_data_manifest(path: str | Path, **kwargs) -> str:
    """Write a data preparation manifest YAML file.

    Args:
        path: Where to write the manifest (e.g., "data/processed/train.manifest.yaml").
        **kwargs: Manifest sections. Expected keys:
            - output_file, output_size_bytes, num_chunks, chunk_size, total_tokens, dtype
            - dataset, languages, streaming
            - filter_mode, filter_stats (dict with total, kept, top_rejections)
            - tokenizer_path, vocab_size
            - workers, batch_size, max_tokens, wall_time_seconds, throughput_tokens_per_sec
            - total_files

    Returns:
        Path to the written manifest file.
    """
    manifest = {
        "version": "1.0",
        "created": _iso_now(),
        "tool": "cola-coder/prepare_data.py",
    }

    # Data section
    manifest["data"] = {
        "output_file": kwargs.get("output_file", ""),
        "output_size_bytes": kwargs.get("output_size_bytes", 0),
        "num_chunks": kwargs.get("num_chunks", 0),
        "chunk_size": kwargs.get("chunk_size", 0),
        "total_tokens": kwargs.get("total_tokens", 0),
        "dtype": kwargs.get("dtype", "uint16"),
    }

    # Source section
    manifest["source"] = {
        "dataset": kwargs.get("dataset", ""),
        "languages": kwargs.get("languages", []),
    }

    # Filter section
    filter_mode = kwargs.get("filter_mode", "none")
    filter_stats = kwargs.get("filter_stats", {})
    total_processed = filter_stats.get("total", kwargs.get("total_files", 0))
    kept = filter_stats.get("kept", total_processed)
    keep_rate = kept / total_processed if total_processed > 0 else 0.0
    manifest["filter"] = {
        "mode": filter_mode,
        "total_files_processed": total_processed,
        "files_kept": kept,
        "keep_rate": round(keep_rate, 3),
    }
    top_rejections = filter_stats.get("top_rejections")
    if top_rejections:
        manifest["filter"]["top_rejections"] = dict(top_rejections)

    # Tokenizer section
    tokenizer_path = kwargs.get("tokenizer_path", "")
    tok_section = {
        "path": str(tokenizer_path),
        "vocab_size": kwargs.get("vocab_size", 0),
    }
    if tokenizer_path and Path(tokenizer_path).exists():
        tok_section["sha256"] = file_sha256(tokenizer_path)
    manifest["tokenizer"] = tok_section

    # Processing section
    manifest["processing"] = {
        "workers": kwargs.get("workers", 1),
        "batch_size": kwargs.get("batch_size", 64),
        "max_tokens": kwargs.get("max_tokens"),
        "wall_time_seconds": round(kwargs.get("wall_time_seconds", 0), 1),
        "throughput_tokens_per_sec": round(
            kwargs.get("throughput_tokens_per_sec", 0), 0
        ),
    }

    # Environment
    manifest["environment"] = _environment_info()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, default_flow_style=False, sort_keys=False))
    return str(path)


def write_training_manifest(path: str | Path, **kwargs) -> str:
    """Write or update a training manifest YAML file.

    If the file already exists, it is loaded and updated (not overwritten).
    The 'updated' timestamp is refreshed on each call.

    Args:
        path: Where to write the manifest (e.g., "checkpoints/tiny/training_manifest.yaml").
        **kwargs: Manifest sections. Expected keys:
            - model_config: dict of model architecture params
            - training_config: dict of training hyperparams
            - data_path, data_manifest_path
            - step, loss, tokens_seen, epochs_completed
            - loss_history: dict mapping step labels to loss values
            - checkpoint_path: path of the checkpoint just saved

    Returns:
        Path to the written manifest file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing manifest if present (for updates)
    if path.exists():
        manifest = read_manifest(str(path))
        manifest["updated"] = _iso_now()
    else:
        manifest = {
            "version": "1.0",
            "created": _iso_now(),
            "updated": _iso_now(),
            "tool": "cola-coder/train.py",
        }

    # Model section
    model_config = kwargs.get("model_config")
    if model_config:
        manifest["model"] = {
            "architecture": "decoder-only transformer",
            **{k: v for k, v in model_config.items()},
        }

    # Training section
    training_config = kwargs.get("training_config")
    if training_config:
        manifest["training"] = dict(training_config)

    # Data section
    data_path = kwargs.get("data_path")
    if data_path:
        data_section = {"train_file": str(data_path)}
        # Include data provenance from the data manifest if available
        data_manifest_path = kwargs.get("data_manifest_path")
        if data_manifest_path and Path(data_manifest_path).exists():
            data_manifest = read_manifest(str(data_manifest_path))
            data_section["data_manifest"] = str(data_manifest_path)
            if "data" in data_manifest:
                data_section["num_chunks"] = data_manifest["data"].get("num_chunks", 0)
                data_section["chunk_size"] = data_manifest["data"].get("chunk_size", 0)
                data_section["total_tokens"] = data_manifest["data"].get("total_tokens", 0)
        manifest["data"] = data_section

    # Progress section
    step = kwargs.get("step")
    if step is not None:
        progress = manifest.get("progress", {})
        progress["current_step"] = step
        progress["total_steps"] = kwargs.get("max_steps", progress.get("total_steps", 0))
        progress["tokens_seen"] = kwargs.get("tokens_seen", 0)
        progress["epochs_completed"] = round(kwargs.get("epochs_completed", 0.0), 3)

        # Loss history
        loss_history = kwargs.get("loss_history", {})
        if loss_history:
            progress["loss_history"] = dict(loss_history)

        loss = kwargs.get("loss")
        if loss is not None:
            progress["current_loss"] = round(loss, 4)
            best_loss = progress.get("best_loss", float("inf"))
            if loss < best_loss:
                progress["best_loss"] = round(loss, 4)
                progress["best_step"] = step

        manifest["progress"] = progress

    # Checkpoints list
    checkpoint_path = kwargs.get("checkpoint_path")
    if checkpoint_path and step is not None:
        checkpoints = manifest.get("checkpoints", [])
        checkpoints.append({
            "step": step,
            "loss": round(kwargs.get("loss", 0.0), 4),
            "path": str(checkpoint_path),
        })
        manifest["checkpoints"] = checkpoints

    # Hardware section
    hardware = {}
    try:
        import torch
        if torch.cuda.is_available():
            hardware["gpu"] = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            hardware["vram_gb"] = round(vram, 1)
    except ImportError:
        pass
    if hardware:
        manifest["hardware"] = hardware

    # Environment
    manifest["environment"] = _environment_info()

    path.write_text(yaml.safe_dump(manifest, default_flow_style=False, sort_keys=False))
    return str(path)


def read_manifest(path: str | Path) -> dict:
    """Read any manifest YAML file.

    Args:
        path: Path to the manifest YAML file.

    Returns:
        Parsed manifest as a dictionary.

    Raises:
        FileNotFoundError: If the manifest file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}
