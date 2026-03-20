"""Migrate Cola-Coder data to a new storage location.

Reads configs/storage.yaml and copies/moves data from the project root
(or HuggingFace default cache) to the configured storage paths.

Usage:
    python scripts/migrate_storage.py              # Interactive mode
    python scripts/migrate_storage.py --copy       # Copy (keep originals)
    python scripts/migrate_storage.py --move       # Move (delete originals after)
    python scripts/migrate_storage.py --hf-only    # Only migrate HuggingFace cache
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from cola_coder.cli import cli  # noqa: E402
from cola_coder.model.config import get_storage_config  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dir_size(path: Path) -> int:
    """Total size of a directory in bytes."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _format_size(size_bytes: int) -> str:
    """Human-readable file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.2f} GB"


def _file_count(path: Path) -> int:
    """Count files in a directory."""
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for f in path.rglob("*") if f.is_file())


def _find_hf_cache() -> Path | None:
    """Find the HuggingFace cache directory."""
    # Check env vars first
    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE"):
        val = os.environ.get(var)
        if val:
            p = Path(val)
            if p.exists():
                return p

    # Default location
    default = Path.home() / ".cache" / "huggingface"
    if default.exists():
        return default

    return None


def _copy_tree(src: Path, dst: Path, *, label: str = "") -> bool:
    """Copy a directory tree with progress reporting."""
    if not src.exists():
        cli.warn(f"Source not found: {src}")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    total_files = _file_count(src)
    copied = 0

    prefix = f"[{label}] " if label else ""

    if src.is_file():
        cli.dim(f"{prefix}Copying {src.name}...")
        shutil.copy2(str(src), str(dst))
        return True

    for item in src.rglob("*"):
        if item.is_file():
            rel = item.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(item), str(target))
            copied += 1
            if copied % 100 == 0 or copied == total_files:
                cli.dim(f"{prefix}{copied}/{total_files} files copied...")

    return True


def _verify_copy(src: Path, dst: Path) -> bool:
    """Verify that copy was successful by comparing file counts and sizes."""
    src_count = _file_count(src)
    dst_count = _file_count(dst)
    src_size = _dir_size(src)
    dst_size = _dir_size(dst)

    if src_count == dst_count and src_size == dst_size:
        return True

    cli.warn(
        f"Verification mismatch: src={src_count} files ({_format_size(src_size)}), "
        f"dst={dst_count} files ({_format_size(dst_size)})"
    )
    return False


# ---------------------------------------------------------------------------
# Migration tasks
# ---------------------------------------------------------------------------

def _discover_migrations(project_root: Path) -> list[dict]:
    """Discover what can be migrated.

    Returns list of dicts: {label, src, dst, size, files, type}
    """
    storage = get_storage_config()
    migrations = []

    # Data directory
    src_data = project_root / "data"
    dst_data = Path(storage.data_dir)
    if not dst_data.is_absolute():
        dst_data = project_root / dst_data

    if src_data.exists() and src_data.resolve() != dst_data.resolve():
        migrations.append({
            "label": "Training Data",
            "src": src_data,
            "dst": dst_data,
            "size": _dir_size(src_data),
            "files": _file_count(src_data),
            "type": "dir",
        })

    # Checkpoints
    src_ckpt = project_root / "checkpoints"
    dst_ckpt = Path(storage.checkpoints_dir)
    if not dst_ckpt.is_absolute():
        dst_ckpt = project_root / dst_ckpt

    if src_ckpt.exists() and src_ckpt.resolve() != dst_ckpt.resolve():
        migrations.append({
            "label": "Checkpoints",
            "src": src_ckpt,
            "dst": dst_ckpt,
            "size": _dir_size(src_ckpt),
            "files": _file_count(src_ckpt),
            "type": "dir",
        })

    # Tokenizer
    src_tok = project_root / "tokenizer.json"
    dst_tok = Path(storage.tokenizer_path)
    if not dst_tok.is_absolute():
        dst_tok = project_root / dst_tok

    if src_tok.exists() and src_tok.resolve() != dst_tok.resolve():
        migrations.append({
            "label": "Tokenizer",
            "src": src_tok,
            "dst": dst_tok,
            "size": _dir_size(src_tok),
            "files": 1,
            "type": "file",
        })

    # HuggingFace cache
    hf_cache = _find_hf_cache()
    dst_hf = None
    if storage.hf_cache_dir:
        dst_hf = Path(storage.hf_cache_dir)
        if not dst_hf.is_absolute():
            dst_hf = project_root / dst_hf

    if hf_cache and dst_hf and hf_cache.resolve() != dst_hf.resolve():
        hub_dir = hf_cache / "hub"
        source = hub_dir if hub_dir.exists() else hf_cache
        migrations.append({
            "label": "HuggingFace Cache",
            "src": source,
            "dst": dst_hf / "hub" if hub_dir.exists() else dst_hf,
            "size": _dir_size(source),
            "files": _file_count(source),
            "type": "dir",
        })

    return migrations


def _run_migration(migration: dict, *, move: bool = False) -> bool:
    """Execute a single migration (copy or move)."""
    src = migration["src"]
    dst = migration["dst"]
    label = migration["label"]
    action = "Moving" if move else "Copying"

    cli.step(1, 3, f"{action} {label}...")
    cli.info("From", str(src))
    cli.info("To", str(dst))
    cli.info("Size", _format_size(migration["size"]))
    cli.info("Files", str(migration["files"]))
    cli.print("")

    if migration["type"] == "file":
        dst.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
        cli.success(f"{label} {'moved' if move else 'copied'} successfully.")
        return True

    # Directory copy/move
    success = _copy_tree(src, dst, label=label)
    if not success:
        return False

    cli.step(2, 3, "Verifying integrity...")
    if not _verify_copy(src, dst):
        cli.error("Verification failed! Source preserved.")
        return False

    cli.success("Verification passed — file counts and sizes match.")

    if move:
        cli.step(3, 3, "Removing original...")
        shutil.rmtree(str(src))
        cli.success(f"Original {label.lower()} removed.")
    else:
        cli.step(3, 3, "Done (originals preserved)")
        cli.dim("Delete originals manually when you're sure everything works.")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cli.header("Cola-Coder", "Storage Migration")
    cli.dim("Migrate data, checkpoints, and HuggingFace cache to new storage paths.")
    cli.dim("Configured in: configs/storage.yaml")
    cli.print("")

    project_root = _project_root

    # Show current config
    storage = get_storage_config()
    cli.kv_table({
        "Data dir": storage.data_dir,
        "Checkpoints": storage.checkpoints_dir,
        "Tokenizer": storage.tokenizer_path,
        "HF cache": storage.hf_cache_dir or "(HuggingFace default)",
    }, title="Storage Config")
    cli.print("")

    # Discover what needs migrating
    migrations = _discover_migrations(project_root)

    if not migrations:
        cli.success("Nothing to migrate! All data is already at the configured paths.")
        cli.dim("Edit configs/storage.yaml to change storage locations.")
        return

    # Show what we found
    cli.info("Found", f"{len(migrations)} item(s) to migrate")
    cli.print("")

    for m in migrations:
        cli.info(m["label"], f"{_format_size(m['size'])} ({m['files']} files)")
        cli.dim(f"  {m['src']}  →  {m['dst']}")
    cli.print("")

    # Parse args for non-interactive mode
    parser = argparse.ArgumentParser(description="Migrate Cola-Coder storage")
    parser.add_argument("--copy", action="store_true", help="Copy files (keep originals)")
    parser.add_argument("--move", action="store_true", help="Move files (delete originals)")
    parser.add_argument("--hf-only", action="store_true", help="Only migrate HuggingFace cache")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    args = parser.parse_args()

    if args.hf_only:
        migrations = [m for m in migrations if m["label"] == "HuggingFace Cache"]
        if not migrations:
            cli.warn("No HuggingFace cache to migrate.")
            cli.dim("Set hf_cache_dir in configs/storage.yaml first.")
            return

    # Determine copy vs move
    if args.move:
        move = True
    elif args.copy:
        move = False
    else:
        # Interactive
        options = [
            {"label": "Copy", "detail": "Copy files — originals are preserved"},
            {"label": "Move", "detail": "Move files — originals deleted after verification"},
        ]
        choice = cli.choose("Copy or move?", options, allow_cancel=True)
        if choice is None:
            cli.dim("Cancelled.")
            return
        move = (choice == 1)

    # Select which items to migrate
    if not args.yes and not args.hf_only:
        options = [
            {"label": "Migrate ALL", "detail": f"All {len(migrations)} items"},
        ]
        for m in migrations:
            options.append({
                "label": m["label"],
                "detail": f"{_format_size(m['size'])} ({m['files']} files)",
            })
        choice = cli.choose("What to migrate?", options, allow_cancel=True)
        if choice is None:
            cli.dim("Cancelled.")
            return
        if choice > 0:
            migrations = [migrations[choice - 1]]

    # Confirm
    total_size = sum(m["size"] for m in migrations)
    action_word = "Move" if move else "Copy"
    if not args.yes:
        if not cli.confirm(
            f"{action_word} {_format_size(total_size)} to new location?",
            default=True,
        ):
            cli.dim("Cancelled.")
            return

    # Execute
    cli.rule("Starting migration")
    success_count = 0
    for m in migrations:
        cli.rule(m["label"])
        if _run_migration(m, move=move):
            success_count += 1
        cli.print("")

    cli.done(
        f"Migration complete: {success_count}/{len(migrations)} items migrated.",
        extras={"Action": "Moved" if move else "Copied"},
    )

    if not move:
        cli.dim(
            "Originals are still at the old locations. "
            "Delete them manually once everything is working."
        )


if __name__ == "__main__":
    main()
