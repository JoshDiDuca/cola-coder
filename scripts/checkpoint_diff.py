"""Compare two cola-coder checkpoints: parameter differences, loss progression, config changes.

Shows a diff of configs, loss/perplexity progression, and parameter-level statistics
between two checkpoints.  Reads safetensors weights and metadata.json.  No model
loading or GPU required.

Usage:
    python scripts/checkpoint_diff.py --a checkpoints/tiny/step_00010000 --b checkpoints/tiny/step_00020000
    python scripts/checkpoint_diff.py --a step_00010000 --b step_00020000  # auto-resolves paths
    python scripts/checkpoint_diff.py  # auto-detect last two checkpoints
    python scripts/checkpoint_diff.py --json  # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli
from cola_coder.model.config import get_storage_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_checkpoints_dir() -> Path:
    storage = get_storage_config()
    return Path(storage.checkpoints_dir)


def _resolve_checkpoint(path_str: str, checkpoints_dir: Path) -> Path:
    """Try to find a checkpoint given a partial or full path."""
    p = Path(path_str)
    if p.exists():
        return p.resolve()
    # Try relative to checkpoints_dir/<size>/<name>
    for size_dir in checkpoints_dir.iterdir():
        if not size_dir.is_dir():
            continue
        candidate = size_dir / path_str
        if candidate.exists():
            return candidate.resolve()
    # Last resort: checkpoints_dir itself
    candidate2 = checkpoints_dir / path_str
    if candidate2.exists():
        return candidate2.resolve()
    cli.fatal(f"Checkpoint not found: {path_str}", hint="Provide a full path or a step name.")
    sys.exit(1)


def _read_metadata(ckpt: Path) -> dict:
    meta = ckpt / "metadata.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    return {}


def _list_safetensors_params(ckpt: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    """Return {name: (shape, dtype)} for every tensor in the safetensors file."""
    st_files = list(ckpt.glob("*.safetensors"))
    if not st_files:
        return {}
    try:
        from safetensors import safe_open  # type: ignore[import-untyped]

        result: dict[str, tuple[tuple[int, ...], str]] = {}
        for st_file in st_files:
            with safe_open(str(st_file), framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    result[key] = (tuple(tensor.shape), str(tensor.dtype))
        return result
    except ImportError:
        cli.warn("safetensors not installed — skipping parameter diff")
        return {}


def _param_count(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _fmt_num(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}K"
    return str(n)


def _flat_config(d: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict to dotted-key form."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flat_config(v, prefix=key + "."))
        else:
            out[key] = v
    return out


# ---------------------------------------------------------------------------
# Auto-detect last two checkpoints
# ---------------------------------------------------------------------------


def _auto_detect_two(checkpoints_dir: Path) -> tuple[Path, Path] | None:
    """Find the two most recent step directories across all size dirs."""
    step_dirs: list[Path] = []
    if not checkpoints_dir.exists():
        return None
    for size_dir in checkpoints_dir.iterdir():
        if not size_dir.is_dir():
            continue
        for step_dir in size_dir.iterdir():
            if step_dir.is_dir() and step_dir.name.startswith("step_"):
                step_dirs.append(step_dir)
    step_dirs.sort(key=lambda p: p.name)
    if len(step_dirs) < 2:
        return None
    return step_dirs[-2], step_dirs[-1]


# ---------------------------------------------------------------------------
# Diff logic
# ---------------------------------------------------------------------------


def diff_configs(meta_a: dict, meta_b: dict) -> list[tuple[str, Any, Any]]:
    """Return list of (key, value_a, value_b) for keys that differ."""
    cfg_a = _flat_config(meta_a.get("config", {}))
    cfg_b = _flat_config(meta_b.get("config", {}))
    all_keys = sorted(set(cfg_a) | set(cfg_b))
    diffs: list[tuple[str, Any, Any]] = []
    for k in all_keys:
        va = cfg_a.get(k, "<missing>")
        vb = cfg_b.get(k, "<missing>")
        if va != vb:
            diffs.append((k, va, vb))
    return diffs


def diff_params(
    params_a: dict[str, tuple[tuple[int, ...], str]],
    params_b: dict[str, tuple[tuple[int, ...], str]],
) -> dict:
    """Compare parameter shapes / counts between two checkpoints."""
    keys_a = set(params_a)
    keys_b = set(params_b)

    added = sorted(keys_b - keys_a)
    removed = sorted(keys_a - keys_b)
    shape_changed: list[tuple[str, tuple, tuple]] = []

    for k in sorted(keys_a & keys_b):
        if params_a[k][0] != params_b[k][0]:
            shape_changed.append((k, params_a[k][0], params_b[k][0]))

    total_a = sum(_param_count(v[0]) for v in params_a.values())
    total_b = sum(_param_count(v[0]) for v in params_b.values())

    return {
        "total_params_a": total_a,
        "total_params_b": total_b,
        "added": added,
        "removed": removed,
        "shape_changed": shape_changed,
    }


def build_diff_report(ckpt_a: Path, ckpt_b: Path) -> dict:
    """Build the full diff report between two checkpoints."""
    meta_a = _read_metadata(ckpt_a)
    meta_b = _read_metadata(ckpt_b)

    step_a = meta_a.get("step", 0)
    step_b = meta_b.get("step", 0)
    loss_a = meta_a.get("loss")
    loss_b = meta_b.get("loss")

    ppl_a = math.exp(loss_a) if loss_a else None
    ppl_b = math.exp(loss_b) if loss_b else None

    params_a = _list_safetensors_params(ckpt_a)
    params_b = _list_safetensors_params(ckpt_b)

    param_diff = diff_params(params_a, params_b)
    config_diff = diff_configs(meta_a, meta_b)

    # Loss delta (positive = improvement)
    loss_delta = None
    if loss_a is not None and loss_b is not None:
        loss_delta = loss_a - loss_b

    return {
        "checkpoint_a": str(ckpt_a),
        "checkpoint_b": str(ckpt_b),
        "step_a": step_a,
        "step_b": step_b,
        "loss_a": loss_a,
        "loss_b": loss_b,
        "ppl_a": ppl_a,
        "ppl_b": ppl_b,
        "loss_delta": loss_delta,
        "param_diff": param_diff,
        "config_diff": config_diff,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _display_report(report: dict, json_mode: bool) -> None:
    if json_mode:
        # Convert non-serialisable tuples to lists

        def _make_serial(obj: Any) -> Any:
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, dict):
                return {k: _make_serial(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_make_serial(i) for i in obj]
            return obj

        print(json.dumps(_make_serial(report), indent=2))
        return

    cli.header("Cola-Coder", "Checkpoint Diff")

    # ── Metrics ───────────────────────────────────────────────────────────
    ckpt_a_name = Path(report["checkpoint_a"]).name
    ckpt_b_name = Path(report["checkpoint_b"]).name

    rows: dict[str, str] = {
        "Checkpoint A": f"{ckpt_a_name}  (step {report['step_a']:,})",
        "Checkpoint B": f"{ckpt_b_name}  (step {report['step_b']:,})",
    }
    if report["loss_a"] is not None:
        rows["Loss A"] = f"{report['loss_a']:.4f}"
    if report["loss_b"] is not None:
        rows["Loss B"] = f"{report['loss_b']:.4f}"
    if report["ppl_a"] is not None:
        rows["Perplexity A"] = f"{report['ppl_a']:.2f}"
    if report["ppl_b"] is not None:
        rows["Perplexity B"] = f"{report['ppl_b']:.2f}"
    if report["loss_delta"] is not None:
        arrow = "improved" if report["loss_delta"] > 0 else "worsened"
        rows["Loss delta (A→B)"] = f"{report['loss_delta']:+.4f}  ({arrow})"

    cli.kv_table(rows, title="Training Progress")

    # ── Parameter diff ────────────────────────────────────────────────────
    pd = report["param_diff"]
    total_a = pd["total_params_a"]
    total_b = pd["total_params_b"]
    param_rows: dict[str, str] = {
        "Total params A": _fmt_num(total_a),
        "Total params B": _fmt_num(total_b),
        "Added tensors": str(len(pd["added"])),
        "Removed tensors": str(len(pd["removed"])),
        "Shape changes": str(len(pd["shape_changed"])),
    }
    cli.kv_table(param_rows, title="Parameter Diff")

    if pd["added"]:
        cli.info("Added tensors", ", ".join(pd["added"][:5]) + ("…" if len(pd["added"]) > 5 else ""))
    if pd["removed"]:
        cli.info(
            "Removed tensors",
            ", ".join(pd["removed"][:5]) + ("…" if len(pd["removed"]) > 5 else ""),
        )
    for name, sha, shb in pd["shape_changed"][:5]:
        cli.info("Shape change", f"{name}: {sha} → {shb}")

    # ── Config diff ───────────────────────────────────────────────────────
    config_diff = report["config_diff"]
    if config_diff:
        cli.info("Config changes", f"{len(config_diff)} field(s) differ")
        for key, va, vb in config_diff[:10]:
            cli.dim(f"  {key}: {va!r} → {vb!r}")
    else:
        cli.success("Configs are identical")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two cola-coder checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--a", type=str, default=None, help="First (older) checkpoint path.")
    parser.add_argument("--b", type=str, default=None, help="Second (newer) checkpoint path.")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    args = parser.parse_args()

    checkpoints_dir = _find_checkpoints_dir()

    if args.a and args.b:
        ckpt_a = _resolve_checkpoint(args.a, checkpoints_dir)
        ckpt_b = _resolve_checkpoint(args.b, checkpoints_dir)
    else:
        result = _auto_detect_two(checkpoints_dir)
        if result is None:
            cli.fatal(
                "Could not auto-detect two checkpoints.",
                hint="Pass --a and --b explicitly.",
            )
            sys.exit(1)
        ckpt_a, ckpt_b = result
        if not args.json:
            cli.info("Auto-detected", f"{ckpt_a.name}  vs  {ckpt_b.name}")

    report = build_diff_report(ckpt_a, ckpt_b)
    _display_report(report, json_mode=args.json)


if __name__ == "__main__":
    main()
