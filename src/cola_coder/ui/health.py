"""Project-health checklist for the local UI/dashboard.

A fast, read-only green/amber/red summary of the project's on-disk state.
This module performs only cheap filesystem facts (existence checks, directory
scans, file stats) — it never imports torch, never shells out, and never hits
the network. All functions are best-effort and never raise: a catastrophic
failure degrades to an ``{"error": ...}`` dict, individual checks degrade to a
failing (``ok=False``) entry.
"""

from __future__ import annotations

from pathlib import Path

# Reuse the canonical tokenizer.json discovery from tokenizer_info when it is
# importable. Guarded so a failure degrades to the local filesystem fallback.
try:
    from cola_coder.ui.tokenizer_info import _resolve_tokenizer_file
except Exception:  # pragma: no cover - defensive: import must never crash health
    _resolve_tokenizer_file = None  # type: ignore[assignment]


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _dir_exists(root: Path, rel: str) -> bool:
    try:
        return (root / rel).is_dir()
    except OSError:
        return False


def _has_config(root: Path) -> tuple[bool, str]:
    configs = root / "configs"
    try:
        if not configs.is_dir():
            return False, "configs/ missing"
        yamls = [
            p
            for p in configs.iterdir()
            if p.is_file() and p.suffix in (".yaml", ".yml")
        ]
    except OSError:
        return False, "configs/ unreadable"
    if yamls:
        return True, f"{len(yamls)} config(s) in configs/"
    return False, "no *.yaml in configs/"


def _has_tokenizer(root: Path) -> tuple[bool, str]:
    # Prefer the shared discovery helper (handles storage.yaml + per-dataset
    # locations). It resolves paths relative to the current working directory,
    # so only trust it when ``root`` is the cwd; otherwise fall back to common
    # paths under ``root``.
    if _resolve_tokenizer_file is not None:
        try:
            if root.resolve() == Path(".").resolve():
                resolved = _resolve_tokenizer_file(None)
                if resolved is not None:
                    return True, f"found: {resolved}"
        except Exception:
            pass

    # Filesystem fallback: probe common tokenizer.json locations under root.
    candidates = [
        root / "tokenizer.json",
        root / "tokenizer" / "tokenizer.json",
        root / "tokenizers" / "tokenizer.json",
    ]
    data_root = root / "data"
    try:
        if data_root.is_dir():
            for child in sorted(data_root.iterdir()):
                if child.is_dir():
                    candidates.append(child / "tokenizer.json")
    except OSError:
        pass
    for candidate in candidates:
        try:
            if candidate.is_file():
                return True, f"found: {candidate}"
        except OSError:
            continue
    return False, "no tokenizer.json discoverable"


def _has_checkpoint(root: Path) -> tuple[bool, str]:
    ckpt_root = root / "checkpoints"
    try:
        if not ckpt_root.is_dir():
            return False, "checkpoints/ missing"
        models = [d for d in ckpt_root.iterdir() if d.is_dir()]
    except OSError:
        return False, "checkpoints/ unreadable"
    for model_dir in models:
        try:
            for step_dir in model_dir.iterdir():
                if step_dir.is_dir() and step_dir.name.startswith("step_"):
                    return True, f"found: {step_dir}"
        except OSError:
            continue
    return False, "no step_* checkpoint under checkpoints/"


def _has_training_log(root: Path) -> tuple[bool, str]:
    try:
        logs = sorted(root.glob("*.log"))
    except OSError:
        return False, "could not scan for *.log"
    if logs:
        return True, f"found: {logs[0].name}"
    return False, "no *.log training log present"


def project_health(root: str = ".") -> dict:
    """A quick read-only health checklist. Returns:
      {"score": int,                  # 0-100 overall (fraction of checks passing * 100)
       "checks": [ {"name": str, "ok": bool, "detail": str} ],
       "summary": str}                # one-line, e.g. "8/9 checks OK"

    Checks (cheap, filesystem only): venv present (.venv), key dirs exist
    (src/cola_coder, configs, scripts, tests), at least one config in configs/,
    a tokenizer discoverable, at least one checkpoint under checkpoints/ (any
    step_* dir), data/processed present, and a training log present. Each check
    is independent and never raises. On a catastrophic failure returns
    {"error": "..."}. Never raises.
    """
    try:
        base = Path(root)
        checks: list[dict] = []

        checks.append(
            _check("venv", _dir_exists(base, ".venv"), ".venv present")
            if _dir_exists(base, ".venv")
            else _check("venv", False, ".venv missing")
        )

        for rel in ("src/cola_coder", "configs", "scripts", "tests"):
            ok = _dir_exists(base, rel)
            checks.append(
                _check(rel, ok, f"{rel}/ {'present' if ok else 'missing'}")
            )

        ok, detail = _has_config(base)
        checks.append(_check("configs_has_yaml", ok, detail))

        ok, detail = _has_tokenizer(base)
        checks.append(_check("tokenizer", ok, detail))

        ok, detail = _has_checkpoint(base)
        checks.append(_check("checkpoint", ok, detail))

        ok = _dir_exists(base, "data/processed")
        checks.append(
            _check(
                "data_processed",
                ok,
                f"data/processed {'present' if ok else 'missing'}",
            )
        )

        ok, detail = _has_training_log(base)
        checks.append(_check("training_log", ok, detail))

        n_total = len(checks)
        n_ok = sum(1 for c in checks if c["ok"])
        score = int(round(100 * n_ok / n_total)) if n_total else 0

        return {
            "score": score,
            "checks": checks,
            "summary": f"{n_ok}/{n_total} checks OK",
        }
    except Exception as exc:  # pragma: no cover - defensive: must never raise
        return {"error": str(exc)}
