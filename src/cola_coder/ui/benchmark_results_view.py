"""Benchmark-results endpoint helper for the local cola-coder UI.

Read-only viewer of *past* throughput / latency benchmark artifacts — the
persisted output of the CLI benchmark scripts. It NEVER runs a benchmark (that
needs the GPU the live trainer is using); it only scans the filesystem for JSON
reports that were written earlier.

Where the artifacts come from (verified against the real scripts):

- ``scripts/inference_benchmark.py`` is the only one that persists results, and
  only when ``--json``/``--output`` is given. It writes a JSON dict shaped::

      {"checkpoint": str, "device": str, "total_sec": float,
       "runs": [{"label", "category", "param_name", "param_value",
                 "tokens_per_sec", "latency_ms_first_token", "total_tokens",
                 "duration_sec", "error"}, ...]}

  The ``--output`` path is user-chosen (no fixed dir), so we scan the
  conventional drop spots: ``benchmarks/``, ``results/``, ``benchmark_results/``
  and the project root, plus ``checkpoints/<model>/`` (a natural place to save a
  report next to its checkpoint).
- ``scripts/benchmark.py`` and ``scripts/nano_benchmark.py`` only print to the
  console — they persist nothing — so they contribute no artifacts here.

Each discovered report is summarized into one :class:`BenchmarkRun`-shaped dict
(best run's throughput, best first-token latency). All functions are best-effort
and never raise: a genuinely broken discovery returns ``{"error": ...}``;
finding no artifacts is NOT an error (an empty list is returned).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Conventional directories a user might point ``--output`` at, relative to root.
_SCAN_DIRS: tuple[str, ...] = (
    ".",
    "benchmarks",
    "results",
    "benchmark_results",
)

# Checkpoint roots — a report may be saved alongside the checkpoint it measured.
_CKPT_DIRS: tuple[str, ...] = ("checkpoints",)

# Filename substrings that mark a file as a likely benchmark artifact. Keeps the
# scan cheap and avoids parsing every unrelated JSON in the project root.
_NAME_HINTS: tuple[str, ...] = ("bench", "benchmark", "throughput", "latency")

_VALID_KINDS: frozenset[str] = frozenset({"throughput", "latency", "nano", "unknown"})


def _read_json(path: Path) -> object | None:
    """Parse a JSON file, or return ``None`` on any read/decode failure."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _as_float(value: object) -> float | None:
    """Coerce a JSON value to ``float``, or ``None`` (bool excluded)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_str(value: object) -> str | None:
    """Return ``value`` if it is a non-empty string, else ``None``."""
    if isinstance(value, str) and value:
        return value
    return None


def _classify(payload: dict, runs: list[dict]) -> str:
    """Infer a kind label from the report's run categories.

    ``inference_benchmark.py`` tags each run with a ``category`` ("temperature",
    "seq_len", "batch_size", "precision"). A report that carries first-token
    latencies is summarized as ``latency``; otherwise the throughput sweep is the
    headline, so ``throughput``. ``nano`` is reserved for nano-benchmark style
    score reports, ``unknown`` when nothing matches.
    """
    has_latency = any(_as_float(r.get("latency_ms_first_token")) is not None for r in runs)
    has_throughput = any(_as_float(r.get("tokens_per_sec")) is not None for r in runs)
    if "nano" in str(payload.get("kind", "")).lower():
        return "nano"
    if has_throughput:
        return "throughput"
    if has_latency:
        return "latency"
    return "unknown"


def _best_throughput(runs: list[dict]) -> float | None:
    """Best (max) non-error tokens/sec across the report's runs, or ``None``."""
    values = [
        v
        for r in runs
        if not r.get("error")
        for v in (_as_float(r.get("tokens_per_sec")),)
        if v is not None and v > 0
    ]
    return max(values) if values else None


def _best_latency(runs: list[dict]) -> float | None:
    """Best (min) non-error first-token latency in ms, or ``None``."""
    values = [
        v
        for r in runs
        if not r.get("error")
        for v in (_as_float(r.get("latency_ms_first_token")),)
        if v is not None and v > 0
    ]
    return min(values) if values else None


def _is_benchmark_report(raw: object) -> bool:
    """True if a parsed payload looks like an inference-benchmark report.

    The telltale shape is a dict carrying a ``runs`` list together with a
    ``checkpoint`` or ``device`` field — exactly what ``inference_benchmark.py``
    serializes. This filters out unrelated JSON (configs, metadata, etc.).
    """
    if not isinstance(raw, dict):
        return False
    if not isinstance(raw.get("runs"), list):
        return False
    return "checkpoint" in raw or "device" in raw


def _summarize(raw: dict, path: Path, mtime: float) -> dict:
    """Collapse one benchmark report into a single ``BenchmarkRun``-shaped dict."""
    runs_raw = raw.get("runs")
    runs: list[dict] = [r for r in runs_raw if isinstance(r, dict)] if isinstance(runs_raw, list) else []
    kind = _classify(raw, runs)
    if kind not in _VALID_KINDS:
        kind = "unknown"
    return {
        "name": path.name,
        "path": str(path),
        "kind": kind,
        "tokens_per_s": _best_throughput(runs),
        "latency_ms": _best_latency(runs),
        "config": _as_str(raw.get("config")),
        "checkpoint": _as_str(raw.get("checkpoint")),
        "mtime": mtime,
    }


def _candidate_files(root_path: Path) -> list[Path]:
    """Find JSON files that may be benchmark artifacts under ``root``.

    Scans the conventional output dirs (and the project root, name-filtered so
    the root scan stays cheap) plus ``checkpoints/<model>/``. Missing dirs are
    silently ignored. Duplicates (same resolved path) are de-duplicated.
    """
    seen: set[Path] = set()
    files: list[Path] = []

    def _add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            return
        seen.add(resolved)
        files.append(path)

    for dirname in _SCAN_DIRS:
        scan_dir = root_path / dirname
        if not scan_dir.is_dir():
            continue
        root_level = dirname == "."
        try:
            entries = [p for p in scan_dir.iterdir() if p.is_file() and p.suffix == ".json"]
        except OSError:
            continue
        for path in entries:
            # At the project root, restrict to name-hinted files to avoid
            # parsing every stray JSON; dedicated bench dirs scan all JSON.
            if root_level and not any(h in path.name.lower() for h in _NAME_HINTS):
                continue
            _add(path)

    for dirname in _CKPT_DIRS:
        ckpt_root = root_path / dirname
        if not ckpt_root.is_dir():
            continue
        try:
            model_dirs = [d for d in ckpt_root.iterdir() if d.is_dir()]
        except OSError:
            continue
        for model_dir in model_dirs:
            try:
                jsons = [p for p in model_dir.iterdir() if p.is_file() and p.suffix == ".json"]
            except OSError:
                continue
            for path in jsons:
                if any(h in path.name.lower() for h in _NAME_HINTS):
                    _add(path)

    return files


def benchmark_results(root: str = ".") -> dict:
    """Collect persisted throughput/latency benchmark reports, newest first.

    Returns a :class:`~cola_coder.ui.schemas.BenchmarkResults`-shaped dict::

        {"runs": [BenchmarkRun, ...], "count": int}

    sorted by modification time (newest first). Reads only past artifacts — it
    never runs a benchmark. On any failure returns ``{"error": "..."}`` and never
    raises. Finding no artifacts is NOT an error: ``{"runs": [], "count": 0}``.
    """
    try:
        root_path = Path(root)
        if not root_path.is_dir():
            return {"error": f"root not found: {root}"}

        runs: list[dict] = []
        for path in _candidate_files(root_path):
            raw = _read_json(path)
            if not _is_benchmark_report(raw):
                continue
            assert isinstance(raw, dict)  # narrowed by _is_benchmark_report
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            runs.append(_summarize(raw, path, mtime))

        # Newest first; path tiebreak for stable ordering.
        runs.sort(key=lambda r: (-r["mtime"], r["path"]))

        return {"runs": runs, "count": len(runs)}
    except Exception as exc:  # noqa: BLE001 — contract: never raise
        logger.warning("benchmark_results scan failed: %s", exc)
        return {"error": str(exc)}
