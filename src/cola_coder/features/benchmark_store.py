"""Benchmark Results Store: persist and compare evaluation results.

Saves benchmark results as JSON files under data/benchmarks/.
Each result is a JSON-serializable dict (or dataclass-converted dict).

Typical use::

    store = BenchmarkStore()
    store.save({"pass_rate": 0.73, "model": "tiny-v1"}, name="humaneval_v1")
    store.save({"pass_rate": 0.81, "model": "tiny-v2"}, name="humaneval_v2")
    print(store.compare("humaneval_v1", "humaneval_v2"))
    print(store.list())

For a TS dev: like a tiny key-value store for JSON objects, backed by the
filesystem (think: a very simple version of tinydb).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkEntry:
    """Metadata for a stored benchmark result."""

    name: str
    path: str
    saved_at: float  # Unix timestamp
    keys: list[str]  # Top-level keys in the result dict

    @property
    def saved_at_iso(self) -> str:
        import datetime

        return datetime.datetime.fromtimestamp(self.saved_at).isoformat(timespec="seconds")


@dataclass
class CompareResult:
    """Side-by-side comparison of two benchmark results."""

    name_a: str
    name_b: str
    keys_a_only: list[str]
    keys_b_only: list[str]
    shared_keys: list[str]
    differences: dict[str, dict[str, Any]]  # key -> {"a": ..., "b": ..., "delta": ...}

    def summary(self) -> str:
        lines = [f"Comparing '{self.name_a}' vs '{self.name_b}'"]
        for key, diff in self.differences.items():
            a_val = diff.get("a", "n/a")
            b_val = diff.get("b", "n/a")
            delta = diff.get("delta")
            if delta is not None:
                arrow = "↑" if delta > 0 else "↓"
                lines.append(f"  {key}: {a_val} → {b_val} ({arrow}{abs(delta):.4g})")
            else:
                lines.append(f"  {key}: {a_val!r} → {b_val!r}")
        if not self.differences:
            lines.append("  (no differences in shared numeric keys)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class BenchmarkStore:
    """Save and load benchmark results as JSON files.

    Parameters
    ----------
    base_dir:
        Directory to store results in.  Defaults to ``data/benchmarks/``
        relative to the current working directory.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        if base_dir is None:
            base_dir = Path.cwd() / "data" / "benchmarks"
        self.base_dir = Path(base_dir)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def save(self, result: dict | Any, name: str) -> Path:
        """Save *result* under *name*.  Overwrites if already exists.

        Parameters
        ----------
        result:
            JSON-serializable dict (or any object with ``__dict__`` / ``__dataclass_fields__``).
        name:
            Logical name for this result (used as filename without extension).

        Returns
        -------
        Path to the saved file.
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        data = self._to_dict(result)
        data["_meta"] = {"name": name, "saved_at": time.time()}
        path = self.base_dir / f"{name}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        return path

    def load(self, name: str) -> dict[str, Any]:
        """Load a result by name.  Raises FileNotFoundError if missing."""
        path = self.base_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No benchmark result named '{name}' found at {path}"
            )
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def list(self) -> list[BenchmarkEntry]:
        """Return all stored benchmark entries, sorted by save time (newest first)."""
        if not self.base_dir.exists():
            return []
        entries: list[BenchmarkEntry] = []
        for json_path in self.base_dir.glob("*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                meta = data.get("_meta", {})
                entries.append(
                    BenchmarkEntry(
                        name=json_path.stem,
                        path=str(json_path),
                        saved_at=meta.get("saved_at", 0.0),
                        keys=[k for k in data if k != "_meta"],
                    )
                )
            except (json.JSONDecodeError, OSError):
                continue  # skip malformed files
        return sorted(entries, key=lambda e: e.saved_at, reverse=True)

    def delete(self, name: str) -> bool:
        """Delete a result by name.  Returns True if found and deleted."""
        path = self.base_dir / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def compare(self, a: str, b: str) -> CompareResult:
        """Compare two benchmark results side-by-side.

        Numeric values are compared with a delta; other values are shown as-is.
        """
        data_a = self.load(a)
        data_b = self.load(b)

        # Strip metadata
        data_a = {k: v for k, v in data_a.items() if k != "_meta"}
        data_b = {k: v for k, v in data_b.items() if k != "_meta"}

        keys_a = set(data_a.keys())
        keys_b = set(data_b.keys())
        shared = keys_a & keys_b

        differences: dict[str, dict[str, Any]] = {}
        for key in sorted(shared):
            va, vb = data_a[key], data_b[key]
            if va != vb:
                entry: dict[str, Any] = {"a": va, "b": vb}
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    entry["delta"] = vb - va
                differences[key] = entry

        return CompareResult(
            name_a=a,
            name_b=b,
            keys_a_only=sorted(keys_a - keys_b),
            keys_b_only=sorted(keys_b - keys_a),
            shared_keys=sorted(shared),
            differences=differences,
        )

    def exists(self, name: str) -> bool:
        """Return True if a result named *name* exists."""
        return (self.base_dir / f"{name}.json").exists()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dict(obj: Any) -> dict:
        if isinstance(obj, dict):
            return dict(obj)
        # dataclass
        try:
            import dataclasses

            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
        except ImportError:
            pass
        # fallback
        return vars(obj) if hasattr(obj, "__dict__") else {"value": obj}
