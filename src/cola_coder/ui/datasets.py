"""Dataset browsing helpers for the local UI/dashboard.

Lightweight, read-only inspection of training datasets (.npy / .jsonl) and their
quality-score weight sidecars (.weights.npy). All functions are robust to missing
or malformed inputs and never raise on bad data — they return empty results or an
{"error": ...} dict instead.
"""

from __future__ import annotations

import json
import os

import numpy as np


def list_datasets(data_root: str = "data") -> list[dict]:
    """Recursively scan ``data_root`` for ``*.npy`` and ``*.jsonl`` files.

    Each entry is a dict with keys: name, path, kind ("npy"|"jsonl"),
    size_bytes, mtime, has_weights, num_samples.

    - ``*.weights.npy`` sidecars are excluded from the listing; instead an
      ``X.npy`` reports ``has_weights=True`` when ``X.weights.npy`` exists.
    - ``num_samples`` is computed cheaply (mmap for npy, line count for jsonl)
      and is ``None`` on any error.
    - Missing ``data_root`` yields ``[]``. Results are sorted by path.
    """
    if not os.path.isdir(data_root):
        return []

    results: list[dict] = []
    for dirpath, _dirnames, filenames in os.walk(data_root):
        names = set(filenames)
        for filename in filenames:
            if filename.endswith(".weights.npy"):
                continue
            if filename.endswith(".npy"):
                kind = "npy"
            elif filename.endswith(".jsonl"):
                kind = "jsonl"
            else:
                continue

            path = os.path.join(dirpath, filename)

            try:
                stat = os.stat(path)
                size_bytes = stat.st_size
                mtime = stat.st_mtime
            except OSError:
                continue

            has_weights = False
            if kind == "npy":
                weights_name = filename[: -len(".npy")] + ".weights.npy"
                has_weights = weights_name in names

            results.append(
                {
                    "name": filename,
                    "path": path,
                    "kind": kind,
                    "size_bytes": size_bytes,
                    "mtime": mtime,
                    "has_weights": has_weights,
                    "num_samples": _count_samples(path, kind),
                }
            )

    results.sort(key=lambda entry: entry["path"])
    return results


def _count_samples(path: str, kind: str) -> int | None:
    """Cheap sample count: mmap shape[0] for npy, non-empty lines for jsonl."""
    try:
        if kind == "npy":
            arr = np.load(path, mmap_mode="r")
            return int(arr.shape[0])
        count = 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count
    except Exception:
        return None


def dataset_preview(path: str, n: int = 20) -> dict:
    """Preview the first ``n`` rows/records of a dataset.

    jsonl -> {"kind":"jsonl","num_samples":int,"preview":[parsed JSON objs]}
    npy   -> {"kind":"npy","shape":list,"dtype":str,"num_samples":int,
              "preview":[rows as plain python lists]}

    Unparseable jsonl lines are skipped (never raised). Missing path returns
    {"error": str}.
    """
    if not os.path.isfile(path):
        return {"error": f"path not found: {path}"}

    if path.endswith(".jsonl"):
        preview: list = []
        num_samples = 0
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    num_samples += 1
                    if len(preview) < n:
                        try:
                            preview.append(json.loads(line))
                        except (ValueError, TypeError):
                            continue
        except OSError as exc:
            return {"error": str(exc)}
        return {"kind": "jsonl", "num_samples": num_samples, "preview": preview}

    try:
        arr = np.load(path, mmap_mode="r")
        rows = arr[:n]
        return {
            "kind": "npy",
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "num_samples": int(arr.shape[0]),
            "preview": rows.tolist(),
        }
    except Exception as exc:
        return {"error": str(exc)}


def score_summary(weights_path: str) -> dict:
    """Summarize a 1-D float ``.weights.npy`` quality-score array.

    Returns {"n", "mean", "min", "max", "histogram" (10 ints), "bins" (11 edges)}.
    Missing or empty array returns {"error": str}.
    """
    if not os.path.isfile(weights_path):
        return {"error": f"path not found: {weights_path}"}

    try:
        arr = np.load(weights_path, mmap_mode="r")
        arr = np.asarray(arr).reshape(-1)
    except Exception as exc:
        return {"error": str(exc)}

    if arr.size == 0:
        return {"error": f"empty array: {weights_path}"}

    histogram, bins = np.histogram(arr, bins=10)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "histogram": [int(count) for count in histogram],
        "bins": [float(edge) for edge in bins],
    }
