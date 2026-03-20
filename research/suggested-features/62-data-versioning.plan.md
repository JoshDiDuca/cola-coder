# 62 - Data Versioning (DVC-Lite)

## Overview

A Git-like versioning system for Cola-Coder's training datasets. Every time data is prepared—with different source files, filter settings, dedup parameters, or split ratios—the result is stored as a named, addressable version. Versions can be compared, tagged, and rolled back without re-running the full pipeline.

**Feature flag:** `--enable-data-versioning` / `config.data_versioning.enabled`

---

## Motivation

Without data versioning, reproducing a training run requires manually remembering (or re-discovering) every filter setting, source directory, and dedup threshold used to produce the dataset. Common failure modes:

- "Which dataset did checkpoint-3000 train on?" — unknown
- Filter thresholds changed mid-project; old checkpoints trained on different data
- A bug in the dedup step was fixed; now it's unclear which checkpoints are affected
- Team member re-runs `prepare.py` with different paths and overwrites the dataset

Data versioning makes datasets first-class artifacts with the same reproducibility guarantees as code commits. Every training run in the provenance manifest references a dataset version hash, making the full experiment reproducible.

**Goal:** Zero-overhead versioning that requires no external services (no DVC server, no S3, no database beyond SQLite).

---

## Architecture / Design

### Content-Addressable Storage

Each dataset version is identified by a SHA-256 hash of its content metadata:

```
version_hash = sha256(
    sorted(source_file_hashes) +
    filter_config_json +
    dedup_config_json +
    split_ratios_json
)
```

Actual data files (`.npy` token arrays) live at:
```
data/versions/{version_hash[:8]}/
    train.npy
    val.npy
    metadata.yaml       # full provenance
    stats.json          # token counts, rejection rates, etc.
```

A lightweight index lives at `data/versions/index.yaml`:
```yaml
versions:
  - hash: a3f7c291
    tag: baseline-v1
    created: 2026-03-10T14:22:00
    description: "Initial dataset, 50k files, min_tokens=50"
  - hash: b8e2a104
    tag: filtered-strict
    created: 2026-03-15T09:45:00
    description: "Stricter quality filter, min_tokens=100"
current: b8e2a104
```

### Metadata Schema (`metadata.yaml`)

```yaml
version:
  hash: b8e2a104ff3a...
  created_at: 2026-03-15T09:45:12Z
  description: "Stricter quality filter"
  tag: filtered-strict

sources:
  - path: /data/repos/typescript-corpus
    file_count: 52341
    total_size_bytes: 847293847
    scan_hash: c4f8a1...   # hash of all file paths+sizes

filters:
  min_tokens: 100
  max_tokens: 1024
  min_lines: 10
  exclude_patterns:
    - "*.d.ts"
    - "node_modules/**"
  quality_threshold: 0.7

dedup:
  method: minhash
  threshold: 0.85
  ngram_size: 5
  num_hashes: 128
  duplicates_removed: 3241

splits:
  train_ratio: 0.95
  val_ratio: 0.05
  seed: 42

stats:
  total_files_scanned: 52341
  files_accepted: 31088
  files_rejected: 21253
  rejection_rate: 0.406
  total_tokens: 28_432_991
  train_tokens: 27_011_341
  val_tokens: 1_421_650
  avg_sequence_length: 914.7
  rejection_reasons:
    too_short: 12401
    too_long: 3892
    duplicate: 3241
    low_quality: 1719
```

---

## Implementation Steps

### Step 1: Version Manager (`data/version_manager.py`)

```python
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import yaml

class DataVersionManager:
    def __init__(self, versions_dir: Path):
        self.versions_dir = versions_dir
        self.index_path = versions_dir / "index.yaml"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._write_index({"versions": [], "current": None})

    def _compute_version_hash(
        self,
        source_hashes: list[str],
        filter_config: dict,
        dedup_config: dict,
        split_config: dict,
    ) -> str:
        payload = json.dumps({
            "sources": sorted(source_hashes),
            "filters": filter_config,
            "dedup": dedup_config,
            "splits": split_config,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _hash_source_dir(self, path: Path) -> str:
        """Hash all file paths and sizes in a directory."""
        entries = sorted(
            (str(f.relative_to(path)), f.stat().st_size)
            for f in path.rglob("*.ts")
        )
        return hashlib.sha256(str(entries).encode()).hexdigest()[:12]

    def create_version(
        self,
        train_tokens: np.ndarray,
        val_tokens: np.ndarray,
        metadata: dict,
        tag: str = None,
        description: str = "",
    ) -> str:
        version_hash = metadata["version"]["hash"]
        version_dir = self.versions_dir / version_hash

        if version_dir.exists():
            print(f"[data-versioning] Version {version_hash} already exists, skipping.")
            return version_hash

        version_dir.mkdir(parents=True)
        np.save(version_dir / "train.npy", train_tokens)
        np.save(version_dir / "val.npy", val_tokens)

        if tag:
            metadata["version"]["tag"] = tag
        metadata["version"]["description"] = description

        with open(version_dir / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

        with open(version_dir / "stats.json", "w") as f:
            json.dump(metadata.get("stats", {}), f, indent=2)

        # Update index
        index = self._read_index()
        index["versions"].append({
            "hash": version_hash,
            "tag": tag or version_hash[:8],
            "created": datetime.now(timezone.utc).isoformat(),
            "description": description,
        })
        index["current"] = version_hash
        self._write_index(index)

        return version_hash

    def get_version(self, ref: str) -> Path:
        """Resolve a ref (hash, tag, or 'current') to a version directory."""
        index = self._read_index()

        if ref == "current":
            ref = index.get("current")
            if not ref:
                raise ValueError("No current version set.")

        # Try exact hash match
        version_dir = self.versions_dir / ref
        if version_dir.exists():
            return version_dir

        # Try tag match
        for entry in index["versions"]:
            if entry.get("tag") == ref:
                return self.versions_dir / entry["hash"]

        raise ValueError(f"Version '{ref}' not found.")

    def list_versions(self) -> list[dict]:
        return self._read_index().get("versions", [])

    def tag_version(self, version_hash: str, tag: str):
        index = self._read_index()
        for entry in index["versions"]:
            if entry["hash"] == version_hash or entry.get("tag") == version_hash:
                entry["tag"] = tag
                self._write_index(index)
                return
        raise ValueError(f"Version '{version_hash}' not found.")

    def set_current(self, ref: str):
        version_dir = self.get_version(ref)  # validates existence
        index = self._read_index()
        # Resolve to actual hash
        for entry in index["versions"]:
            if entry.get("tag") == ref or entry["hash"].startswith(ref):
                index["current"] = entry["hash"]
                break
        self._write_index(index)

    def compare_versions(self, ref_a: str, ref_b: str) -> dict:
        """Return a diff of stats between two versions."""
        dir_a = self.get_version(ref_a)
        dir_b = self.get_version(ref_b)

        with open(dir_a / "stats.json") as f:
            stats_a = json.load(f)
        with open(dir_b / "stats.json") as f:
            stats_b = json.load(f)

        diff = {}
        all_keys = set(stats_a) | set(stats_b)
        for key in all_keys:
            val_a = stats_a.get(key, None)
            val_b = stats_b.get(key, None)
            if val_a != val_b:
                diff[key] = {"a": val_a, "b": val_b}
                if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)) and val_a:
                    diff[key]["delta_pct"] = round((val_b - val_a) / val_a * 100, 1)
        return diff

    def _read_index(self) -> dict:
        with open(self.index_path) as f:
            return yaml.safe_load(f) or {"versions": [], "current": None}

    def _write_index(self, index: dict):
        with open(self.index_path, "w") as f:
            yaml.dump(index, f, default_flow_style=False)
```

### Step 2: CLI Commands (`cli/data_version_cmd.py`)

```python
# cola-coder data-version list
# cola-coder data-version tag <hash> <tag>
# cola-coder data-version use <ref>
# cola-coder data-version diff <ref-a> <ref-b>
# cola-coder data-version info <ref>

from rich.table import Table
from rich.console import Console
from rich import print as rprint

def cmd_list(manager: DataVersionManager):
    console = Console()
    table = Table(title="Dataset Versions")
    table.add_column("Hash", style="cyan")
    table.add_column("Tag", style="green")
    table.add_column("Created", style="dim")
    table.add_column("Description")

    index = manager._read_index()
    current = index.get("current", "")

    for entry in manager.list_versions():
        marker = " [bold yellow]*[/]" if entry["hash"] == current else ""
        table.add_row(
            entry["hash"][:8] + marker,
            entry.get("tag", ""),
            entry.get("created", "")[:10],
            entry.get("description", ""),
        )
    console.print(table)

def cmd_diff(manager: DataVersionManager, ref_a: str, ref_b: str):
    diff = manager.compare_versions(ref_a, ref_b)
    console = Console()
    table = Table(title=f"Diff: {ref_a} → {ref_b}")
    table.add_column("Metric")
    table.add_column(ref_a, justify="right")
    table.add_column(ref_b, justify="right")
    table.add_column("Delta %", justify="right")

    for key, vals in diff.items():
        delta = f"{vals.get('delta_pct', 'N/A'):+.1f}%" if "delta_pct" in vals else "—"
        style = "green" if vals.get("delta_pct", 0) > 0 else "red"
        table.add_row(key, str(vals["a"]), str(vals["b"]), f"[{style}]{delta}[/]")
    console.print(table)
```

### Step 3: Integration with `prepare.py`

```python
# At the end of data preparation, after saving train.npy and val.npy:

if config.data_versioning.enabled:
    manager = DataVersionManager(Path("data/versions"))
    version_hash = manager.create_version(
        train_tokens=train_array,
        val_tokens=val_array,
        metadata=build_metadata(config, stats),
        tag=args.version_tag,
        description=args.version_description,
    )
    print(f"[data-versioning] Saved version: {version_hash}")

    # Write version hash into training manifest
    config.training.dataset_version = version_hash
```

---

## Key Files to Modify

- `data/prepare.py` - Call `DataVersionManager.create_version()` after data prep
- `data/version_manager.py` - New file: all versioning logic
- `cli/data_version_cmd.py` - New file: CLI subcommands
- `cli/main.py` - Register `data-version` subcommand group
- `config/training.yaml` - Add `data_versioning` section
- `training/trainer.py` - Load dataset by version ref, not raw path
- `manifests/` - Include `dataset_version` hash in training manifests

---

## Testing Strategy

1. **Round-trip test**: prepare a dataset, save version, reload it, assert `train.npy` contents match byte-for-byte.
2. **Hash stability test**: run `create_version` twice with identical inputs, assert same hash is returned and no duplicate entry added to index.
3. **Tag test**: create version, apply tag, resolve by tag name, assert same directory returned.
4. **Diff test**: create two versions with differing filter configs (different `min_tokens`), assert diff shows non-empty stats delta.
5. **Rollback test**: set current to version A, then version B, then back to A; load dataset from "current" and assert it's version A's data.

---

## Performance Considerations

- `.npy` files are stored uncompressed for fast `np.load()` with `mmap_mode='r'`. Typical 28M token dataset ≈ 56MB (int32). Storage is cheap.
- Version hashing is O(1) given pre-computed source hashes. Source hashing (`_hash_source_dir`) is O(N files) but runs once and can be cached.
- The version directory doubles storage for each version. Implement a `prune` command to delete versions older than N days or not referenced by any manifest.
- Symlink `data/versions/current → data/versions/{hash}` for zero-cost "rollback" at the filesystem level.

---

## Dependencies

No new Python dependencies. Uses only: `hashlib`, `json`, `shutil`, `pathlib`, `numpy`, `yaml` (already required).

---

## Estimated Complexity

**Low-Medium.** The storage and metadata logic is straightforward. The main complexity is integrating with `prepare.py` (non-trivial because it has multiple data sources) and ensuring the trainer always loads from a versioned path rather than a hardcoded `data/train.npy`. Estimated implementation time: 2-3 days.

---

## 2026 Best Practices

- **Content addressing over path addressing**: Identify datasets by what they contain, not where they live. This makes datasets portable across machines and OS.
- **Immutable versions**: Once written, a version directory is never modified. Mutations create new versions. Enforce this with a write-lock or at minimum a runtime check.
- **Version in manifest**: Every training manifest should include the dataset version hash. Without this link, experiment reproducibility is impossible.
- **Prune command**: Storage compounds quickly. Provide `data-version prune --keep-tagged --keep-last 5` to manage disk usage.
- **Human-readable tags over hashes**: Encourage tagging with semantic names (`baseline-v1`, `filtered-strict`) in CI/CD workflows. Hashes are for machines; tags are for humans.
