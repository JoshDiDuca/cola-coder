"""Data Versioning: track training data versions with checksums.

Provides content-addressed versioning for training data files, so you
always know exactly which data was used for a given training run.

For a TS dev: like package-lock.json but for training data — a manifest
that records exactly what data files and versions were used.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class DataVersion:
    """A versioned snapshot of a data file."""
    file_path: str
    sha256: str
    size_bytes: int
    created_at: float  # Unix timestamp
    metadata: dict = field(default_factory=dict)
    version_tag: str = ""

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    @property
    def short_hash(self) -> str:
        return self.sha256[:12]

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "version_tag": self.version_tag,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DataVersion":
        return cls(**d)


@dataclass
class DataManifest:
    """A manifest tracking all versioned data files."""
    versions: list[DataVersion] = field(default_factory=list)
    manifest_path: str = ""

    def add(self, version: DataVersion) -> None:
        """Add a version to the manifest."""
        self.versions.append(version)

    def get_latest(self, file_path: str) -> DataVersion | None:
        """Get the latest version of a specific file."""
        matching = [v for v in self.versions if v.file_path == file_path]
        if not matching:
            return None
        return max(matching, key=lambda v: v.created_at)

    def get_by_hash(self, sha256: str) -> DataVersion | None:
        """Find a version by its SHA-256 hash."""
        for v in self.versions:
            if v.sha256 == sha256 or v.sha256.startswith(sha256):
                return v
        return None

    def list_files(self) -> list[str]:
        """List all unique file paths in the manifest."""
        return sorted(set(v.file_path for v in self.versions))

    def has_changed(self, file_path: str) -> bool:
        """Check if a file has changed since it was last versioned."""
        latest = self.get_latest(file_path)
        if not latest:
            return True
        current_hash = compute_sha256(file_path)
        return current_hash != latest.sha256

    def save(self, path: str | None = None) -> None:
        """Save manifest to JSON file."""
        save_path = path or self.manifest_path
        if not save_path:
            raise ValueError("No manifest path specified")

        data = {
            "versions": [v.to_dict() for v in self.versions],
            "saved_at": time.time(),
        }
        Path(save_path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "DataManifest":
        """Load manifest from JSON file."""
        p = Path(path)
        if not p.exists():
            return cls(manifest_path=path)

        data = json.loads(p.read_text())
        manifest = cls(manifest_path=path)
        for v_data in data.get("versions", []):
            manifest.versions.append(DataVersion.from_dict(v_data))
        return manifest

    def summary(self) -> str:
        files = self.list_files()
        total_size = sum(v.size_bytes for v in self.versions)
        return (
            f"Data Manifest: {len(self.versions)} versions across {len(files)} files\n"
            f"Total size: {total_size / (1024*1024):.1f} MB"
        )


def compute_sha256(file_path: str, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of a file.

    Uses chunked reading for memory efficiency with large files.
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def version_file(
    file_path: str,
    version_tag: str = "",
    metadata: dict | None = None,
) -> DataVersion:
    """Create a versioned snapshot of a data file.

    Args:
        file_path: Path to the data file
        version_tag: Optional version tag (e.g., "v1.0")
        metadata: Optional metadata to attach

    Returns:
        DataVersion with computed hash and file info
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    sha256 = compute_sha256(file_path)

    return DataVersion(
        file_path=str(p.resolve()),
        sha256=sha256,
        size_bytes=p.stat().st_size,
        created_at=time.time(),
        metadata=metadata or {},
        version_tag=version_tag,
    )


def compare_versions(v1: DataVersion, v2: DataVersion) -> dict:
    """Compare two data versions.

    Returns:
        Dict with comparison results
    """
    return {
        "same_file": v1.file_path == v2.file_path,
        "same_content": v1.sha256 == v2.sha256,
        "size_diff_bytes": v2.size_bytes - v1.size_bytes,
        "size_diff_mb": (v2.size_bytes - v1.size_bytes) / (1024 * 1024),
        "time_diff_seconds": v2.created_at - v1.created_at,
    }


def print_manifest(manifest: DataManifest) -> None:
    """Print a formatted manifest report."""
    from cola_coder.cli import cli

    cli.header("Data Manifest", f"{len(manifest.versions)} versions")
    for f in manifest.list_files():
        latest = manifest.get_latest(f)
        if latest:
            tag = f" [{latest.version_tag}]" if latest.version_tag else ""
            cli.info(
                Path(f).name + tag,
                f"{latest.size_mb:.1f} MB | {latest.short_hash}"
            )
