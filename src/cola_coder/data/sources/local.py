"""Local file data source plugin.

Streams code files from local directories on disk.
Useful for training on your own codebase or curated datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from cola_coder.data.pipeline import DataRecord, DataSource
from cola_coder.data.registry import register_source


@register_source("local")
class LocalFileSource(DataSource):
    """Stream code files from local directories.

    Recursively walks the given paths, yielding files that match
    the specified extensions.

    Args:
        paths: List of directory paths to scan.
        extensions: File extensions to include (e.g. [".py", ".ts"]).
                    If empty/None, includes all files.
    """

    def __init__(
        self,
        paths: list[str],
        extensions: list[str] | None = None,
    ):
        self._paths = [Path(p) for p in paths]
        self._extensions = set(extensions) if extensions else None
        self._file_count: int | None = None

    def name(self) -> str:
        dirs = ", ".join(str(p) for p in self._paths)
        return f"local([{dirs}])"

    def _iter_files(self) -> Iterator[Path]:
        """Yield all matching file paths across all directories."""
        for base_path in self._paths:
            if not base_path.exists():
                print(f"  Warning: path does not exist: {base_path}")
                continue

            if base_path.is_file():
                if self._extensions is None or base_path.suffix in self._extensions:
                    yield base_path
                continue

            for file_path in sorted(base_path.rglob("*")):
                if not file_path.is_file():
                    continue
                if self._extensions is not None and file_path.suffix not in self._extensions:
                    continue
                yield file_path

    def stream(self) -> Iterator[DataRecord]:
        """Yield DataRecords from local files."""
        count = 0
        for file_path in self._iter_files():
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, PermissionError) as e:
                print(f"  Warning: could not read {file_path}: {e}")
                continue

            if not content or len(content) < 10:
                continue

            count += 1
            yield DataRecord(
                content=content,
                metadata={
                    "source": "local",
                    # "file_path" is the CANONICAL key the language detectors
                    # (scorers/language_detect.py) and github source use to infer
                    # language from the extension. "path" is kept for backward
                    # compatibility with existing readers/tests.
                    "file_path": str(file_path),
                    "path": str(file_path),
                    "extension": file_path.suffix,
                },
            )

        self._file_count = count

    def estimate_size(self) -> int | None:
        """Count files if we haven't streamed yet, otherwise return cached count."""
        if self._file_count is not None:
            return self._file_count
        # Quick count without reading file contents
        count = sum(1 for _ in self._iter_files())
        return count
