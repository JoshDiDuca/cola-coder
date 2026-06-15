"""Vector-index stats endpoint helper for the local UI.

Read-only mirror of the CLI "Vector Store Stats" action
(``features/master_menu.py`` ``_vector_store_stats`` /
``features/menus/tools_menu.py`` repository-index). Reports on the persisted
retrieval index built by :class:`cola_coder.retrieval.vector_store.VectorStore`,
which saves two sidecar files for a given base path:

* ``{base}.json`` — ``ids``, ``texts``, ``metadata`` and ``embed_dim``
* ``{base}.npz`` — the ``(n, embed_dim)`` embedding matrix

The index lives under ``data/vector_index`` (relative to the project root), the
same location the CLI reads and writes. This helper never raises: a missing
index yields a valid "not built" result (``exists=False``, ``doc_count=0``); any
read failure yields ``{"error": ...}``.

SCOPE: stats only — no search, no indexing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Index location, relative to the project root — matches the CLI menus.
_INDEX_REL_DIR = Path("data") / "vector_index"

# Metadata keys a chunk may carry that name the embedding model used to build
# the index. The VectorStore itself is model-agnostic, so the model name (if
# known) is stored per-chunk in metadata; we read it opportunistically.
_EMBED_MODEL_KEYS = ("embedding_model", "embed_model", "model")


def _project_root() -> Path:
    """Return the repository root (…/src/cola_coder/ui/this_file → up 4)."""
    return Path(__file__).resolve().parents[3]


def _find_store_base(index_dir: Path) -> Path | None:
    """Locate the VectorStore base path whose ``{base}.json`` sidecar exists.

    ``VectorStore.save(base)`` writes ``{base}.json`` / ``{base}.npz``. The CLI
    uses ``data/vector_index`` as that base, so the sidecars are
    ``data/vector_index.json`` / ``.npz``. Some setups instead place the store
    files *inside* a ``data/vector_index/`` directory. Handle both: prefer the
    direct ``{index_dir}.json`` sidecar, else the first ``*.json`` whose ``.npz``
    twin exists within the directory.
    """
    direct = index_dir.with_suffix(".json")
    if direct.is_file():
        return index_dir

    if index_dir.is_dir():
        for json_path in sorted(index_dir.glob("*.json")):
            if json_path.with_suffix(".npz").is_file() or json_path.is_file():
                return json_path.with_suffix("")

    return None


def _read_embedding_model(metadata: list[object]) -> str | None:
    """Best-effort extraction of the embedding-model name from chunk metadata."""
    for entry in metadata:
        if not isinstance(entry, dict):
            continue
        for key in _EMBED_MODEL_KEYS:
            value = entry.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _file_size(path: Path) -> int:
    """Size of ``path`` in bytes, or 0 if it is absent/unreadable."""
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def _not_built(path: str | None) -> dict[str, object]:
    """A valid, non-error 'index has not been built yet' stats payload."""
    return {
        "exists": False,
        "doc_count": 0,
        "chunk_count": None,
        "embedding_model": None,
        "embedding_dim": None,
        "size_bytes": 0,
        "path": path,
        "last_updated": None,
    }


def index_stats() -> dict[str, object]:
    """Return stats for the persisted vector index (matches ``IndexStats``).

    Reads the ``VectorStore`` sidecars under ``data/vector_index`` without
    loading the embedding matrix into memory. When no index exists, returns a
    well-formed "not built" result rather than an error.
    """
    index_dir = _project_root() / _INDEX_REL_DIR

    base = _find_store_base(index_dir)
    if base is None:
        return _not_built(str(index_dir))

    json_path = base.with_suffix(".json")
    npz_path = base.with_suffix(".npz")

    try:
        with json_path.open(encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read vector index sidecar %s: %s", json_path, exc)
        return {"error": f"failed to read vector index: {exc}"}

    if not isinstance(meta, dict):
        return {"error": f"malformed vector index sidecar: {json_path}"}

    ids = meta.get("ids")
    doc_count = len(ids) if isinstance(ids, list) else 0

    metadata = meta.get("metadata")
    embedding_model = _read_embedding_model(metadata) if isinstance(metadata, list) else None

    embed_dim_raw = meta.get("embed_dim")
    embedding_dim = embed_dim_raw if isinstance(embed_dim_raw, int) else None

    size_bytes = _file_size(json_path) + _file_size(npz_path)

    try:
        mtime = json_path.stat().st_mtime
        last_updated: str | None = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except OSError:
        last_updated = None

    return {
        "exists": True,
        "doc_count": doc_count,
        # Each stored item is one chunk, so chunk_count mirrors doc_count here.
        "chunk_count": doc_count,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "size_bytes": size_bytes,
        "path": str(base),
        "last_updated": last_updated,
    }
