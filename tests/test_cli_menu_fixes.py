"""Regression tests for two CLI menu bugs that called non-existent/wrong APIs.

BUG-130: tools_menu project-memory actions referenced a non-existent MemoryManager
         API (initialize/list_recent/MemoryUpdater/project.db). They must use the
         REAL MemoryManager (.cola/memory/ markdown files): is_initialized,
         init_project, stats, export, compact, plus module-level _iter_sections.

BUG-131: master_menu._vector_store_stats constructed VectorStore(str(path)) and read
         stats keys that don't exist. It must LOAD the persisted store from its base
         path and surface the real stats() keys (total_items/embed_dim/memory_mb/
         unique_sources).

All tests use temp dirs, tiny fake vectors, and patched CLI prompts — no network/GPU.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from cola_coder.features.master_menu import MasterMenu
from cola_coder.memory.manager import MemoryManager


# ── BUG-130: project memory uses the REAL MemoryManager API ──────────────────


def _seed_memory(project_root: Path) -> MemoryManager:
    """Initialize a real memory store under <root>/.cola/memory/ with one entry."""
    manager = MemoryManager(project_root)
    manager.init_project(description="A test project", tech_stack={"language": "TypeScript"})
    manager.add_pattern("Use named exports", "export const foo = 1;")
    return manager


def test_tools_menu_memory_helper_uses_real_manager(tmp_path):
    """ToolsMenu._memory_manager builds a real MemoryManager over .cola/memory/."""
    menu = MasterMenu(project_root=tmp_path)
    manager = menu._tools._memory_manager()
    assert isinstance(manager, MemoryManager)
    assert manager.memory_path == tmp_path / ".cola" / "memory"


def test_memory_stats_reads_real_api_without_attribute_error(tmp_path):
    """_memory_stats must call the real .stats() (per-file dict) — the old code
    called .stats().get('total'/'pinned'/...) keys that never existed, and the
    old import path (data/memory) was also wrong."""
    _seed_memory(tmp_path)
    menu = MasterMenu(project_root=tmp_path)

    with patch.object(menu, "_pause", lambda: None):
        # Must not raise AttributeError (no .list_recent / .initialize / wrong keys).
        menu._tools._memory_stats()

    # Prove the real API is what's being consumed.
    stats = menu._tools._memory_manager().stats()
    assert "project" in stats
    assert "chunks" in stats["project"]


def test_memory_view_uses_iter_sections(tmp_path):
    """_memory_view must read entries via export() + _iter_sections, not list_recent()."""
    _seed_memory(tmp_path)
    menu = MasterMenu(project_root=tmp_path)
    with patch.object(menu, "_pause", lambda: None):
        menu._tools._memory_view()  # no AttributeError


def test_memory_view_uninitialized_is_friendly(tmp_path):
    """No store yet → friendly message, no crash (is_initialized is False)."""
    menu = MasterMenu(project_root=tmp_path)
    assert menu._tools._memory_manager().is_initialized is False
    with patch.object(menu, "_pause", lambda: None):
        menu._tools._memory_view()  # must not raise


def test_memory_init_creates_real_store(tmp_path):
    """_memory_init initializes the real .cola/memory/ store via init_project()."""
    menu = MasterMenu(project_root=tmp_path)
    with patch.object(menu, "_pause", lambda: None):
        menu._tools._memory_init()
    memory_dir = tmp_path / ".cola" / "memory"
    assert (memory_dir / "project.md").exists()
    assert (memory_dir / "patterns.md").exists()


def test_memory_compact_uses_real_compact(tmp_path):
    """_memory_compact calls the real compact() (returns {file_key: removed})."""
    _seed_memory(tmp_path)
    menu = MasterMenu(project_root=tmp_path)
    with patch.object(menu, "_pause", lambda: None), patch(
        "cola_coder.cli.cli.confirm", return_value=True
    ):
        menu._tools._memory_compact()  # no AttributeError (no older_than_days kw)


# ── BUG-131: vector store stats LOADS the store and reads real keys ──────────


def _build_and_save_vector_store(base: Path) -> None:
    """Build a tiny VectorStore with fake vectors and persist it to `base`."""
    from cola_coder.retrieval.vector_store import VectorStore

    store = VectorStore(embed_dim=4)
    store.add_batch(
        ids=["a.ts:1-2", "b.ts:1-2"],
        texts=["const a = 1;", "const b = 2;"],
        embeddings=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        metadata=[{"source": "a.ts"}, {"source": "b.ts"}],
    )
    store.save(str(base))


def test_vector_store_stats_no_index_is_friendly(tmp_path):
    """No persisted index → friendly 'no index built yet', no crash."""
    menu = MasterMenu(project_root=tmp_path)
    with patch.object(menu, "_pause", lambda: None):
        menu._vector_store_stats()  # must not raise


def test_vector_store_stats_loads_and_surfaces_real_keys(tmp_path):
    """_vector_store_stats must load the persisted store and read the REAL stats()
    keys (total_items/embed_dim/memory_mb/unique_sources) without AttributeError/
    KeyError. The old code did VectorStore(str(path)) and read document_count etc."""
    base = tmp_path / "data" / "vector_index"
    _build_and_save_vector_store(base)
    assert base.with_suffix(".json").exists()  # persisted as {base}.json + {base}.npz

    menu = MasterMenu(project_root=tmp_path)

    captured: dict[str, str] = {}

    def _capture(rows, title=""):
        captured.update(rows)

    with patch.object(menu, "_pause", lambda: None), patch(
        "cola_coder.cli.cli.kv_table", side_effect=_capture
    ):
        menu._vector_store_stats()

    # Real keys surfaced (these are the human labels mapped from real stats()).
    assert captured["Total items"] == "2"
    assert captured["Embedding dim"] == "4"
    assert captured["Unique sources"] == "2"


def test_vector_store_real_stats_keys_exist():
    """Guard the contract: the real VectorStore.stats() returns the keys the menu
    relies on (and NOT the phantom document_count/size_mb/last_updated keys)."""
    from cola_coder.retrieval.vector_store import VectorStore

    stats = VectorStore(embed_dim=8).stats()
    assert set(stats) == {"total_items", "embed_dim", "memory_mb", "unique_sources"}
