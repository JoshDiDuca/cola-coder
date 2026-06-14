"""Tests for the read-only router subsystem view (``ui/router.py``).

All tests are hermetic — they build fake checkpoint trees under ``tmp_path``
and never touch the real ``checkpoints/`` directory.
"""

from __future__ import annotations

import json

from cola_coder.ui.router import router_overview


def _make_router_step(tmp_path, step: int, with_meta: bool = False):
    """Create checkpoints/router/step_<n> under tmp_path; return the dir."""
    name = f"step_{step:08d}"
    step_dir = tmp_path / "checkpoints" / "router" / name
    step_dir.mkdir(parents=True)
    if with_meta:
        (step_dir / "metadata.json").write_text(
            json.dumps({"step": step}), encoding="utf-8"
        )
    return step_dir


def test_no_router_dir_returns_false_and_empty():
    result = router_overview(root="/this/path/does/not/exist/at/all")
    assert result["has_router"] is False
    assert result["checkpoints"] == []
    assert isinstance(result["domains"], list)
    assert result["domains"]  # non-empty


def test_missing_router_dir_under_existing_root(tmp_path):
    # root exists, but no checkpoints/router subdir.
    (tmp_path / "checkpoints").mkdir()
    result = router_overview(root=str(tmp_path))
    assert result["has_router"] is False
    assert result["checkpoints"] == []


def test_single_checkpoint_parsed(tmp_path):
    _make_router_step(tmp_path, 500)
    result = router_overview(root=str(tmp_path))
    assert result["has_router"] is True
    assert len(result["checkpoints"]) == 1
    ckpt = result["checkpoints"][0]
    assert ckpt["step"] == 500
    assert ckpt["name"] == "step_00000500"
    assert ckpt["path"].endswith("step_00000500")


def test_single_checkpoint_with_metadata(tmp_path):
    _make_router_step(tmp_path, 500, with_meta=True)
    result = router_overview(root=str(tmp_path))
    assert result["has_router"] is True
    assert result["checkpoints"][0]["step"] == 500


def test_multiple_checkpoints_newest_first(tmp_path):
    _make_router_step(tmp_path, 100)
    _make_router_step(tmp_path, 2000)
    _make_router_step(tmp_path, 500)
    result = router_overview(root=str(tmp_path))
    steps = [c["step"] for c in result["checkpoints"]]
    assert steps == [2000, 500, 100]  # strictly newest-first


def test_empty_router_dir_has_router_true_no_checkpoints(tmp_path):
    # Router dir exists (e.g. holds best_router.pt at top level) but no step_* dirs.
    router_dir = tmp_path / "checkpoints" / "router"
    router_dir.mkdir(parents=True)
    (router_dir / "best_router.pt").write_bytes(b"\x00")
    result = router_overview(root=str(tmp_path))
    assert result["has_router"] is True
    assert result["checkpoints"] == []


def test_non_step_dirs_ignored(tmp_path):
    router_dir = tmp_path / "checkpoints" / "router"
    router_dir.mkdir(parents=True)
    (router_dir / "some-run").mkdir()  # not a step_* dir
    _make_router_step(tmp_path, 42)
    result = router_overview(root=str(tmp_path))
    assert result["has_router"] is True
    assert [c["step"] for c in result["checkpoints"]] == [42]


def test_malformed_step_dir_skipped(tmp_path):
    router_dir = tmp_path / "checkpoints" / "router"
    router_dir.mkdir(parents=True)
    (router_dir / "step_notanumber").mkdir()
    _make_router_step(tmp_path, 7)
    result = router_overview(root=str(tmp_path))
    assert [c["step"] for c in result["checkpoints"]] == [7]


def test_domains_match_source_of_truth(tmp_path):
    from cola_coder.features.router_model import DEFAULT_DOMAINS

    result = router_overview(root=str(tmp_path))
    assert result["domains"] == list(DEFAULT_DOMAINS)
    # Sanity: the vision's specialists are present.
    for expected in ("react", "nextjs", "graphql", "prisma", "zod", "testing"):
        assert expected in result["domains"]


def test_domains_non_empty_always():
    result = router_overview()
    assert isinstance(result["domains"], list)
    assert len(result["domains"]) >= 1


def test_error_dict_on_broken_input():
    # A non-string root (int) triggers a genuine failure inside Path() handling
    # -> caught and returned as {"error": ...}, never raised.
    result = router_overview(root=12345)  # type: ignore[arg-type]
    assert "error" in result
    assert isinstance(result["error"], str)


def test_result_is_json_serializable(tmp_path):
    _make_router_step(tmp_path, 1000)
    result = router_overview(root=str(tmp_path))
    # Must round-trip through JSON (no Path objects etc.).
    encoded = json.dumps(result)
    assert json.loads(encoded) == result
