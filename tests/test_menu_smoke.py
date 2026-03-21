"""Smoke tests for MasterMenu.

Tests that the MasterMenu initializes correctly and that its helper methods
behave correctly in common edge cases. All tests use temporary directories
and no subprocess calls — no GPU or network access needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


from cola_coder.features.master_menu import MasterMenu


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_menu(tmp_path: Path) -> MasterMenu:
    """Create a MasterMenu rooted at tmp_path."""
    return MasterMenu(project_root=tmp_path)


# ── 1. Initialization ─────────────────────────────────────────────────────────

def test_master_menu_initializes(tmp_path):
    """MasterMenu should construct without errors."""
    menu = _make_menu(tmp_path)
    assert menu is not None


def test_master_menu_project_root_set(tmp_path):
    """MasterMenu stores the project_root."""
    menu = _make_menu(tmp_path)
    assert menu.project_root == tmp_path


def test_master_menu_has_storage(tmp_path):
    """MasterMenu has a storage attribute from get_storage_config()."""
    menu = _make_menu(tmp_path)
    assert menu.storage is not None


def test_master_menu_uses_cwd_when_no_root():
    """MasterMenu defaults to cwd when no project_root is provided."""
    menu = MasterMenu()
    assert menu.project_root == Path.cwd()


# ── 2. _pick_checkpoint with no checkpoints ───────────────────────────────────

def test_pick_checkpoint_returns_none_when_no_checkpoints(tmp_path):
    """_pick_checkpoint returns None when checkpoint directory is empty."""
    menu = _make_menu(tmp_path)
    with patch.object(menu, "_list_checkpoints", return_value=[]):
        result = menu._pick_checkpoint()
    assert result is None


def test_pick_checkpoint_returns_none_for_missing_dir(tmp_path):
    """_pick_checkpoint returns None when the checkpoints directory doesn't exist."""
    menu = _make_menu(tmp_path)
    with patch.object(menu, "_list_checkpoints", return_value=[]):
        result = menu._pick_checkpoint()
    assert result is None


# ── 3. _config_for_checkpoint ─────────────────────────────────────────────────

def test_config_for_checkpoint_tiny(tmp_path):
    """_config_for_checkpoint maps 'tiny' path component to configs/tiny.yaml."""
    menu = _make_menu(tmp_path)
    result = menu._config_for_checkpoint("/some/path/checkpoints/tiny/latest")
    assert result == "configs/tiny.yaml"


def test_config_for_checkpoint_small(tmp_path):
    """_config_for_checkpoint maps 'small' to configs/small.yaml."""
    menu = _make_menu(tmp_path)
    result = menu._config_for_checkpoint("/checkpoints/small/step_00001000")
    assert result == "configs/small.yaml"


def test_config_for_checkpoint_medium(tmp_path):
    """_config_for_checkpoint maps 'medium' to configs/medium.yaml."""
    menu = _make_menu(tmp_path)
    result = menu._config_for_checkpoint("checkpoints/medium/latest")
    assert result == "configs/medium.yaml"


def test_config_for_checkpoint_large(tmp_path):
    """_config_for_checkpoint maps 'large' to configs/large.yaml."""
    menu = _make_menu(tmp_path)
    result = menu._config_for_checkpoint("checkpoints/large/step_00010000")
    assert result == "configs/large.yaml"


def test_config_for_checkpoint_unknown_defaults_to_tiny(tmp_path):
    """_config_for_checkpoint returns configs/tiny.yaml for unrecognised size."""
    menu = _make_menu(tmp_path)
    result = menu._config_for_checkpoint("checkpoints/custom-model/step_00000100")
    assert result == "configs/tiny.yaml"


# ── 4. _list_checkpoints ──────────────────────────────────────────────────────

def test_list_checkpoints_empty_dir(tmp_path):
    """_list_checkpoints returns empty list when no checkpoints exist."""
    menu = _make_menu(tmp_path)
    # Override _resolve_path so storage.checkpoints_dir maps to an empty tmp dir
    ckpt_root = tmp_path / "checkpoints"
    ckpt_root.mkdir()
    with patch.object(menu, "_resolve_path", return_value=ckpt_root):
        result = menu._list_checkpoints()
    assert result == []


def test_list_checkpoints_missing_dir(tmp_path):
    """_list_checkpoints returns empty list when the checkpoints directory is absent."""
    menu = _make_menu(tmp_path)
    nonexistent = tmp_path / "no_checkpoints_here"
    with patch.object(menu, "_resolve_path", return_value=nonexistent):
        result = menu._list_checkpoints()
    assert result == []


def test_list_checkpoints_finds_step_dirs(tmp_path):
    """_list_checkpoints finds step directories inside size subdirs."""
    # Create a fake checkpoint structure
    ckpt_root = tmp_path / "checkpoints"
    step_dir = ckpt_root / "tiny" / "step_00000001"
    step_dir.mkdir(parents=True)
    (step_dir / "metadata.json").write_text('{"step": 1, "loss": 3.0}')

    # Point storage at tmp_path/checkpoints
    menu = _make_menu(tmp_path)
    with patch.object(menu, "_resolve_path", return_value=ckpt_root):
        result = menu._list_checkpoints()
    assert len(result) >= 1
    assert any("step_00000001" in r["label"] for r in result)
