"""
Feature configuration loader for Cola-Coder optional feature modules.

Each feature module under this package exposes a ``FEATURE_ENABLED`` boolean.
This package provides utilities to read/write those flags from a central YAML
file so the toggles are no longer disconnected hard-codes.

Usage
-----
Explicit load (call once at startup, e.g. in your train script)::

    from cola_coder.features import load_feature_config
    load_feature_config()          # reads configs/features.yaml automatically

Runtime toggle::

    from cola_coder.features import set_feature_enabled
    set_feature_enabled("moe_layer", True)

Inspect state::

    from cola_coder.features import get_feature_status
    print(get_feature_status())

Config path resolution order
-----------------------------
1. ``COLA_FEATURES_CONFIG`` environment variable (absolute or relative path).
2. ``configs/features.yaml`` relative to the project root.
   Project root is the first ancestor directory that contains either
   ``pyproject.toml`` or a ``configs/`` subdirectory.
3. If no file is found, all features keep their module-level defaults (True).
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PACKAGE_DIR = Path(__file__).parent
_FEATURES_PACKAGE = "cola_coder.features"


def _find_project_root() -> Optional[Path]:
    """Walk up from this file until we find pyproject.toml or a configs/ dir."""
    candidate = _PACKAGE_DIR
    for _ in range(10):  # never walk more than 10 levels
        candidate = candidate.parent
        if (candidate / "pyproject.toml").exists() or (candidate / "configs").is_dir():
            return candidate
    return None


def _default_config_path() -> Optional[Path]:
    """Return the default features.yaml path, or None if it cannot be found."""
    # 1. Explicit env var
    env_val = os.environ.get("COLA_FEATURES_CONFIG")
    if env_val:
        p = Path(env_val)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p

    # 2. Project root + configs/features.yaml
    root = _find_project_root()
    if root is not None:
        candidate = root / "configs" / "features.yaml"
        if candidate.exists():
            return candidate

    return None


def _iter_feature_names() -> list[str]:
    """Return module stems for all .py files in this package (excluding __init__)."""
    return sorted(
        p.stem
        for p in _PACKAGE_DIR.glob("*.py")
        if p.stem != "__init__"
    )


def _get_module(feature_name: str) -> Optional[ModuleType]:
    """
    Return the already-imported module for *feature_name*, or None.

    We deliberately never import a module here — only patch one that has
    already been imported by user code.  This keeps __init__ load-lazy.
    """
    full_name = f"{_FEATURES_PACKAGE}.{feature_name}"
    return sys.modules.get(full_name)


def _patch_module(feature_name: str, enabled: bool) -> None:
    """Set FEATURE_ENABLED on *feature_name* if it is already imported."""
    mod = _get_module(feature_name)
    if mod is not None:
        try:
            mod.FEATURE_ENABLED = enabled
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not set FEATURE_ENABLED on feature '%s': %s", feature_name, exc
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_feature_config(config_path: str | None = None) -> dict[str, bool]:
    """Load feature toggles from YAML and apply them to any already-imported modules.

    Parameters
    ----------
    config_path:
        Path to a YAML file.  If *None*, the default resolution order is used
        (env var → configs/features.yaml relative to project root).

    Returns
    -------
    dict[str, bool]
        Mapping of ``{module_name: enabled}`` as read from the file.
        Returns an empty dict if no config file is found.
    """
    try:
        import yaml  # soft dependency — only needed when this function is called
    except ImportError:
        logger.warning(
            "PyYAML is not installed; feature config cannot be loaded. "
            "Install it with: pip install pyyaml"
        )
        return {}

    path: Optional[Path] = None
    if config_path is not None:
        path = Path(config_path)
    else:
        path = _default_config_path()

    if path is None or not path.exists():
        if config_path is not None:
            logger.warning("Feature config not found: %s", config_path)
        return {}

    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read feature config '%s': %s", path, exc)
        return {}

    features_section = raw.get("features") or {}
    if not isinstance(features_section, dict):
        logger.warning(
            "Feature config '%s' has an unexpected format (expected a 'features:' mapping).",
            path,
        )
        return {}

    config: dict[str, bool] = {}
    for name, value in features_section.items():
        if not isinstance(value, bool):
            logger.warning(
                "Feature '%s' in config has non-bool value %r; skipping.", name, value
            )
            continue
        config[name] = value
        _patch_module(name, value)

    logger.debug("Loaded %d feature toggle(s) from '%s'.", len(config), path)
    return config


def save_feature_config(
    config: dict[str, bool],
    config_path: str | None = None,
) -> None:
    """Persist feature toggles to a YAML file.

    Parameters
    ----------
    config:
        Mapping of ``{module_name: enabled}``.
    config_path:
        Destination path.  If *None*, the default resolution order is used.
        If the resolved path still cannot be determined, a ``RuntimeError``
        is raised.
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to save feature config. "
            "Install it with: pip install pyyaml"
        ) from exc

    path: Optional[Path] = None
    if config_path is not None:
        path = Path(config_path)
    else:
        path = _default_config_path()
        if path is None:
            # Fall back to creating the file in the project root
            root = _find_project_root()
            if root is None:
                raise RuntimeError(
                    "Cannot determine project root to save features.yaml. "
                    "Pass an explicit config_path."
                )
            path = root / "configs" / "features.yaml"

    path.parent.mkdir(parents=True, exist_ok=True)

    data = {"features": config}
    try:
        with path.open("w", encoding="utf-8") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=True)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to write feature config to '%s': %s", path, exc)
        raise

    logger.debug("Saved %d feature toggle(s) to '%s'.", len(config), path)


def get_feature_status() -> dict[str, bool]:
    """Return the current FEATURE_ENABLED value for every feature module.

    Only modules that have already been imported into ``sys.modules`` will
    reflect any runtime changes made via :func:`set_feature_enabled`.
    Modules that have not yet been imported are reported at their on-disk
    default (``True`` unless overridden in the source).

    Returns
    -------
    dict[str, bool]
        ``{module_name: FEATURE_ENABLED}``, covering all 81 feature files.
    """
    status: dict[str, bool] = {}
    for name in _iter_feature_names():
        mod = _get_module(name)
        if mod is not None:
            status[name] = bool(getattr(mod, "FEATURE_ENABLED", True))
        else:
            # Module not imported yet — peek at the source-level default by
            # importing it now (in a try/except so broken modules never crash).
            try:
                full = f"{_FEATURES_PACKAGE}.{name}"
                imported = importlib.import_module(full)
                status[name] = bool(getattr(imported, "FEATURE_ENABLED", True))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not import feature module '%s' to read status: %s", name, exc
                )
                status[name] = True  # assume enabled if we can't tell
    return status


def set_feature_enabled(module_name: str, enabled: bool) -> None:
    """Set a feature's FEATURE_ENABLED flag at runtime.

    If the module has not been imported yet it will be imported now so the
    flag can be set before any consumer code reaches it.

    Parameters
    ----------
    module_name:
        Bare module stem, e.g. ``"moe_layer"``.
    enabled:
        ``True`` to enable, ``False`` to disable.

    Raises
    ------
    ValueError
        If *module_name* does not correspond to any known feature module.
    """
    known = _iter_feature_names()
    if module_name not in known:
        raise ValueError(
            f"Unknown feature module: '{module_name}'. "
            f"Available modules: {known}"
        )

    full_name = f"{_FEATURES_PACKAGE}.{module_name}"
    mod = sys.modules.get(full_name)
    if mod is None:
        try:
            mod = importlib.import_module(full_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not import feature module '%s': %s. Flag not set.", module_name, exc
            )
            return

    try:
        mod.FEATURE_ENABLED = enabled
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not set FEATURE_ENABLED on '%s': %s", module_name, exc
        )


def list_features() -> list[str]:
    """Return a sorted list of all available feature module names.

    Returns
    -------
    list[str]
        One entry per ``.py`` file in this package (excluding ``__init__``).
    """
    return _iter_feature_names()
