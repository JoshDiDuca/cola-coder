"""Menu → script wiring validation.

Statically verifies that every script a menu invokes actually exists and
that every literal --flag the menu passes is declared by the target
script's argparse. Catches the two most common menu bugs (deleted/renamed
scripts and renamed flags) without running anything.

Also enforces the project rule that every script has a menu entry
(no orphan scripts).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MENU_MODULES = sorted(
    list((PROJECT_ROOT / "src" / "cola_coder" / "features" / "menus").glob("*.py"))
    + [PROJECT_ROOT / "src" / "cola_coder" / "features" / "master_menu.py"]
)

_RUNNER_NAMES = {"_run_script", "_run_stage_script"}

# Scripts that are legitimately not wired into any menu.
_MENU_EXEMPT = {
    "menu.py",            # IS the menu
    "background_train.py",  # launched via subprocess.Popen in training_menu
    "full_pipeline.py",     # invoked by auto_pipeline.py + training menu
}


def _extract_invocations() -> list[tuple[str, str, list[str]]]:
    """Return (menu_file, script_name, literal_flags) for every runner call."""
    invocations: list[tuple[str, str, list[str]]] = []
    for menu_path in MENU_MODULES:
        tree = ast.parse(menu_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "attr", getattr(func, "id", ""))
            if name not in _RUNNER_NAMES:
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            script = node.args[0].value
            if not isinstance(script, str) or not script.endswith(".py"):
                continue
            flags = [
                el.value
                for arg in node.args[1:]
                if isinstance(arg, ast.List)
                for el in arg.elts
                if isinstance(el, ast.Constant)
                and isinstance(el.value, str)
                and el.value.startswith("--")
            ]
            invocations.append((menu_path.name, script, flags))
    return invocations


def _declared_flags(script_path: Path) -> set[str] | None:
    """Flags declared via argparse in *script_path*, or None if no argparse."""
    source = script_path.read_text(encoding="utf-8")
    flags = set(re.findall(r"add_argument\(\s*['\"](--[\w-]+)['\"]", source))
    if not flags and "add_argument" not in source:
        return None  # No argparse — flag validation not applicable
    return flags


_INVOCATIONS = _extract_invocations()


class TestMenuScriptWiring:
    def test_invocations_found(self):
        """Sanity: the extractor sees a realistic number of menu calls."""
        assert len(_INVOCATIONS) > 30, (
            f"Only {len(_INVOCATIONS)} runner calls found — extractor broke?"
        )

    @pytest.mark.parametrize(
        "menu,script,flags",
        _INVOCATIONS,
        ids=[f"{m}:{s}" for m, s, _ in _INVOCATIONS],
    )
    def test_script_exists_and_flags_declared(
        self, menu: str, script: str, flags: list[str],
    ):
        script_path = SCRIPTS_DIR / script
        assert script_path.exists(), (
            f"{menu} invokes scripts/{script} which does not exist"
        )

        declared = _declared_flags(script_path)
        if declared is None:
            return  # Script has no argparse — nothing to validate

        unknown = [f for f in flags if f not in declared]
        assert not unknown, (
            f"{menu} passes {unknown} to scripts/{script}, but its argparse "
            f"only declares: {sorted(declared)}"
        )


class TestNoOrphanScripts:
    def test_every_script_has_a_menu_entry(self):
        """Project rule: every user-facing script must be wired into a menu."""
        menu_sources = "\n".join(
            p.read_text(encoding="utf-8") for p in MENU_MODULES
        )
        orphans = []
        for script_path in sorted(SCRIPTS_DIR.glob("*.py")):
            name = script_path.name
            if name in _MENU_EXEMPT:
                continue
            if f'"{name}"' not in menu_sources and f"'{name}'" not in menu_sources:
                orphans.append(name)
        assert not orphans, (
            f"Scripts with no menu entry (rule: no orphan scripts): {orphans}. "
            f"Wire them into a menu or add to _MENU_EXEMPT with a reason."
        )
