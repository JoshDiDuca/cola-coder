"""Parity guard: the UI's typed argument forms must stay 1:1 with the CLI.

The web UI renders a typed form per launchable action from ``ACTION_PARAMS``
(``src/cola_coder/ui/action_params.py``), which mirrors each script's argparse
1:1. This test locks that invariant so a flag renamed/removed in a script (or a
new action added to ``ACTIONS`` without a param spec) fails CI instead of
silently producing a UI form that builds wrong/incomplete command lines.

It is purely STATIC — it reads script source text and never imports or executes
a script (the scripts have heavy/side-effecting imports, and executing scraped
or arbitrary code is against project security rules).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cola_coder.ui.action_params import ACTION_PARAMS
from cola_coder.ui.app import ACTIONS
from cola_coder.ui.schemas import ActionParam

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

# Flags handled by argparse itself / shared helpers rather than an explicit
# add_argument literal in the script body. None today, but kept as the documented
# escape hatch so a justified exception is explicit rather than a silent skip.
_FLAG_EXEMPTIONS: dict[str, set[str]] = {}


def _script_for(action_key: str) -> Path:
    """Resolve the script file backing an action key."""
    spec = ACTIONS[action_key]
    return _SCRIPTS_DIR / str(spec["script"])


def test_every_action_with_params_is_in_actions() -> None:
    """Every ACTION_PARAMS key must correspond to a real ACTIONS entry."""
    unknown = sorted(set(ACTION_PARAMS) - set(ACTIONS))
    assert not unknown, (
        f"ACTION_PARAMS has keys not in ACTIONS (dead specs): {unknown}. "
        "Either add the action to ACTIONS or remove the spec."
    )


def test_trainer_and_gpu_actions_have_param_specs() -> None:
    """Actions a user launches with arguments should expose a typed form.

    Specifically: every trainer-class or gpu-class action (the ones with real,
    non-trivial argument surfaces) must have a non-empty param spec so the UI
    never falls back to a raw flag-string box for them.
    """
    missing: list[str] = []
    for key, spec in ACTIONS.items():
        if (spec.get("trainer") or spec.get("gpu")) and not ACTION_PARAMS.get(key):
            missing.append(key)
    assert not missing, (
        f"trainer/gpu actions without a typed param spec: {sorted(missing)}. "
        "Add a 1:1 argparse spec to action_params.py."
    )


@pytest.mark.parametrize("action_key", sorted(ACTION_PARAMS))
def test_action_param_flags_exist_in_script(action_key: str) -> None:
    """Every non-positional flag in a spec must appear in its script's source.

    Catches drift where a script renames or drops a flag but the UI spec (and
    thus the rendered form) still references the old flag.
    """
    if action_key not in ACTIONS:
        pytest.skip(f"{action_key!r} not in ACTIONS (covered by the dead-spec test)")

    script_path = _script_for(action_key)
    assert script_path.exists(), f"script for {action_key!r} not found: {script_path}"
    source = script_path.read_text(encoding="utf-8", errors="replace")
    exempt = _FLAG_EXEMPTIONS.get(action_key, set())

    params: list[ActionParam] = ACTION_PARAMS[action_key]
    missing: list[str] = []
    for param in params:
        if param.flag == "" or param.flag in exempt:
            continue  # positional argument or documented exemption
        # Match the flag as a whole token inside an add_argument("...") literal,
        # i.e. quoted exactly. Avoids matching a flag that is merely a substring
        # of another (e.g. --data vs --data-sources).
        pattern = rf'["\']{re.escape(param.flag)}["\']'
        if not re.search(pattern, source):
            missing.append(param.flag)

    assert not missing, (
        f"{action_key}: flags in action_params.py not found in {script_path.name}: "
        f"{missing}. The UI form is no longer 1:1 with the script's argparse."
    )


@pytest.mark.parametrize("action_key", sorted(ACTION_PARAMS))
def test_action_param_specs_are_internally_consistent(action_key: str) -> None:
    """Each spec is well-formed: choice params have choices; names are unique."""
    params = ACTION_PARAMS[action_key]
    names = [p.name for p in params]
    assert len(names) == len(set(names)), f"{action_key}: duplicate param names {names}"
    for param in params:
        if param.type == "choice":
            assert param.choices, (
                f"{action_key}.{param.name}: type 'choice' but no choices listed"
            )
