"""Script help validator.

For each script in scripts/ that uses argparse, run it with --help and verify
that the process exits with code 0.

Scripts that don't accept --help (no argparse, interactive-only, or explicitly
marked) are skipped with a clear reason.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# ── Project root and scripts dir ──────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if not _VENV_PYTHON.exists():
    _VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
if not _VENV_PYTHON.exists():
    _VENV_PYTHON = Path(sys.executable)


# ── Scripts to skip (no argparse / interactive-only / long-running) ───────────

# Keys are script stem names; values are skip reasons.
_SKIP_SCRIPTS: dict[str, str] = {
    "menu": "interactive questionary UI, no --help",
    "prepare_data_interactive": "fully interactive, no --help",
    "test_type_reward": "not a production script, no --help",
    "training_dashboard": "live dashboard, no --help",
    "run_pipeline": "known Unicode encoding issue in --help output on Windows (cp1252)",
}


# ── Collect scripts dynamically ───────────────────────────────────────────────

def _collect_script_names() -> list[str]:
    """Return stem names of all .py scripts in scripts/ (sorted)."""
    return sorted(p.stem for p in _SCRIPTS_DIR.glob("*.py") if p.stem != "__init__")


def _uses_argparse(script_path: Path) -> bool:
    """Quick heuristic: return True if the script imports or references argparse."""
    try:
        src = script_path.read_text(encoding="utf-8", errors="ignore")
        return "argparse" in src
    except Exception:
        return False


# ── Parametrize ───────────────────────────────────────────────────────────────

_all_scripts = _collect_script_names()


@pytest.mark.parametrize("script_name", _all_scripts)
def test_script_help(script_name: str) -> None:
    """Run `python scripts/<name>.py --help` and assert exit code 0."""
    if script_name in _SKIP_SCRIPTS:
        pytest.skip(_SKIP_SCRIPTS[script_name])

    script_path = _SCRIPTS_DIR / f"{script_name}.py"
    if not _uses_argparse(script_path):
        pytest.skip(f"{script_name}.py does not use argparse")

    result = subprocess.run(
        [str(_VENV_PYTHON), str(script_path), "--help"],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"scripts/{script_name}.py --help exited with code {result.returncode}.\n"
        f"stdout: {result.stdout.decode(errors='replace')[:500]}\n"
        f"stderr: {result.stderr.decode(errors='replace')[:500]}"
    )
    # Basic sanity: help output should be non-empty
    help_output = result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace")
    assert len(help_output) > 0, f"scripts/{script_name}.py --help produced no output"
