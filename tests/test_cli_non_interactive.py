"""BUG-116: interactive CLI prompts must not CRASH in non-interactive contexts.

When a script is run as a pipeline subprocess / with redirected I/O / in a
terminal prompt_toolkit can't drive, questionary's `.ask()` raises
NoConsoleScreenBufferError on Windows — previously uncaught, so e.g. the full
pipeline crashed at prepare_data's "overwrite existing data?" chooser. Both
cli.choose and cli.confirm now fall through and degrade to a default instead of
crashing or looping forever on EOF.
"""

import builtins

import pytest

from cola_coder.cli import cli


def _raise(exc):
    def _f(*a, **k):
        raise exc
    return _f


class _BoomPrompt:
    """Mimics questionary's prompt object whose .ask() fails (no console)."""

    def ask(self):
        raise RuntimeError("No Windows console found. Are you running cmd.exe?")


@pytest.fixture
def no_console(monkeypatch):
    """Force the questionary path to fail AND stdin to be unavailable."""
    import questionary

    monkeypatch.setattr(questionary, "select", lambda *a, **k: _BoomPrompt())
    monkeypatch.setattr(questionary, "confirm", lambda *a, **k: _BoomPrompt())
    monkeypatch.setattr(builtins, "input", _raise(EOFError()))


class TestChooseNonInteractive:
    def test_returns_explicit_default(self, no_console):
        opts = [{"label": "Create new"}, {"label": "Overwrite"}]
        assert cli.choose("?", opts, allow_cancel=True, default=0) == 0

    def test_allow_cancel_defaults_to_none(self, no_console):
        opts = [{"label": "a"}, {"label": "b"}]
        assert cli.choose("?", opts, allow_cancel=True) is None

    def test_no_cancel_defaults_to_first(self, no_console):
        opts = [{"label": "a"}, {"label": "b"}]
        assert cli.choose("?", opts) == 0

    def test_does_not_raise(self, no_console):
        # The whole point: no NoConsoleScreenBufferError / infinite loop.
        cli.choose("pick", [{"label": "x"}], default=0)


class TestConfirmNonInteractive:
    def test_returns_default_true(self, no_console):
        assert cli.confirm("proceed?", default=True) is True

    def test_returns_default_false(self, no_console):
        assert cli.confirm("overwrite?", default=False) is False


class TestPrepareDataResolveOutput:
    """The exact crash the user hit: prepare_data found existing data and
    prompted to overwrite, but as a pipeline subprocess there was no console."""

    def _load_prepare_data(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).parent.parent / "scripts" / "prepare_data.py"
        spec = importlib.util.spec_from_file_location("prepare_data_script", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_non_interactive_creates_new_instead_of_crashing(self, no_console, monkeypatch):
        mod = self._load_prepare_data()
        # Pretend a dataset already exists (would trigger the overwrite chooser).
        monkeypatch.setattr(mod, "_scan_datasets", lambda d: [
            {"name": "code_data", "size": "13 GB", "date": "2026-03-25"},
        ])
        monkeypatch.setattr(mod, "_auto_name", lambda d, langs, mt: "auto_new")
        # Must NOT raise NoConsoleScreenBufferError; picks "create new" (default=0).
        name = mod._resolve_output("out", ["typescript"], None)
        assert name == "auto_new"
