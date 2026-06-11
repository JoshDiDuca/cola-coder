"""TOOL-010: full_pipeline.py stage-arg wiring (the project's stage-arg rules).

full_pipeline.py is a documented end-to-end runner that diverged from the
audited Pipeline Manager (pipeline_menu.py). Stage 5 ran
`generate_instructions.py` with NO args, which (a) drops into the INTERACTIVE
menu and hangs forever in an unattended pipeline, (b) ignores the config's
languages (defaults to typescript), and (c) writes to the script's default
output path, NOT data/sft/instructions.jsonl where Stage 6 reads from (so Stage 6
would FileNotFoundError even if Stage 5 didn't hang).

These tests load the script module and capture the subprocess invocation.
"""

import importlib.util
import types
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "full_pipeline.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("full_pipeline_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fp():
    return _load_module()


@pytest.fixture
def captured_cmd(fp, monkeypatch):
    calls: list[list[str]] = []
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    return fp, calls


def _run_stage5(fp, calls, languages, dataset="bigcode/the-stack-v2-dedup"):
    cfg = types.SimpleNamespace(data=types.SimpleNamespace(languages=languages, dataset=dataset))
    args = types.SimpleNamespace(config="configs/medium.yaml", tokenizer=None)
    fp._stage_generate_instructions(cfg, args)
    assert len(calls) == 1
    return calls[0]


class TestStage5GenerateInstructions:
    def test_runs_non_interactive(self, captured_cmd):
        fp, calls = captured_cmd
        cmd = _run_stage5(fp, calls, ["typescript"])
        # Must NOT hang on the interactive menu.
        assert "--non-interactive" in cmd

    def test_passes_config_languages(self, captured_cmd):
        fp, calls = captured_cmd
        cmd = _run_stage5(fp, calls, ["typescript", "python"])
        i = cmd.index("--languages")
        # All config languages are forwarded (nargs="+").
        assert cmd[i + 1] == "typescript"
        assert cmd[i + 2] == "python"

    def test_output_matches_stage6_input(self, captured_cmd):
        fp, calls = captured_cmd
        cmd = _run_stage5(fp, calls, ["typescript"])
        i = cmd.index("--output")
        # Stage 6 (_stage_instruction_tune) reads exactly this path.
        assert cmd[i + 1] == "data/sft/instructions.jsonl"

    def test_uses_config_dataset_and_hf_source(self, captured_cmd):
        fp, calls = captured_cmd
        cmd = _run_stage5(fp, calls, ["typescript"], dataset="bigcode/starcoderdata")
        assert "--source" in cmd and cmd[cmd.index("--source") + 1] == "huggingface"
        assert "--dataset" in cmd and cmd[cmd.index("--dataset") + 1] == "bigcode/starcoderdata"
