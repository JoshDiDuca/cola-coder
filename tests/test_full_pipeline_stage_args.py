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


def _cfg(languages, max_steps, out_dir):
    return types.SimpleNamespace(
        data=types.SimpleNamespace(languages=languages, dataset="bigcode/the-stack-v2-dedup"),
        training=types.SimpleNamespace(max_steps=max_steps),
        checkpoint=types.SimpleNamespace(output_dir=out_dir),
    )


class TestStage6Sft:
    def test_epochs_scale_with_model_size(self, captured_cmd, tmp_path):
        fp, calls = captured_cmd
        # Stage 6 reads data/sft/instructions.jsonl + checkpoints/<...>/latest;
        # create them so the stage doesn't bail before building the command.
        (tmp_path / "data" / "sft").mkdir(parents=True)
        (tmp_path / "data" / "sft" / "instructions.jsonl").write_text("{}\n")
        ck = tmp_path / "ck"
        (ck / "latest").mkdir(parents=True)
        import os
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            args = types.SimpleNamespace(config="configs/tiny.yaml", tokenizer=None)
            # tiny: max_steps 20000 -> 3 epochs
            fp._stage_instruction_tune(_cfg(["typescript"], 20000, str(ck)), args)
            cmd = calls[-1]
            assert cmd[cmd.index("--epochs") + 1] == "3"
            calls.clear()
            # medium: max_steps 150000 -> 2 epochs
            fp._stage_instruction_tune(_cfg(["typescript"], 150000, str(ck)), args)
            assert calls[-1][calls[-1].index("--epochs") + 1] == "2"
        finally:
            os.chdir(cwd)


class TestStage9Reasoning:
    def _run(self, fp, calls, languages, tmp_path):
        ck = tmp_path / "ckpt"
        (ck / "latest").mkdir(parents=True, exist_ok=True)
        args = types.SimpleNamespace(config="configs/4080_max.yaml", tokenizer=None)
        fp._stage_train_reasoning(_cfg(languages, 200000, str(ck)), args)
        return calls[-1]

    def test_uses_model_config_not_reasoning_yaml(self, captured_cmd, tmp_path):
        fp, calls = captured_cmd
        cmd = self._run(fp, calls, ["typescript"], tmp_path)
        # The model config (args.config) must be passed, NOT configs/reasoning.yaml
        # (whose fixed 101M model would mismatch the checkpoint).
        assert cmd[cmd.index("--config") + 1] == "configs/4080_max.yaml"
        assert "configs/reasoning.yaml" not in cmd

    def test_reward_derived_from_languages(self, captured_cmd, tmp_path):
        fp, calls = captured_cmd
        cmd = self._run(fp, calls, ["typescript"], tmp_path)
        assert cmd[cmd.index("--reward") + 1] == "typescript"
        cmd = self._run(fp, calls, ["python"], tmp_path)
        assert cmd[cmd.index("--reward") + 1] == "python_exec"
        cmd = self._run(fp, calls, ["typescript", "python"], tmp_path)
        assert cmd[cmd.index("--reward") + 1] == "combined"

    def test_group_size_present(self, captured_cmd, tmp_path):
        fp, calls = captured_cmd
        cmd = self._run(fp, calls, ["typescript"], tmp_path)
        assert "--group-size" in cmd and cmd[cmd.index("--group-size") + 1] == "16"
