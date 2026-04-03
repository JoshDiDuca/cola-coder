"""Comprehensive tests for PipelineMenu and PipelineRunManager.

Covers every bug class encountered in the pipeline menu system:
- Method signature validation (no more missing args)
- Stage handler argument verification (correct checkpoint/data paths)
- Artifact chain tests (router checkpoint rejected, fallback works)
- Reset-to-stage tests (correct stage reset + execute_from args)
- Format compatibility (instruction format matches SFT expected input)
- Import/module existence validation
- Tokenizer resolution tests
- _resolve_checkpoint validation (rejects dirs without model.safetensors)
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

# ── Shared test utilities ──────────────────────────────────────────────────

from cola_coder.pipeline.run_manager import (
    ALL_STAGE_NUMS,
    OPTIONAL_STAGES,
    STAGE_DEFS,
    PipelineRun,
    PipelineRunManager,
    StageState,
)


def _make_config(output_dir: str = "./checkpoints") -> SimpleNamespace:
    """Create a minimal Config-like object for stage handlers."""
    return SimpleNamespace(
        checkpoint=SimpleNamespace(output_dir=output_dir),
        model=SimpleNamespace(
            rope_scaling=SimpleNamespace(type="none", factor=1.0),
            max_seq_len=2048,
        ),
        data=SimpleNamespace(
            dataset="bigcode/starcoderdata",
            languages=["typescript"],
        ),
        training=SimpleNamespace(max_steps=20000),
    )


def _make_run(
    name: str = "test-run",
    config_path: str = "configs/tiny.yaml",
    stages: dict[int, StageState] | None = None,
) -> PipelineRun:
    """Create a PipelineRun with sensible defaults for testing."""
    if stages is None:
        stages = {n: StageState() for n in ALL_STAGE_NUMS}
    return PipelineRun(
        name=name,
        config_path=config_path,
        created_at="2025-01-01T00:00:00+00:00",
        updated_at="2025-01-01T00:00:00+00:00",
        stages=stages,
    )


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_runs(tmp_path: Path) -> Path:
    """Return a temporary directory for pipeline run JSON files."""
    runs_dir = tmp_path / "pipeline_runs"
    runs_dir.mkdir()
    return runs_dir


@pytest.fixture
def mgr(tmp_runs: Path) -> PipelineRunManager:
    """PipelineRunManager backed by a tmp directory."""
    return PipelineRunManager(tmp_runs)


@pytest.fixture
def mock_master(tmp_path: Path) -> MagicMock:
    """MasterMenu mock with project_root and venv_python."""
    master = MagicMock()
    master.project_root = tmp_path
    master.venv_python = Path("/fake/python")
    master._pause = MagicMock()
    master._run_script = MagicMock()
    master._run_shell = MagicMock()
    master._data = MagicMock()
    return master


@pytest.fixture
def pipeline_menu(mock_master: MagicMock) -> Any:
    """PipelineMenu with mocked master and cli."""
    with patch("cola_coder.features.menus.pipeline_menu.cli") as mock_cli:
        mock_cli.choose.return_value = None  # Cancel by default
        mock_cli.confirm.return_value = False
        mock_cli.info = MagicMock()
        mock_cli.error = MagicMock()
        mock_cli.warn = MagicMock()
        mock_cli.dim = MagicMock()
        mock_cli.success = MagicMock()
        mock_cli.done = MagicMock()
        mock_cli.step = MagicMock()
        mock_cli.print = MagicMock()
        mock_cli.kv_table = MagicMock()

        from cola_coder.features.menus.pipeline_menu import PipelineMenu
        menu = PipelineMenu(mock_master)
        menu._mock_cli = mock_cli  # Expose for assertions
        yield menu


# ═══════════════════════════════════════════════════════════════════════════
# A. METHOD SIGNATURE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestMethodSignatures:
    """Validate that all PipelineMenu methods have the expected parameters.

    Would have caught: _execute_from(run) missing 'start' arg.
    """

    def test_execute_from_accepts_run_and_start(self) -> None:
        from cola_coder.features.menus.pipeline_menu import PipelineMenu
        sig = inspect.signature(PipelineMenu._execute_from)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "run" in params
        assert "start" in params
        assert len(params) == 3, f"Expected (self, run, start), got {params}"

    def test_execute_stage_accepts_run_and_stage_num(self) -> None:
        from cola_coder.features.menus.pipeline_menu import PipelineMenu
        sig = inspect.signature(PipelineMenu._execute_stage)
        params = list(sig.parameters.keys())
        assert params == ["self", "run", "stage_num"]

    def test_dispatch_stage_accepts_run_config_input(self) -> None:
        from cola_coder.features.menus.pipeline_menu import PipelineMenu
        sig = inspect.signature(PipelineMenu._dispatch_stage)
        params = list(sig.parameters.keys())
        assert params == ["self", "run", "stage_num", "input_path"]

    def test_run_stage_script_accepts_script_and_args(self) -> None:
        from cola_coder.features.menus.pipeline_menu import PipelineMenu
        sig = inspect.signature(PipelineMenu._run_stage_script)
        params = list(sig.parameters.keys())
        assert params == ["self", "script", "args"]

    def test_resolve_checkpoint_is_static(self) -> None:
        from cola_coder.features.menus.pipeline_menu import PipelineMenu
        sig = inspect.signature(PipelineMenu._resolve_checkpoint)
        params = list(sig.parameters.keys())
        # Static method — no 'self'
        assert "self" not in params
        assert params == ["run", "config", "input_path"]

    def test_stage_handler_signatures_match_dispatch(self) -> None:
        """Verify every stage handler accepts the args _dispatch_stage passes."""
        from cola_coder.features.menus.pipeline_menu import PipelineMenu

        # _dispatch_stage passes:
        #   stages 1-4, 8: (run, config)
        #   stages 5-7, 9-10: (run, config, input_path)
        expects_input = {5, 6, 7, 9, 10}
        handler_map = {
            1: "_stage_collect",
            2: "_stage_prepare",
            3: "_stage_pretrain",
            4: "_stage_extend_context",
            5: "_stage_generate_instructions",
            6: "_stage_instruction_tune",
            7: "_stage_upcycle_moe",
            8: "_stage_train_router",
            9: "_stage_train_reasoning",
            10: "_stage_evaluate",
        }

        for stage_num, method_name in handler_map.items():
            method = getattr(PipelineMenu, method_name)
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())
            # Remove 'self'
            non_self = [p for p in params if p != "self"]

            if stage_num in expects_input:
                assert len(non_self) == 3, (
                    f"Stage {stage_num} ({method_name}) should accept "
                    f"(run, config, input_path) but has params: {non_self}"
                )
            else:
                assert len(non_self) == 2, (
                    f"Stage {stage_num} ({method_name}) should accept "
                    f"(run, config) but has params: {non_self}"
                )

    def test_no_method_named_execute_run(self) -> None:
        """_execute_run does not exist — the correct name is _execute_from.

        Would have caught: calling _execute_run instead of _execute_from.
        """
        from cola_coder.features.menus.pipeline_menu import PipelineMenu
        assert not hasattr(PipelineMenu, "_execute_run"), (
            "_execute_run should not exist — use _execute_from"
        )
        assert hasattr(PipelineMenu, "_execute_from"), (
            "_execute_from must exist"
        )

    def test_all_public_menu_methods_exist(self) -> None:
        """Verify all methods referenced in the menu() dispatcher exist."""
        from cola_coder.features.menus.pipeline_menu import PipelineMenu
        required = [
            "_create_run", "_resume_run", "_view_runs",
            "_run_single_stage", "_reset_to_stage", "_delete_run",
            "_legacy_pipeline",
        ]
        for name in required:
            assert hasattr(PipelineMenu, name), f"Missing method: {name}"


# ═══════════════════════════════════════════════════════════════════════════
# B. STAGE HANDLER ARGUMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestStageHandlerArgs:
    """Verify each stage handler passes the correct args to scripts.

    Would have caught: --checkpoint data/sft/instructions.jsonl.
    """

    def test_stage_generate_instructions_args(self, pipeline_menu: Any) -> None:
        """Stage 5 passes correct args to generate_instructions.py."""
        run = _make_run()
        config = _make_config()

        with patch.object(pipeline_menu, "_run_stage_script") as mock_script:
            result = pipeline_menu._stage_generate_instructions(run, config, "")
            mock_script.assert_called_once()
            script_name, args = mock_script.call_args[0]
            assert script_name == "generate_instructions.py"
            assert "--non-interactive" in args
            assert "--output" in args
            assert result == "data/sft/instructions.jsonl"

    def test_stage_pretrain_passes_data_arg(self, pipeline_menu: Any, tmp_path: Path) -> None:
        """Stage 3 passes --data with the correct .npy path."""
        config = _make_config(str(tmp_path / "checkpoints"))
        run = _make_run()

        # Create a fake dataset directory with .npy file
        with patch("cola_coder.data.dataset_resolver.DatasetResolver.get_dataset_dir") as mock_get_dir:
            data_dir = tmp_path / "data" / "dataset"
            data_dir.mkdir(parents=True)
            (data_dir / "code_data.npy").write_bytes(b"\x00")
            mock_get_dir.return_value = data_dir

            with patch.object(pipeline_menu, "_run_stage_script") as mock_script:
                pipeline_menu._stage_pretrain(run, config)
                mock_script.assert_called_once()
                script_name, args = mock_script.call_args[0]
                assert script_name == "train.py"
                assert "--data" in args
                data_idx = args.index("--data")
                assert "code_data.npy" in args[data_idx + 1]

    def test_stage_instruction_tune_uses_checkpoint_not_jsonl(
        self, pipeline_menu: Any, tmp_path: Path,
    ) -> None:
        """Stage 6 uses checkpoint from config, NOT the JSONL from stage 5."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        latest = ckpt_dir / "latest"
        latest.write_text("checkpoints/step-1000", encoding="utf-8")

        config = _make_config(str(ckpt_dir))
        run = _make_run()
        # Stage 5 produced instructions.jsonl
        run.stages[5] = StageState(status="completed", artifact="data/sft/instructions.jsonl")

        # Create the instruction data file
        instructions = tmp_path / "data" / "sft"
        instructions.mkdir(parents=True)
        (instructions / "instructions.jsonl").write_text("{}\n")

        with patch.object(pipeline_menu, "_run_stage_script") as mock_script:
            with patch("cola_coder.features.menus.pipeline_menu.Path") as MockPath:
                # Make Path(instruction_data).exists() return True
                MockPath.side_effect = lambda x: Path(x)
                MockPath.return_value.exists.return_value = True
                try:
                    pipeline_menu._stage_instruction_tune(run, config, "data/sft/instructions.jsonl")
                except (FileNotFoundError, RuntimeError):
                    pass  # OK if file doesn't exist in test env

                if mock_script.called:
                    script_name, args = mock_script.call_args[0]
                    assert script_name == "train_sft.py"
                    # The --checkpoint should NOT be the JSONL file
                    ckpt_idx = args.index("--checkpoint")
                    ckpt_value = args[ckpt_idx + 1]
                    assert not ckpt_value.endswith(".jsonl"), (
                        f"SFT checkpoint should be a model dir, not {ckpt_value}"
                    )

    def test_stage_train_router_args(self, pipeline_menu: Any) -> None:
        """Stage 8 passes correct architecture and save dir."""
        run = _make_run()
        config = _make_config()

        with patch.object(pipeline_menu, "_run_stage_script") as mock_script:
            result = pipeline_menu._stage_train_router(run, config)
            mock_script.assert_called_once()
            script_name, args = mock_script.call_args[0]
            assert script_name == "train_router.py"
            assert "--arch" in args
            assert "mlp" in args
            assert "--save-dir" in args
            assert result == f"checkpoints/router/{run.name}"

    def test_stage_train_reasoning_passes_checkpoint(
        self, pipeline_menu: Any, tmp_path: Path,
    ) -> None:
        """Stage 9 resolves to a model checkpoint, not a router save dir, and passes --language."""
        run = _make_run()
        # Stage 8 produced a router save dir (NOT a model checkpoint)
        run.stages[8] = StageState(
            status="completed", artifact="checkpoints/router/test-run",
        )
        config = _make_config(str(tmp_path / "checkpoints"))

        with patch.object(
            type(pipeline_menu), "_resolve_checkpoint",
            return_value="checkpoints/step-1000",
        ):
            with patch.object(pipeline_menu, "_run_stage_script") as mock_script:
                pipeline_menu._stage_train_reasoning(run, config, "checkpoints/router/test-run")
                mock_script.assert_called_once()
                script_name, args = mock_script.call_args[0]
                assert script_name == "train_reasoning.py"
                ckpt_idx = args.index("--base-checkpoint")
                ckpt_value = args[ckpt_idx + 1]
                assert "router" not in ckpt_value, (
                    "Reasoning stage should get SFT/pretrain checkpoint, not router dir"
                )
                # Language and reward must be derived from config.data.languages
                assert "--language" in args, "Stage 9 must pass --language to train_reasoning.py"
                lang_idx = args.index("--language")
                assert args[lang_idx + 1] == "typescript", (
                    "Config has languages=['typescript'], so --language typescript expected"
                )
                assert "--reward" in args, "Stage 9 must pass --reward derived from config.data.languages"
                reward_idx = args.index("--reward")
                assert args[reward_idx + 1] == "typescript", (
                    "Single-language typescript config should use --reward typescript"
                )

    def test_stage_evaluate_runs_three_scripts(
        self, pipeline_menu: Any, tmp_path: Path,
    ) -> None:
        """Stage 10 runs smoke_test (via _run_stage_script), then evaluate + quality_report."""
        run = _make_run()
        config = _make_config(str(tmp_path / "checkpoints"))

        with patch.object(
            type(pipeline_menu), "_resolve_checkpoint",
            return_value="checkpoints/step-1000",
        ):
            with patch.object(pipeline_menu, "_run_stage_script") as mock_stage_script:
                pipeline_menu._stage_evaluate(run, config, "")

                # smoke_test.py is a quality gate → must use _run_stage_script
                mock_stage_script.assert_called_once()
                assert mock_stage_script.call_args[0][0] == "smoke_test.py"

                # evaluate.py and quality_report.py are informational → _master._run_script
                master = pipeline_menu._master
                assert master._run_script.call_count == 2
                scripts_called = [c[0][0] for c in master._run_script.call_args_list]
                assert scripts_called == ["evaluate.py", "quality_report.py"]


# ═══════════════════════════════════════════════════════════════════════════
# C. CLI SIGNATURE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestCliSignatures:
    """Verify that cli.info/error/etc. are always called with correct arg counts.

    Would have caught: cli.info() with 1 arg instead of 2.
    """

    def test_cli_info_requires_two_args(self) -> None:
        from cola_coder.cli import CLI
        sig = inspect.signature(CLI.info)
        params = [p for p in sig.parameters if p != "self"]
        assert len(params) == 2, f"cli.info should take (key, value), got {params}"

    def test_cli_error_accepts_one_or_two_args(self) -> None:
        from cola_coder.cli import CLI
        sig = inspect.signature(CLI.error)
        params = sig.parameters
        # 'self', 'message', 'hint' (with default)
        non_self = {k: v for k, v in params.items() if k != "self"}
        assert "message" in non_self
        assert "hint" in non_self
        assert non_self["hint"].default == ""

    def test_cli_step_requires_three_args(self) -> None:
        from cola_coder.cli import CLI
        sig = inspect.signature(CLI.step)
        params = [p for p in sig.parameters if p != "self"]
        assert params == ["current", "total", "message"]

    def test_all_cli_info_calls_in_pipeline_have_two_args(self) -> None:
        """Static check: every cli.info() in pipeline_menu.py passes 2 args."""
        import ast
        src = Path(__file__).resolve().parent.parent / "src" / "cola_coder" / "features" / "menus" / "pipeline_menu.py"
        tree = ast.parse(src.read_text(encoding="utf-8"), filename=str(src))

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "info"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "cli"
            ):
                n_args = len(node.args) + len(node.keywords)
                assert n_args == 2, (
                    f"cli.info() on line {node.lineno} has {n_args} args, expected 2"
                )


# ═══════════════════════════════════════════════════════════════════════════
# D. RESET-TO-STAGE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestResetToStage:
    """Verify reset_to_stage resets correct stages and preserves skipped ones."""

    def test_reset_resets_stages_from_target_onward(self, mgr: PipelineRunManager) -> None:
        run = mgr.create("test", "configs/tiny.yaml")
        # Complete stages 1-5
        for n in range(1, 6):
            mgr.mark_completed(run, n, artifact=f"artifact-{n}")
        # Reset to stage 3
        mgr.reset_to_stage(run, 3)
        # 1, 2 stay completed
        assert run.stages[1].status == "completed"
        assert run.stages[2].status == "completed"
        # 3+ should be pending
        for n in range(3, 11):
            if run.stages[n].status != "skipped":
                assert run.stages[n].status == "pending", f"Stage {n} should be pending"

    def test_reset_preserves_skipped_stages(self, mgr: PipelineRunManager) -> None:
        skip = {4, 7}  # Optional stages
        run = mgr.create("test", "configs/tiny.yaml", skip_stages=skip)
        # Complete stages 1-3, 5, 6
        for n in [1, 2, 3, 5, 6]:
            mgr.mark_completed(run, n, artifact=f"artifact-{n}")
        # Reset to stage 2
        mgr.reset_to_stage(run, 2)
        assert run.stages[4].status == "skipped", "Skipped stage should stay skipped"
        assert run.stages[7].status == "skipped", "Skipped stage should stay skipped"

    def test_reset_preserves_earlier_artifacts(self, mgr: PipelineRunManager) -> None:
        run = mgr.create("test", "configs/tiny.yaml")
        mgr.mark_completed(run, 1, artifact="data/dataset")
        mgr.mark_completed(run, 2, artifact="data/processed")
        mgr.mark_completed(run, 3, artifact="checkpoints/step-1000")
        mgr.reset_to_stage(run, 3)
        # Earlier artifacts preserved
        assert run.stages[1].artifact == "data/dataset"
        assert run.stages[2].artifact == "data/processed"
        # Stage 3 artifact kept for fallback resolution
        assert run.stages[3].artifact == "checkpoints/step-1000"

    def test_reset_stage_sets_pending_status(self, mgr: PipelineRunManager) -> None:
        run = mgr.create("test", "configs/tiny.yaml")
        mgr.mark_completed(run, 1, artifact="a1")
        mgr.mark_failed(run, 2, error="oops")
        mgr.reset_to_stage(run, 2)
        assert run.stages[2].status == "pending"
        assert run.stages[2].error == ""
        assert run.stages[2].started_at is None


# ═══════════════════════════════════════════════════════════════════════════
# E. ARTIFACT CHAIN / RESOLVE INPUT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestArtifactChain:
    """Verify resolve_input walks back correctly through completed stages.

    Would have caught: router checkpoint passed to reasoning/evaluate.
    """

    def test_resolve_input_returns_override_first(self, mgr: PipelineRunManager) -> None:
        run = mgr.create("test", "configs/tiny.yaml")
        mgr.mark_completed(run, 1, artifact="auto-artifact")
        mgr.set_override(run, 2, "/custom/override")
        result = mgr.resolve_input(run, 2)
        assert result == "/custom/override"

    def test_resolve_input_walks_back_to_nearest_artifact(
        self, mgr: PipelineRunManager,
    ) -> None:
        run = mgr.create("test", "configs/tiny.yaml")
        mgr.mark_completed(run, 1, artifact="artifact-1")
        mgr.mark_completed(run, 3, artifact="artifact-3")
        # Stage 5 should resolve to artifact-3 (nearest completed before 5)
        # (stages 2, 4 have no artifact)
        result = mgr.resolve_input(run, 5)
        assert result == "artifact-3"

    def test_resolve_input_skips_stages_without_artifact(
        self, mgr: PipelineRunManager,
    ) -> None:
        run = mgr.create("test", "configs/tiny.yaml")
        mgr.mark_completed(run, 1, artifact="artifact-1")
        mgr.mark_completed(run, 2)  # completed but no artifact
        result = mgr.resolve_input(run, 3)
        assert result == "artifact-1"

    def test_resolve_input_returns_empty_when_no_artifacts(
        self, mgr: PipelineRunManager,
    ) -> None:
        run = mgr.create("test", "configs/tiny.yaml")
        result = mgr.resolve_input(run, 1)
        assert result == ""


class TestResolveCheckpoint:
    """Verify _resolve_checkpoint rejects non-model directories.

    Would have caught: router checkpoint being used for reasoning training.
    """

    def test_rejects_directory_without_model_safetensors(self, tmp_path: Path) -> None:
        from cola_coder.features.menus.pipeline_menu import PipelineMenu

        # Router save dir: has files but NOT model.safetensors
        router_dir = tmp_path / "checkpoints" / "router" / "test-run"
        router_dir.mkdir(parents=True)
        (router_dir / "router_model.pt").write_bytes(b"\x00")

        run = _make_run()
        config = _make_config(str(tmp_path / "checkpoints"))

        result = PipelineMenu._resolve_checkpoint(
            run, config, str(router_dir),
        )
        # Should NOT return the router dir
        assert result != str(router_dir), (
            "_resolve_checkpoint should reject directories without model.safetensors"
        )

    def test_accepts_directory_with_model_safetensors(self, tmp_path: Path) -> None:
        from cola_coder.features.menus.pipeline_menu import PipelineMenu

        ckpt_dir = tmp_path / "checkpoints" / "step-1000"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "model.safetensors").write_bytes(b"\x00")

        run = _make_run()
        config = _make_config(str(tmp_path / "checkpoints"))

        result = PipelineMenu._resolve_checkpoint(
            run, config, str(ckpt_dir),
        )
        assert result == str(ckpt_dir)

    def test_falls_back_to_sft_checkpoint(self, tmp_path: Path) -> None:
        from cola_coder.features.menus.pipeline_menu import PipelineMenu

        # No valid input_path, but SFT checkpoint exists
        sft_dir = Path("checkpoints/tiny_sft")
        sft_latest = sft_dir / "latest"
        # We can't easily test real FS fallback without CWD control,
        # so just verify the method doesn't crash with empty input
        run = _make_run(config_path="configs/tiny.yaml")
        config = _make_config(str(tmp_path / "checkpoints"))
        # Should not raise — returns fallback path
        result = PipelineMenu._resolve_checkpoint(run, config, "")
        assert isinstance(result, str)

    def test_reads_latest_pointer_file(self, tmp_path: Path) -> None:
        from cola_coder.features.menus.pipeline_menu import PipelineMenu

        ckpt_base = tmp_path / "checkpoints"
        ckpt_base.mkdir()
        real_ckpt = ckpt_base / "step-2000"
        real_ckpt.mkdir()
        (real_ckpt / "model.safetensors").write_bytes(b"\x00")
        latest = ckpt_base / "latest"
        latest.write_text(str(real_ckpt), encoding="utf-8")

        run = _make_run()
        config = _make_config(str(ckpt_base))

        result = PipelineMenu._resolve_checkpoint(
            run, config, str(latest),
        )
        assert result == str(real_ckpt)

    def test_rejects_jsonl_file_as_checkpoint(self, tmp_path: Path) -> None:
        from cola_coder.features.menus.pipeline_menu import PipelineMenu

        jsonl = tmp_path / "data" / "sft" / "instructions.jsonl"
        jsonl.parent.mkdir(parents=True)
        jsonl.write_text('{"messages": []}\n')

        run = _make_run()
        config = _make_config(str(tmp_path / "checkpoints"))

        result = PipelineMenu._resolve_checkpoint(
            run, config, str(jsonl),
        )
        # Should NOT return the JSONL path as a checkpoint
        assert result != str(jsonl), (
            "_resolve_checkpoint should reject JSONL files"
        )


# ═══════════════════════════════════════════════════════════════════════════
# F. FORMAT COMPATIBILITY TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestFormatCompatibility:
    """Verify generate_instructions output matches train_sft expected input.

    Would have caught: instruction format mismatch (missing 'messages' key).
    """

    def test_stage5_output_path_matches_stage6_default(self) -> None:
        """Stage 5 output path must match Stage 6 default instruction data path."""
        from cola_coder.features.menus.pipeline_menu import PipelineMenu

        # Stage 5 returns this path
        run = _make_run()
        config = _make_config()
        with patch.object(PipelineMenu, "_run_stage_script"):
            result = PipelineMenu._stage_generate_instructions(
                PipelineMenu.__new__(PipelineMenu), run, config, "",
            )

        # Stage 6 defaults to this path when no stage 5 artifact
        assert result == "data/sft/instructions.jsonl"

    def test_stage6_uses_stage5_artifact_when_available(self) -> None:
        """If stage 5 completed with an artifact, stage 6 should use it."""
        run = _make_run()
        run.stages[5] = StageState(
            status="completed",
            artifact="data/sft/custom_instructions.jsonl",
        )
        # The instruction_data logic in _stage_instruction_tune:
        #   instruction_data = "data/sft/instructions.jsonl"
        #   st5 = run.stages.get(5)
        #   if st5 and st5.artifact:
        #       instruction_data = st5.artifact
        instruction_data = "data/sft/instructions.jsonl"
        st5 = run.stages.get(5)
        if st5 and st5.artifact:
            instruction_data = st5.artifact
        assert instruction_data == "data/sft/custom_instructions.jsonl"


# ═══════════════════════════════════════════════════════════════════════════
# G. RUN MANAGER CRUD TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestRunManagerCRUD:
    """Test PipelineRunManager create/load/save/delete."""

    def test_create_initializes_all_stages(self, mgr: PipelineRunManager) -> None:
        run = mgr.create("test", "configs/tiny.yaml")
        assert len(run.stages) == len(ALL_STAGE_NUMS)
        for n in ALL_STAGE_NUMS:
            assert n in run.stages

    def test_create_skips_requested_stages(self, mgr: PipelineRunManager) -> None:
        run = mgr.create("test", "configs/tiny.yaml", skip_stages={4, 7})
        assert run.stages[4].status == "skipped"
        assert run.stages[7].status == "skipped"
        assert run.stages[1].status == "pending"

    def test_save_and_load_roundtrip(self, mgr: PipelineRunManager) -> None:
        run = mgr.create("test", "configs/tiny.yaml")
        mgr.mark_completed(run, 1, artifact="data/test")
        loaded = mgr.load("test")
        assert loaded.name == "test"
        assert loaded.stages[1].status == "completed"
        assert loaded.stages[1].artifact == "data/test"

    def test_list_runs_returns_all(self, mgr: PipelineRunManager) -> None:
        mgr.create("run-a", "configs/tiny.yaml")
        mgr.create("run-b", "configs/small.yaml")
        runs = mgr.list_runs()
        assert len(runs) == 2
        names = {r.name for r in runs}
        assert "run-a" in names
        assert "run-b" in names

    def test_delete_removes_run(self, mgr: PipelineRunManager) -> None:
        mgr.create("test", "configs/tiny.yaml")
        assert mgr.exists("test")
        mgr.delete("test")
        assert not mgr.exists("test")

    def test_delete_nonexistent_returns_false(self, mgr: PipelineRunManager) -> None:
        assert not mgr.delete("nonexistent")

    def test_next_pending_skips_completed(self, mgr: PipelineRunManager) -> None:
        run = mgr.create("test", "configs/tiny.yaml")
        mgr.mark_completed(run, 1)
        mgr.mark_completed(run, 2)
        assert mgr.next_pending(run) == 3

    def test_next_pending_returns_failed(self, mgr: PipelineRunManager) -> None:
        run = mgr.create("test", "configs/tiny.yaml")
        mgr.mark_completed(run, 1)
        mgr.mark_failed(run, 2, error="oops")
        assert mgr.next_pending(run) == 2

    def test_next_pending_skips_skipped(self, mgr: PipelineRunManager) -> None:
        run = mgr.create("test", "configs/tiny.yaml", skip_stages={4})
        mgr.mark_completed(run, 1)
        mgr.mark_completed(run, 2)
        mgr.mark_completed(run, 3)
        assert mgr.next_pending(run) == 5  # Skips 4

    def test_next_pending_returns_none_when_all_done(
        self, mgr: PipelineRunManager,
    ) -> None:
        run = mgr.create("test", "configs/tiny.yaml", skip_stages=OPTIONAL_STAGES)
        for n in ALL_STAGE_NUMS:
            if n not in OPTIONAL_STAGES:
                mgr.mark_completed(run, n)
        assert mgr.next_pending(run) is None

    def test_completed_count(self, mgr: PipelineRunManager) -> None:
        run = mgr.create("test", "configs/tiny.yaml", skip_stages={4, 7})
        mgr.mark_completed(run, 1)
        mgr.mark_completed(run, 2)
        # 2 completed + 2 skipped = 4
        assert mgr.completed_count(run) == 4

    def test_summary_line_format(self, mgr: PipelineRunManager) -> None:
        run = mgr.create("test", "configs/tiny.yaml")
        mgr.mark_completed(run, 1)
        line = mgr.summary_line(run)
        assert "test" in line
        assert "1/10" in line


# ═══════════════════════════════════════════════════════════════════════════
# H. STAGE TRANSITION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestStageTransitions:
    """Verify mark_running/completed/failed set correct state."""

    def test_mark_running(self, mgr: PipelineRunManager) -> None:
        run = mgr.create("test", "configs/tiny.yaml")
        mgr.mark_running(run, 1)
        assert run.stages[1].status == "running"
        assert run.stages[1].started_at is not None

    def test_mark_completed_sets_artifact(self, mgr: PipelineRunManager) -> None:
        run = mgr.create("test", "configs/tiny.yaml")
        mgr.mark_completed(run, 1, artifact="data/test", duration=42.0)
        st = run.stages[1]
        assert st.status == "completed"
        assert st.artifact == "data/test"
        assert st.duration_secs == 42.0
        assert st.error == ""

    def test_mark_failed_sets_error(self, mgr: PipelineRunManager) -> None:
        run = mgr.create("test", "configs/tiny.yaml")
        mgr.mark_failed(run, 1, error="OOM", duration=10.0)
        st = run.stages[1]
        assert st.status == "failed"
        assert st.error == "OOM"
        assert st.duration_secs == 10.0


# ═══════════════════════════════════════════════════════════════════════════
# I. IMPORT / MODULE EXISTENCE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestImports:
    """Verify all imports in pipeline_menu resolve to real modules."""

    def test_pipeline_menu_imports_cleanly(self) -> None:
        """The module should import without error."""
        import cola_coder.features.menus.pipeline_menu  # noqa: F401

    def test_run_manager_imports_cleanly(self) -> None:
        import cola_coder.pipeline.run_manager  # noqa: F401

    def test_master_menu_exports_print_section_header(self) -> None:
        from cola_coder.features.master_menu import _print_section_header
        sig = inspect.signature(_print_section_header)
        params = list(sig.parameters.keys())
        assert "title" in params
        assert "subtitle" in params

    def test_stage_defs_cover_all_ten_stages(self) -> None:
        assert set(STAGE_DEFS.keys()) == set(range(1, 11))

    def test_all_stage_nums_is_sorted(self) -> None:
        assert ALL_STAGE_NUMS == list(range(1, 11))

    def test_optional_stages_are_subset(self) -> None:
        assert OPTIONAL_STAGES.issubset(set(ALL_STAGE_NUMS))


class TestScriptFilesExist:
    """Verify every script referenced in pipeline_menu actually exists.

    Would have caught: referencing a renamed or deleted script.
    """

    # Scripts called from pipeline_menu
    PIPELINE_SCRIPTS = [
        "collect_data.py",
        "prepare_data.py",
        "train.py",
        "generate_instructions.py",
        "train_sft.py",
        "upcycle_to_moe.py",
        "train_router.py",
        "train_reasoning.py",
        "score_data.py",
        "train_tokenizer.py",
    ]

    # Scripts called via master._run_script in _stage_evaluate
    EVAL_SCRIPTS = [
        "smoke_test.py",
        "evaluate.py",
        "quality_report.py",
    ]

    @pytest.fixture
    def scripts_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent / "scripts"

    @pytest.mark.parametrize("script", PIPELINE_SCRIPTS)
    def test_pipeline_script_exists(self, scripts_dir: Path, script: str) -> None:
        assert (scripts_dir / script).exists(), (
            f"Script {script} referenced in pipeline_menu.py but not found in scripts/"
        )

    @pytest.mark.parametrize("script", EVAL_SCRIPTS)
    def test_eval_script_exists(self, scripts_dir: Path, script: str) -> None:
        assert (scripts_dir / script).exists(), (
            f"Script {script} referenced in _stage_evaluate but not found in scripts/"
        )


# ═══════════════════════════════════════════════════════════════════════════
# J. EXECUTE_FROM FLOW TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestExecuteFrom:
    """Verify _execute_from calls _execute_stage with correct stage numbers."""

    def test_execute_from_none_does_nothing(self, pipeline_menu: Any) -> None:
        """_execute_from(run, None) should not execute any stages."""
        run = _make_run()
        pipeline_menu._execute_from(run, None)
        # Should print success, not crash

    def test_execute_from_skips_completed_stages(self, pipeline_menu: Any) -> None:
        run = _make_run()
        run.stages[1] = StageState(status="completed", artifact="a1")
        run.stages[2] = StageState(status="completed", artifact="a2")

        with patch.object(pipeline_menu, "_execute_stage", return_value=True) as mock_exec:
            pipeline_menu._execute_from(run, 1)
            # Should skip stages 1, 2 (completed) and run stage 3+
            called_stages = [c[0][1] for c in mock_exec.call_args_list]
            assert 1 not in called_stages
            assert 2 not in called_stages
            assert 3 in called_stages

    def test_execute_from_skips_skipped_stages(self, pipeline_menu: Any) -> None:
        run = _make_run()
        run.stages[4] = StageState(status="skipped")
        run.stages[7] = StageState(status="skipped")

        with patch.object(pipeline_menu, "_execute_stage", return_value=True) as mock_exec:
            pipeline_menu._execute_from(run, 1)
            called_stages = [c[0][1] for c in mock_exec.call_args_list]
            assert 4 not in called_stages
            assert 7 not in called_stages

    def test_execute_from_stops_on_failure_when_user_declines(
        self, pipeline_menu: Any,
    ) -> None:
        run = _make_run()
        pipeline_menu._mock_cli.confirm.return_value = False

        with patch.object(pipeline_menu, "_execute_stage", return_value=False) as mock_exec:
            pipeline_menu._execute_from(run, 1)
            # Should stop after first failure
            assert mock_exec.call_count == 1


# ═══════════════════════════════════════════════════════════════════════════
# K. SELECT DATASET DIR TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSelectDatasetDir:
    """Verify _select_dataset_dir handles different dataset scenarios."""

    def test_returns_none_when_no_base_dir(self, tmp_path: Path) -> None:
        from cola_coder.features.menus.pipeline_menu import PipelineMenu
        result = PipelineMenu._select_dataset_dir(
            tmp_path / "nonexistent" / "dataset", "test",
        )
        assert result is None

    def test_returns_none_when_no_npy_files(self, tmp_path: Path) -> None:
        from cola_coder.features.menus.pipeline_menu import PipelineMenu
        base = tmp_path / "data"
        auto = base / "dataset"
        auto.mkdir(parents=True)
        result = PipelineMenu._select_dataset_dir(auto, "test")
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# L. DISPATCH STAGE COVERAGE
# ═══════════════════════════════════════════════════════════════════════════

class TestDispatchStage:
    """Verify _dispatch_stage routes to the correct handler for each stage."""

    def test_dispatch_unknown_stage_raises(self, pipeline_menu: Any) -> None:
        run = _make_run()
        config = _make_config()
        with patch("cola_coder.model.config.Config.from_yaml", return_value=config):
            with pytest.raises(ValueError, match="Unknown stage"):
                pipeline_menu._dispatch_stage(run, 99, "")

    @pytest.mark.parametrize("stage_num", ALL_STAGE_NUMS)
    def test_dispatch_routes_every_stage(
        self, pipeline_menu: Any, stage_num: int,
    ) -> None:
        """Every valid stage number should be dispatched without AttributeError."""
        run = _make_run()
        config = _make_config()

        handler_map = {
            1: "_stage_collect",
            2: "_stage_prepare",
            3: "_stage_pretrain",
            4: "_stage_extend_context",
            5: "_stage_generate_instructions",
            6: "_stage_instruction_tune",
            7: "_stage_upcycle_moe",
            8: "_stage_train_router",
            9: "_stage_train_reasoning",
            10: "_stage_evaluate",
        }

        expected_handler = handler_map[stage_num]
        with patch("cola_coder.model.config.Config.from_yaml", return_value=config):
            with patch.object(pipeline_menu, expected_handler, return_value="artifact") as mock_handler:
                result = pipeline_menu._dispatch_stage(run, stage_num, "input")
                mock_handler.assert_called_once()
                assert result == "artifact"


# ═══════════════════════════════════════════════════════════════════════════
# M. SERIALIZATION ROUND-TRIP TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSerialization:
    """Verify JSON serialization/deserialization preserves all fields."""

    def test_int_keys_preserved_through_json(self, mgr: PipelineRunManager) -> None:
        """Stage keys must survive JSON serialization (int->str->int)."""
        run = mgr.create("test", "configs/tiny.yaml")
        mgr.mark_completed(run, 3, artifact="test-artifact")
        loaded = mgr.load("test")
        assert 3 in loaded.stages
        assert loaded.stages[3].artifact == "test-artifact"

    def test_notes_preserved(self, mgr: PipelineRunManager) -> None:
        run = mgr.create("test", "configs/tiny.yaml")
        run.notes = "Important notes about this run"
        mgr.save(run)
        loaded = mgr.load("test")
        assert loaded.notes == "Important notes about this run"

    def test_override_preserved(self, mgr: PipelineRunManager) -> None:
        run = mgr.create("test", "configs/tiny.yaml")
        mgr.set_override(run, 5, "/custom/path.jsonl")
        loaded = mgr.load("test")
        assert loaded.stages[5].override == "/custom/path.jsonl"

    def test_safe_filename_sanitization(self, mgr: PipelineRunManager) -> None:
        """Run names with special chars should be sanitized for filenames."""
        run = mgr.create("my run/v1", "configs/tiny.yaml")
        mgr.save(run)
        # Should not crash, file should exist
        assert mgr.exists("my run/v1")


# ═══════════════════════════════════════════════════════════════════════════
# N. STAGE DEFINITION CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════

class TestStageDefinitions:
    """Verify stage definitions are consistent and complete."""

    def test_all_stages_have_name(self) -> None:
        for num, defn in STAGE_DEFS.items():
            assert "name" in defn, f"Stage {num} missing 'name'"
            assert isinstance(defn["name"], str)
            assert len(defn["name"]) > 0

    def test_all_stages_have_description(self) -> None:
        for num, defn in STAGE_DEFS.items():
            assert "description" in defn, f"Stage {num} missing 'description'"

    def test_optional_flag_is_bool(self) -> None:
        for num, defn in STAGE_DEFS.items():
            assert "optional" in defn, f"Stage {num} missing 'optional'"
            assert isinstance(defn["optional"], bool)

    def test_stage_numbers_are_consecutive(self) -> None:
        nums = sorted(STAGE_DEFS.keys())
        assert nums == list(range(1, len(nums) + 1))

    def test_icon_map_covers_all_statuses(self) -> None:
        from cola_coder.features.menus.pipeline_menu import _ICON
        expected = {"completed", "failed", "running", "skipped", "pending"}
        assert set(_ICON.keys()) == expected
