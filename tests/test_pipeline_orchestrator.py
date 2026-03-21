"""Tests for PipelineOrchestrator.

All tests mock subprocess calls so no actual training or evaluation happens.

Run:
    cd "C:/Users/josh/ai research/cola-coder"
    .venv/Scripts/pytest tests/test_pipeline_orchestrator.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cola_coder.pipeline.orchestrator import (
    PipelineOrchestrator,
    PipelineStage,
    StageResult,
    _fmt_duration,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def config_path(tmp_path: Path) -> str:
    """Write a minimal tiny.yaml config and return its path."""
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(
        "model:\n  vocab_size: 32768\n  dim: 512\n  n_layers: 8\n"
        "  n_heads: 8\n  n_kv_heads: 4\n  max_seq_len: 1024\n"
        "training:\n  batch_size: 4\n  max_steps: 10\n  precision: bf16\n"
        "data:\n  dataset: bigcode/starcoderdata\n  languages: [python]\n"
        "  data_dir: ./data\n"
        "checkpoint:\n  output_dir: ./checkpoints/tiny\n  save_every: 5\n",
        encoding="utf-8",
    )
    return str(cfg)


@pytest.fixture
def orchestrator(config_path: str, tmp_path: Path) -> PipelineOrchestrator:
    """Return an orchestrator with a tmp log dir and no skip_existing by default."""
    return PipelineOrchestrator(
        config_path=config_path,
        skip_existing=False,
        dry_run=False,
        log_dir=str(tmp_path / "logs"),
    )


def _make_completed_proc(returncode: int = 0) -> MagicMock:
    """Return a mock CompletedProcess object."""
    mock = MagicMock()
    mock.returncode = returncode
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# 1. Stage ordering
# ─────────────────────────────────────────────────────────────────────────────


class TestStageOrdering:
    def test_default_stages_are_in_canonical_order(self, config_path: str, tmp_path: Path):
        """Default stage list must follow the pipeline dependency order."""
        orch = PipelineOrchestrator(config_path=config_path, log_dir=str(tmp_path / "logs"))
        expected = [
            PipelineStage.TOKENIZER,
            PipelineStage.DATA_PREP,
            PipelineStage.TRAINING,
            PipelineStage.SMOKE_TEST,
            PipelineStage.EVALUATION,
            PipelineStage.EXPORT,
        ]
        assert orch.stages == expected

    def test_custom_stages_are_respected(self, config_path: str, tmp_path: Path):
        """Orchestrator should honour a user-specified subset of stages."""
        custom = [PipelineStage.TRAINING, PipelineStage.EXPORT]
        orch = PipelineOrchestrator(
            config_path=config_path,
            stages=custom,
            log_dir=str(tmp_path / "logs"),
        )
        assert orch.stages == custom

    def test_run_executes_stages_in_order(self, orchestrator: PipelineOrchestrator):
        """Stages must appear in results in the same order they were configured."""
        orchestrator.stages = [PipelineStage.TOKENIZER, PipelineStage.DATA_PREP]

        with patch("subprocess.run", return_value=_make_completed_proc(0)):
            results = orchestrator.run()

        assert [r.stage for r in results] == [
            PipelineStage.TOKENIZER,
            PipelineStage.DATA_PREP,
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 2. skip_existing logic
# ─────────────────────────────────────────────────────────────────────────────


class TestSkipExisting:
    def test_skips_tokenizer_when_file_exists(
        self, config_path: str, tmp_path: Path
    ):
        """TOKENIZER stage is skipped when tokenizer.json already exists."""
        tokenizer_file = tmp_path / "tokenizer.json"
        tokenizer_file.write_text("{}", encoding="utf-8")

        orch = PipelineOrchestrator(
            config_path=config_path,
            stages=[PipelineStage.TOKENIZER],
            skip_existing=True,
            log_dir=str(tmp_path / "logs"),
        )

        with patch.object(orch, "_tokenizer_exists", return_value=True):
            with patch("subprocess.run") as mock_proc:
                results = orch.run()

        mock_proc.assert_not_called()
        assert results[0].skipped is True
        assert results[0].success is True

    def test_does_not_skip_when_skip_existing_is_false(
        self, orchestrator: PipelineOrchestrator
    ):
        """When skip_existing=False, the stage runs even if outputs exist."""
        orchestrator.stages = [PipelineStage.TOKENIZER]
        orchestrator.skip_existing = False

        with patch.object(orchestrator, "_tokenizer_exists", return_value=True):
            with patch("subprocess.run", return_value=_make_completed_proc(0)):
                results = orchestrator.run()

        assert results[0].skipped is False

    def test_does_not_skip_smoke_test_even_when_existing(
        self, config_path: str, tmp_path: Path
    ):
        """Smoke test and evaluation should never be skipped by skip_existing."""
        orch = PipelineOrchestrator(
            config_path=config_path,
            stages=[PipelineStage.SMOKE_TEST],
            skip_existing=True,
            log_dir=str(tmp_path / "logs"),
        )
        # _check_stage_complete for SMOKE_TEST must return False
        assert orch._check_stage_complete(PipelineStage.SMOKE_TEST) is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. dry_run mode
# ─────────────────────────────────────────────────────────────────────────────


class TestDryRun:
    def test_dry_run_does_not_call_subprocess(
        self, config_path: str, tmp_path: Path
    ):
        """In dry_run mode the orchestrator must not launch any stage subprocess.

        We assert this by patching _run_stage (the method that wraps subprocess.run)
        rather than patching subprocess.run globally.  The global subprocess.run is
        also called by torch -> platform.machine() during import which would cause
        spurious failures.
        """
        orch = PipelineOrchestrator(
            config_path=config_path,
            dry_run=True,
            log_dir=str(tmp_path / "logs"),
        )

        with patch.object(orch, "_run_stage") as mock_run_stage:
            orch.run()

        mock_run_stage.assert_not_called()

    def test_dry_run_records_all_stages(self, config_path: str, tmp_path: Path):
        """All configured stages should appear in results even in dry_run."""
        orch = PipelineOrchestrator(
            config_path=config_path,
            dry_run=True,
            log_dir=str(tmp_path / "logs"),
        )

        with patch("subprocess.run"):
            results = orch.run()

        assert len(results) == len(list(PipelineStage))

    def test_dry_run_results_are_marked_skipped(
        self, config_path: str, tmp_path: Path
    ):
        """Dry-run results are skipped=True so timing stats are excluded."""
        orch = PipelineOrchestrator(
            config_path=config_path,
            dry_run=True,
            log_dir=str(tmp_path / "logs"),
        )

        with patch("subprocess.run"):
            results = orch.run()

        assert all(r.skipped for r in results)

    def test_dry_run_message_contains_dry_run(
        self, config_path: str, tmp_path: Path
    ):
        """Stage messages in dry-run should mention 'dry-run'."""
        orch = PipelineOrchestrator(
            config_path=config_path,
            dry_run=True,
            log_dir=str(tmp_path / "logs"),
        )

        with patch("subprocess.run"):
            results = orch.run()

        assert all("dry-run" in r.message for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stage result recording
# ─────────────────────────────────────────────────────────────────────────────


class TestStageResultRecording:
    def test_successful_stage_records_success_true(
        self, orchestrator: PipelineOrchestrator
    ):
        orchestrator.stages = [PipelineStage.TOKENIZER]

        with patch("subprocess.run", return_value=_make_completed_proc(0)):
            results = orchestrator.run()

        assert results[0].success is True
        assert results[0].returncode == 0

    def test_failed_stage_records_success_false(
        self, orchestrator: PipelineOrchestrator
    ):
        orchestrator.stages = [PipelineStage.TOKENIZER]

        with patch("subprocess.run", return_value=_make_completed_proc(1)):
            results = orchestrator.run()

        assert results[0].success is False
        assert results[0].returncode == 1

    def test_failed_stage_stops_pipeline_by_default(
        self, orchestrator: PipelineOrchestrator
    ):
        """When a stage fails and continue_on_failure is False, subsequent stages don't run."""
        orchestrator.stages = [
            PipelineStage.TOKENIZER,
            PipelineStage.DATA_PREP,
            PipelineStage.TRAINING,
        ]
        orchestrator.continue_on_failure = False

        with patch("subprocess.run", return_value=_make_completed_proc(1)):
            results = orchestrator.run()

        # Only the first stage should have run (it failed, pipeline stopped)
        assert len(results) == 1
        assert results[0].stage == PipelineStage.TOKENIZER

    def test_continue_on_failure_runs_all_stages(
        self, orchestrator: PipelineOrchestrator
    ):
        """When continue_on_failure=True, the pipeline runs all stages regardless."""
        orchestrator.stages = [
            PipelineStage.TOKENIZER,
            PipelineStage.DATA_PREP,
        ]
        orchestrator.continue_on_failure = True

        with patch("subprocess.run", return_value=_make_completed_proc(1)):
            results = orchestrator.run()

        assert len(results) == 2

    def test_duration_is_recorded(self, orchestrator: PipelineOrchestrator):
        """Each executed stage should have a non-negative duration."""
        orchestrator.stages = [PipelineStage.TOKENIZER]

        with patch("subprocess.run", return_value=_make_completed_proc(0)):
            results = orchestrator.run()

        assert results[0].duration_seconds >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Report formatting
# ─────────────────────────────────────────────────────────────────────────────


class TestReportFormatting:
    def test_format_report_returns_string(self, orchestrator: PipelineOrchestrator):
        orchestrator.stages = [PipelineStage.TOKENIZER]

        with patch("subprocess.run", return_value=_make_completed_proc(0)):
            orchestrator.run()

        report = orchestrator.format_report()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_plain_report_contains_stage_names(
        self, orchestrator: PipelineOrchestrator
    ):
        """Plain-text report must include all executed stage names."""
        orchestrator.stages = [PipelineStage.TOKENIZER, PipelineStage.DATA_PREP]

        with patch("subprocess.run", return_value=_make_completed_proc(0)):
            orchestrator.run()

        report = orchestrator._format_report_plain()
        assert "tokenizer" in report
        assert "data_prep" in report

    def test_plain_report_shows_passed(self, orchestrator: PipelineOrchestrator):
        orchestrator.stages = [PipelineStage.TOKENIZER]

        with patch("subprocess.run", return_value=_make_completed_proc(0)):
            orchestrator.run()

        report = orchestrator._format_report_plain()
        assert "PASSED" in report

    def test_plain_report_shows_failed(self, orchestrator: PipelineOrchestrator):
        orchestrator.stages = [PipelineStage.TOKENIZER]
        orchestrator.continue_on_failure = True

        with patch("subprocess.run", return_value=_make_completed_proc(1)):
            orchestrator.run()

        report = orchestrator._format_report_plain()
        assert "FAILED" in report

    def test_empty_results_report(self, orchestrator: PipelineOrchestrator):
        """format_report with no results should still return a string."""
        orchestrator.stages = []
        orchestrator.run()

        report = orchestrator.format_report()
        assert isinstance(report, str)


# ─────────────────────────────────────────────────────────────────────────────
# 6. _check_stage_complete with mocked file existence
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckStageComplete:
    def test_tokenizer_complete_when_file_exists(
        self, orchestrator: PipelineOrchestrator
    ):
        with patch.object(orchestrator, "_tokenizer_exists", return_value=True):
            assert orchestrator._check_stage_complete(PipelineStage.TOKENIZER) is True

    def test_tokenizer_incomplete_when_file_missing(
        self, orchestrator: PipelineOrchestrator
    ):
        with patch.object(orchestrator, "_tokenizer_exists", return_value=False):
            assert orchestrator._check_stage_complete(PipelineStage.TOKENIZER) is False

    def test_data_prep_complete_when_data_exists(
        self, orchestrator: PipelineOrchestrator
    ):
        with patch.object(orchestrator, "_data_exists", return_value=True):
            assert orchestrator._check_stage_complete(PipelineStage.DATA_PREP) is True

    def test_training_complete_when_checkpoint_found(
        self, orchestrator: PipelineOrchestrator
    ):
        with patch.object(
            orchestrator, "_find_latest_checkpoint", return_value="checkpoints/tiny/step_10000"
        ):
            assert orchestrator._check_stage_complete(PipelineStage.TRAINING) is True

    def test_training_incomplete_when_no_checkpoint(
        self, orchestrator: PipelineOrchestrator
    ):
        with patch.object(orchestrator, "_find_latest_checkpoint", return_value=None):
            assert orchestrator._check_stage_complete(PipelineStage.TRAINING) is False

    def test_smoke_test_never_complete(self, orchestrator: PipelineOrchestrator):
        """Smoke test always returns False — it should always re-run."""
        assert orchestrator._check_stage_complete(PipelineStage.SMOKE_TEST) is False

    def test_evaluation_never_complete(self, orchestrator: PipelineOrchestrator):
        """Evaluation always returns False — results may change."""
        assert orchestrator._check_stage_complete(PipelineStage.EVALUATION) is False

    def test_export_complete_when_gguf_exists(
        self, orchestrator: PipelineOrchestrator
    ):
        with patch.object(orchestrator, "_export_exists", return_value=True):
            assert orchestrator._check_stage_complete(PipelineStage.EXPORT) is True


# ─────────────────────────────────────────────────────────────────────────────
# 7. Resume from failed stage
# ─────────────────────────────────────────────────────────────────────────────


class TestResumeFromFailedStage:
    def test_state_file_is_created_after_run(
        self, orchestrator: PipelineOrchestrator
    ):
        """A state JSON file should be written after at least one stage runs."""
        orchestrator.stages = [PipelineStage.TOKENIZER]

        with patch("subprocess.run", return_value=_make_completed_proc(0)):
            orchestrator.run()

        assert orchestrator._state_file.exists()

    def test_state_file_contains_artifacts(
        self, orchestrator: PipelineOrchestrator, tmp_path: Path
    ):
        """Artifacts are persisted in the state file so resume can find them."""
        orchestrator.stages = [PipelineStage.TOKENIZER]
        orchestrator._artifacts["tokenizer_path"] = "tokenizer.json"

        with patch("subprocess.run", return_value=_make_completed_proc(0)):
            orchestrator.run()

        with open(orchestrator._state_file, encoding="utf-8") as f:
            state = json.load(f)

        assert state["artifacts"]["tokenizer_path"] == "tokenizer.json"

    def test_artifacts_loaded_on_init(
        self, config_path: str, tmp_path: Path
    ):
        """If a state file already exists, artifacts are loaded on construction."""
        state = {"artifacts": {"checkpoint": "checkpoints/tiny/step_05000"}}

        # Build the orchestrator once so we know the state file path, then write it
        orch = PipelineOrchestrator(
            config_path=config_path,
            log_dir=str(tmp_path / "logs"),
        )
        orch._state_file.write_text(json.dumps(state), encoding="utf-8")

        # Reinitialise to trigger _load_state
        orch2 = PipelineOrchestrator(
            config_path=config_path,
            log_dir=str(tmp_path / "logs"),
        )
        assert orch2._artifacts.get("checkpoint") == "checkpoints/tiny/step_05000"

    def test_failed_stage_not_in_completed_list(
        self, orchestrator: PipelineOrchestrator
    ):
        """A failed stage should not appear in the state's stages_completed list."""
        orchestrator.stages = [PipelineStage.TOKENIZER]
        orchestrator.continue_on_failure = True

        with patch("subprocess.run", return_value=_make_completed_proc(1)):
            orchestrator.run()

        with open(orchestrator._state_file, encoding="utf-8") as f:
            state = json.load(f)

        assert "tokenizer" not in state.get("stages_completed", [])


# ─────────────────────────────────────────────────────────────────────────────
# 8. Utility helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestFmtDuration:
    def test_sub_minute(self):
        assert _fmt_duration(5.3) == "5.3s"

    def test_minutes_and_seconds(self):
        assert _fmt_duration(125.0) == "2m05s"

    def test_hours(self):
        assert _fmt_duration(3661.0) == "1h01m"

    def test_zero(self):
        assert _fmt_duration(0.0) == "0.0s"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Timeout and error handling
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_timeout_returns_failure(self, orchestrator: PipelineOrchestrator):
        """A TimeoutExpired exception should become a failed StageResult."""
        import subprocess as _subprocess

        orchestrator.stages = [PipelineStage.TOKENIZER]

        with patch(
            "subprocess.run",
            side_effect=_subprocess.TimeoutExpired(cmd=[], timeout=30),
        ):
            results = orchestrator.run()

        assert results[0].success is False
        assert "Timed out" in results[0].message

    def test_file_not_found_returns_failure(self, orchestrator: PipelineOrchestrator):
        """A missing Python executable should become a failed StageResult."""
        orchestrator.stages = [PipelineStage.TOKENIZER]

        with patch("subprocess.run", side_effect=FileNotFoundError("python not found")):
            results = orchestrator.run()

        assert results[0].success is False
        assert "not found" in results[0].message


# ─────────────────────────────────────────────────────────────────────────────
# 10. StageResult dataclass
# ─────────────────────────────────────────────────────────────────────────────


class TestStageResultDataclass:
    def test_default_artifacts_is_empty_dict(self):
        r = StageResult(
            stage=PipelineStage.TOKENIZER,
            success=True,
            message="OK",
            duration_seconds=1.0,
        )
        assert r.artifacts == {}

    def test_artifacts_not_shared_between_instances(self):
        """Each StageResult should have its own artifacts dict."""
        r1 = StageResult(
            stage=PipelineStage.TOKENIZER, success=True, message="", duration_seconds=0.0
        )
        r2 = StageResult(
            stage=PipelineStage.DATA_PREP, success=True, message="", duration_seconds=0.0
        )
        r1.artifacts["key"] = "value"
        assert "key" not in r2.artifacts

    def test_skipped_false_by_default(self):
        r = StageResult(
            stage=PipelineStage.TRAINING, success=True, message="OK", duration_seconds=5.0
        )
        assert r.skipped is False


# ─────────────────────────────────────────────────────────────────────────────
# 11. PipelineStage enum
# ─────────────────────────────────────────────────────────────────────────────


class TestPipelineStageEnum:
    def test_all_stages_have_string_values(self):
        for stage in PipelineStage:
            assert isinstance(stage.value, str)
            assert len(stage.value) > 0

    def test_stage_count(self):
        """Exactly 6 stages must be defined."""
        assert len(list(PipelineStage)) == 6

    def test_stage_lookup_by_value(self):
        assert PipelineStage("tokenizer") is PipelineStage.TOKENIZER
        assert PipelineStage("export") is PipelineStage.EXPORT
