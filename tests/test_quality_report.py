"""Tests for quality_report.py and model_comparison.py.

All model loading and generation is mocked — no GPU required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cola_coder.evaluation.quality_report import (
    STANDARD_PROMPTS,
    QualityReport,
    QualityReportGenerator,
    _human_params,
)
from cola_coder.evaluation.model_comparison import (
    ComparisonResult,
    ModelComparator,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_smoke_details(n: int = 8, all_pass: bool = True) -> list[dict]:
    return [
        {"name": f"test_{i}", "passed": all_pass, "message": "ok", "duration_ms": 10.0}
        for i in range(n)
    ]


def _make_report(**kwargs) -> QualityReport:
    """Build a minimal QualityReport with sensible defaults."""
    defaults: dict = {
        "checkpoint_path": "checkpoints/tiny/step_00020000",
        "config": {"training": {"max_steps": 100_000}},
        "timestamp": "2026-03-21 12:00:00",
        "model_params": 40_000_000,
        "model_config": {"n_layers": 8, "dim": 512, "n_heads": 8},
        "training_step": 20_000,
        "training_loss": 2.31,
        "smoke_test_passed": True,
        "smoke_test_details": _make_smoke_details(),
        "samples": [
            {
                "prompt": "def fibonacci(n: int) -> int:",
                "output": "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                "tokens": 32,
                "time_ms": 120.0,
            }
        ],
        "humaneval_pass_at_1": None,
    }
    defaults.update(kwargs)
    return QualityReport(**defaults)


# ── Tests: QualityReport.to_markdown ─────────────────────────────────────────


class TestQualityReportToMarkdown:
    def test_markdown_has_title(self) -> None:
        report = _make_report()
        md = report.to_markdown()
        assert "# Cola-Coder Quality Report" in md

    def test_markdown_has_checkpoint_path(self) -> None:
        report = _make_report(checkpoint_path="checkpoints/tiny/step_00020000")
        md = report.to_markdown()
        assert "checkpoints/tiny/step_00020000" in md

    def test_markdown_has_date(self) -> None:
        report = _make_report(timestamp="2026-03-21 12:00:00")
        md = report.to_markdown()
        assert "2026-03-21" in md

    def test_markdown_has_model_info(self) -> None:
        report = _make_report(
            model_params=40_000_000,
            model_config={"n_layers": 8, "dim": 512, "n_heads": 8},
        )
        md = report.to_markdown()
        assert "## Model Info" in md
        assert "40M" in md
        assert "8 layers" in md

    def test_markdown_has_training_status(self) -> None:
        report = _make_report(training_step=20_000, training_loss=2.31)
        md = report.to_markdown()
        assert "## Training Status" in md
        assert "20,000" in md
        assert "2.3100" in md

    def test_markdown_has_smoke_test_section(self) -> None:
        report = _make_report(smoke_test_passed=True, smoke_test_details=_make_smoke_details(8))
        md = report.to_markdown()
        assert "## Smoke Test:" in md
        assert "PASSED" in md
        assert "8/8" in md

    def test_markdown_smoke_test_failed(self) -> None:
        details = _make_smoke_details(8, all_pass=False)
        report = _make_report(smoke_test_passed=False, smoke_test_details=details)
        md = report.to_markdown()
        assert "FAILED" in md

    def test_markdown_has_sample_outputs(self) -> None:
        report = _make_report()
        md = report.to_markdown()
        assert "## Sample Outputs" in md
        assert "fibonacci" in md

    def test_markdown_smoke_table_has_columns(self) -> None:
        report = _make_report()
        md = report.to_markdown()
        assert "| Test |" in md
        assert "| Result |" in md
        assert "| Time |" in md

    def test_markdown_humaneval_section_when_present(self) -> None:
        report = _make_report(humaneval_pass_at_1=0.123)
        md = report.to_markdown()
        assert "## Evaluation" in md
        assert "12.3%" in md

    def test_markdown_no_humaneval_section_when_none(self) -> None:
        report = _make_report(humaneval_pass_at_1=None)
        md = report.to_markdown()
        assert "HumanEval" not in md


# ── Tests: QualityReport.to_dict / JSON serialization ────────────────────────


class TestQualityReportToDict:
    def test_to_dict_is_json_serializable(self) -> None:
        report = _make_report()
        d = report.to_dict()
        # Should not raise
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_to_dict_contains_key_fields(self) -> None:
        report = _make_report(training_step=20_000, training_loss=2.31)
        d = report.to_dict()
        assert "checkpoint_path" in d
        assert "training_step" in d
        assert "training_loss" in d
        assert "smoke_test_passed" in d
        assert "samples" in d

    def test_to_dict_nan_loss_becomes_none(self) -> None:
        report = _make_report(training_loss=float("nan"))
        d = report.to_dict()
        # NaN is not JSON-serializable; to_dict should replace with None
        assert d["training_loss"] is None
        # Verify it's now JSON-serializable
        json.dumps(d)  # should not raise

    def test_to_dict_round_trips_samples(self) -> None:
        samples = [{"prompt": "def f():", "output": "def f(): pass", "tokens": 5, "time_ms": 50.0}]
        report = _make_report(samples=samples)
        d = report.to_dict()
        assert d["samples"] == samples

    def test_to_dict_humaneval_none_serializable(self) -> None:
        report = _make_report(humaneval_pass_at_1=None)
        d = report.to_dict()
        json.dumps(d)  # should not raise


# ── Tests: QualityReportGenerator ────────────────────────────────────────────


class TestQualityReportGenerator:
    """Tests for QualityReportGenerator.generate() with all model loading mocked."""

    def _build_mocks(self) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
        """Return (mock_config, mock_transformer, mock_generator, mock_tokenizer, mock_smoke)."""
        # Model config mock with required dataclass field
        mock_model_cfg = MagicMock()
        mock_model_cfg.__dataclass_fields__ = {}

        mock_config = MagicMock()
        mock_config.model = mock_model_cfg

        mock_transformer = MagicMock()
        mock_transformer.to.return_value = mock_transformer
        mock_transformer.eval.return_value = None
        mock_transformer.parameters.return_value = iter(
            [MagicMock(**{"numel.return_value": 1_000_000}) for _ in range(40)]
        )

        mock_gen = MagicMock()
        mock_gen.generate.return_value = "def foo(): pass"
        mock_gen.model = mock_transformer
        mock_gen.device = "cpu"

        mock_tok = MagicMock()
        mock_tok.encode.return_value = [1, 2, 3, 4, 5]
        mock_tok.eos_id = 2

        mock_smoke_result = MagicMock()
        mock_smoke_result.passed = True
        mock_smoke_result.results = [
            MagicMock(name=f"test_{i}", passed=True, message="ok", duration_ms=10.0)
            for i in range(8)
        ]

        return mock_config, mock_transformer, mock_gen, mock_tok, mock_smoke_result

    def _run_generate(self, tmp_path: Path, **kwargs) -> QualityReport:
        """Run QualityReportGenerator.generate() with all model loading mocked.

        Patches the local import targets that generate() uses inside the function body.
        """
        config_yaml = tmp_path / "tiny.yaml"
        config_yaml.write_text("model:\n  n_layers: 4\n  dim: 64\n  n_heads: 4\n", encoding="utf-8")
        ckpt_path = str(tmp_path)

        mock_config, mock_transformer, mock_gen, mock_tok, mock_smoke_result = self._build_mocks()

        with patch("cola_coder.model.config.Config") as mock_Config_cls, \
             patch("cola_coder.model.config.ModelConfig") as mock_ModelConfig_cls, \
             patch("cola_coder.model.transformer.Transformer") as mock_T_cls, \
             patch("cola_coder.training.checkpoint.load_model_only"), \
             patch("cola_coder.inference.generator.CodeGenerator") as mock_CG_cls, \
             patch("cola_coder.tokenizer.tokenizer_utils.CodeTokenizer") as mock_CT_cls, \
             patch("cola_coder.evaluation.smoke_test.SmokeTest") as mock_ST_cls, \
             patch.object(QualityReportGenerator, "_find_tokenizer", return_value=str(tmp_path / "tok.json")):

            mock_Config_cls.from_yaml.return_value = mock_config
            mock_ModelConfig_cls.__dataclass_fields__ = {}
            mock_T_cls.return_value = mock_transformer
            mock_CG_cls.return_value = mock_gen
            mock_CT_cls.return_value = mock_tok
            mock_ST_cls.return_value.run_all.return_value = mock_smoke_result

            gen = QualityReportGenerator(
                checkpoint_path=ckpt_path,
                config_path=str(config_yaml),
                device="cpu",
            )
            return gen.generate(**kwargs)

    def test_generator_uses_standard_prompts_when_default(self, tmp_path: Path) -> None:
        """When num_samples=5 (default), all 5 STANDARD_PROMPTS are used."""
        report = self._run_generate(tmp_path, num_samples=5)
        assert len(report.samples) == 5
        for sample, expected_prompt in zip(report.samples, STANDARD_PROMPTS[:5]):
            assert sample["prompt"] == expected_prompt

    def test_generator_limits_samples_to_num_samples(self, tmp_path: Path) -> None:
        """num_samples=2 generates exactly 2 samples."""
        report = self._run_generate(tmp_path, num_samples=2)
        assert len(report.samples) == 2

    def test_generator_reads_step_from_metadata(self, tmp_path: Path) -> None:
        (tmp_path / "metadata.json").write_text(
            json.dumps({"step": 15000, "loss": 2.5}), encoding="utf-8"
        )
        report = self._run_generate(tmp_path, num_samples=1)
        assert report.training_step == 15000
        assert abs(report.training_loss - 2.5) < 1e-6

    def test_generator_smoke_test_details_in_report(self, tmp_path: Path) -> None:
        report = self._run_generate(tmp_path, num_samples=1)
        assert isinstance(report.smoke_test_details, list)
        assert len(report.smoke_test_details) > 0
        for detail in report.smoke_test_details:
            assert "name" in detail
            assert "passed" in detail


# ── Tests: save_report ────────────────────────────────────────────────────────


class TestSaveReport:
    def test_save_creates_markdown_file(self, tmp_path: Path) -> None:
        report = _make_report()
        gen = QualityReportGenerator("fake_ckpt", "fake.yaml", device="cpu")
        out_dir = tmp_path / "reports"
        gen.save_report(report, output_dir=str(out_dir))
        md_files = list(out_dir.glob("*.md"))
        assert len(md_files) == 1

    def test_save_creates_json_file(self, tmp_path: Path) -> None:
        report = _make_report()
        gen = QualityReportGenerator("fake_ckpt", "fake.yaml", device="cpu")
        out_dir = tmp_path / "reports"
        gen.save_report(report, output_dir=str(out_dir))
        json_files = list(out_dir.glob("*.json"))
        assert len(json_files) == 1

    def test_save_creates_output_dir_automatically(self, tmp_path: Path) -> None:
        report = _make_report()
        gen = QualityReportGenerator("fake_ckpt", "fake.yaml", device="cpu")
        out_dir = tmp_path / "nested" / "reports"
        assert not out_dir.exists()
        gen.save_report(report, output_dir=str(out_dir))
        assert out_dir.exists()

    def test_saved_json_is_valid(self, tmp_path: Path) -> None:
        report = _make_report(training_step=5_000, training_loss=3.14)
        gen = QualityReportGenerator("fake_ckpt", "fake.yaml", device="cpu")
        out_dir = tmp_path / "reports"
        gen.save_report(report, output_dir=str(out_dir))
        json_files = list(out_dir.glob("*.json"))
        data = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert data["training_step"] == 5_000
        assert abs(data["training_loss"] - 3.14) < 1e-6


# ── Tests: ComparisonResult.to_markdown ──────────────────────────────────────


class TestComparisonResultToMarkdown:
    def _make_result(self) -> ComparisonResult:
        return ComparisonResult(
            models=[
                {"name": "step_18000", "checkpoint": "ckpt/step_18000", "params": 40_000_000,
                 "params_human": "40M", "step": 18000, "loss": 2.5},
                {"name": "step_20000", "checkpoint": "ckpt/step_20000", "params": 40_000_000,
                 "params_human": "40M", "step": 20000, "loss": 2.3},
            ],
            prompts=["def fibonacci(n):", "class Stack:"],
            outputs=[
                ["def fibonacci(n): pass", "class Stack: pass"],
                ["def fibonacci(n): return n", "class Stack: ..."],
            ],
            metrics=[
                {"tokens_per_sec": 45.0, "avg_output_len": 30.0},
                {"tokens_per_sec": 50.0, "avg_output_len": 28.0},
            ],
        )

    def test_comparison_markdown_has_title(self) -> None:
        result = self._make_result()
        md = result.to_markdown()
        assert "# Cola-Coder Model Comparison" in md

    def test_comparison_markdown_has_models_section(self) -> None:
        result = self._make_result()
        md = result.to_markdown()
        assert "## Models" in md
        assert "step_18000" in md
        assert "step_20000" in md

    def test_comparison_markdown_has_outputs_section(self) -> None:
        result = self._make_result()
        md = result.to_markdown()
        assert "## Outputs" in md
        assert "fibonacci" in md

    def test_comparison_markdown_has_metrics_section(self) -> None:
        result = self._make_result()
        md = result.to_markdown()
        assert "## Performance Metrics" in md


# ── Tests: ModelComparator ────────────────────────────────────────────────────


class TestModelComparator:
    def test_raises_on_empty_checkpoints(self) -> None:
        with pytest.raises(ValueError, match="at least one checkpoint"):
            ModelComparator(checkpoints=[], configs=["cfg.yaml"])

    def test_raises_on_mismatched_configs(self) -> None:
        with pytest.raises(ValueError, match="configs must have length"):
            ModelComparator(
                checkpoints=["ckpt_a", "ckpt_b"],
                configs=["cfg_a.yaml", "cfg_b.yaml", "cfg_c.yaml"],
            )

    def test_single_config_reused_for_all_checkpoints(self) -> None:
        comp = ModelComparator(
            checkpoints=["ckpt_a", "ckpt_b"],
            configs=["shared.yaml"],
        )
        assert comp.configs == ["shared.yaml", "shared.yaml"]

    @patch("cola_coder.evaluation.model_comparison.ModelComparator._load_generator")
    def test_compare_uses_default_prompts_when_none(
        self, mock_load: MagicMock, tmp_path: Path
    ) -> None:
        """compare() with prompts=None uses DEFAULT_COMPARISON_PROMPTS."""
        from cola_coder.evaluation.model_comparison import DEFAULT_COMPARISON_PROMPTS

        mock_gen = MagicMock()
        mock_gen.generate.return_value = "def foo(): pass"
        mock_gen.model = MagicMock()
        mock_gen.model.parameters.return_value = iter([
            MagicMock(**{"numel.return_value": 1_000}) for _ in range(5)
        ])
        mock_tok = MagicMock()
        mock_tok.encode.return_value = [1, 2, 3]
        mock_load.return_value = (mock_gen, mock_tok)

        ckpt = str(tmp_path / "step_00010000")
        Path(ckpt).mkdir()
        (Path(ckpt) / "metadata.json").write_text(
            json.dumps({"step": 10000, "loss": 2.8}), encoding="utf-8"
        )

        comp = ModelComparator(checkpoints=[ckpt], configs=["dummy.yaml"])
        result = comp.compare(prompts=None)
        assert result.prompts == DEFAULT_COMPARISON_PROMPTS
        assert len(result.outputs[0]) == len(DEFAULT_COMPARISON_PROMPTS)

    @patch("cola_coder.evaluation.model_comparison.ModelComparator._load_generator")
    def test_compare_quick_uses_3_prompts(
        self, mock_load: MagicMock, tmp_path: Path
    ) -> None:
        mock_gen = MagicMock()
        mock_gen.generate.return_value = "result"
        mock_gen.model = MagicMock()
        mock_gen.model.parameters.return_value = iter([])
        mock_tok = MagicMock()
        mock_tok.encode.return_value = [1, 2]
        mock_load.return_value = (mock_gen, mock_tok)

        ckpt = str(tmp_path / "step_00005000")
        Path(ckpt).mkdir()

        comp = ModelComparator(checkpoints=[ckpt], configs=["dummy.yaml"])
        result = comp.compare_quick()
        assert len(result.prompts) == 3

    @patch("cola_coder.evaluation.model_comparison.ModelComparator._load_generator")
    def test_compare_multiple_models(
        self, mock_load: MagicMock, tmp_path: Path
    ) -> None:
        """Compares 2 models and returns outputs for each."""
        call_count = 0

        def side_effect(ckpt, cfg):
            nonlocal call_count
            call_count += 1
            gen = MagicMock()
            gen.generate.return_value = f"output_from_model_{call_count}"
            gen.model = MagicMock()
            gen.model.parameters.return_value = iter([])
            tok = MagicMock()
            tok.encode.return_value = [1, 2, 3]
            return gen, tok

        mock_load.side_effect = side_effect

        ckpt_a = str(tmp_path / "step_18000")
        ckpt_b = str(tmp_path / "step_20000")
        Path(ckpt_a).mkdir()
        Path(ckpt_b).mkdir()
        (Path(ckpt_a) / "metadata.json").write_text(json.dumps({"step": 18000, "loss": 2.5}))
        (Path(ckpt_b) / "metadata.json").write_text(json.dumps({"step": 20000, "loss": 2.3}))

        comp = ModelComparator(
            checkpoints=[ckpt_a, ckpt_b],
            configs=["cfg.yaml"],
        )
        result = comp.compare(prompts=["def foo():"], temperature=0.8, max_tokens=32)

        assert len(result.models) == 2
        assert len(result.outputs) == 2
        assert result.outputs[0][0] == "output_from_model_1"
        assert result.outputs[1][0] == "output_from_model_2"


# ── Tests: helper functions ───────────────────────────────────────────────────


class TestHelperFunctions:
    def test_human_params_billions(self) -> None:
        assert _human_params(1_000_000_000) == "1.0B"
        assert _human_params(1_300_000_000) == "1.3B"

    def test_human_params_millions(self) -> None:
        assert _human_params(40_000_000) == "40M"
        assert _human_params(125_000_000) == "125M"

    def test_human_params_thousands(self) -> None:
        assert _human_params(50_000) == "50K"

    def test_human_params_small(self) -> None:
        assert _human_params(100) == "100"

    def test_standard_prompts_are_non_empty(self) -> None:
        assert len(STANDARD_PROMPTS) >= 5
        for prompt in STANDARD_PROMPTS:
            assert isinstance(prompt, str)
            assert len(prompt) > 0
