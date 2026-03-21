"""Unit tests for the smoke test module.

Tests the SmokeTest class using a configurable MockGenerator — no GPU or
trained checkpoint required.  The mock generator returns a fixed string so
each test case exercises one specific pass/fail path in the smoke test logic.
"""

from __future__ import annotations

from cola_coder.evaluation.smoke_test import (
    SmokeTest,
    SmokeTestReport,
    TestResult,
    _PROMPTS,
)


# ── Mock helpers ──────────────────────────────────────────────────────────────


class MockGenerator:
    """A fake generator that returns configurable strings.

    ``outputs`` can be:
    - A single string → returned for every prompt/call.
    - A list of strings → cycled through in order (wraps around).
    - A callable(prompt, **kwargs) → called each time, must return str.
    """

    def __init__(self, outputs: str | list[str] | None = None):
        if outputs is None:
            outputs = "def hello():\n    return 42\n"
        self._outputs = outputs
        self._call_count = 0

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: list[str] | None = None,
    ) -> str:
        if callable(self._outputs):
            result = self._outputs(prompt, temperature=temperature)
        elif isinstance(self._outputs, list):
            result = self._outputs[self._call_count % len(self._outputs)]
        else:
            result = self._outputs
        self._call_count += 1
        return result


class ExplodingGenerator:
    """A generator that always raises an exception."""

    def generate(self, prompt: str, **kwargs) -> str:
        raise RuntimeError("Simulated generator failure")


# ── Tests: TestResult and SmokeTestReport dataclasses ────────────────────────


class TestDataclasses:
    def test_test_result_fields(self):
        """TestResult stores name, passed, message, duration_ms."""
        r = TestResult(name="test_foo", passed=True, message="ok", duration_ms=12.3)
        assert r.name == "test_foo"
        assert r.passed is True
        assert r.message == "ok"
        assert r.duration_ms == 12.3

    def test_report_passed_all_pass(self):
        """SmokeTestReport.passed is True when all results pass."""
        results = [
            TestResult("a", True, "ok", 1.0),
            TestResult("b", True, "ok", 2.0),
        ]
        report = SmokeTestReport(results=results, total_duration_ms=10.0)
        assert report.passed is True

    def test_report_passed_one_fail(self):
        """SmokeTestReport.passed is False if any result fails."""
        results = [
            TestResult("a", True, "ok", 1.0),
            TestResult("b", False, "nope", 2.0),
        ]
        report = SmokeTestReport(results=results, total_duration_ms=10.0)
        assert report.passed is False

    def test_report_summary_contains_counts(self):
        """Summary string mentions pass/fail counts."""
        results = [
            TestResult("a", True, "ok", 1.0),
            TestResult("b", False, "fail", 2.0),
        ]
        report = SmokeTestReport(results=results, total_duration_ms=50.0)
        assert "1" in report.summary  # 1 passed
        assert "2" in report.summary  # 2 total

    def test_report_num_passed_and_failed(self):
        """num_passed and num_failed are computed correctly."""
        results = [
            TestResult("a", True, "ok", 1.0),
            TestResult("b", True, "ok", 1.0),
            TestResult("c", False, "fail", 1.0),
        ]
        report = SmokeTestReport(results=results, total_duration_ms=10.0)
        assert report.num_passed == 2
        assert report.num_failed == 1


# ── Tests: individual test methods ───────────────────────────────────────────


class TestGeneratesTokens:
    def test_normal_output_passes(self):
        gen = MockGenerator("def hello():\n    pass\n")
        smoke = SmokeTest(gen)
        result = smoke.test_generates_tokens()
        assert result.passed, result.message

    def test_empty_output_fails(self):
        # Return only the prompt — no new tokens
        gen = MockGenerator(_PROMPTS[0])
        smoke = SmokeTest(gen)
        result = smoke.test_generates_tokens()
        assert not result.passed

    def test_exception_fails_gracefully(self):
        smoke = SmokeTest(ExplodingGenerator())
        result = smoke.test_generates_tokens()
        assert not result.passed
        assert "Exception" in result.message


class TestCodeSyntax:
    def test_valid_python_passes(self):
        gen = MockGenerator("def hello():\n    return 42\n")
        smoke = SmokeTest(gen)
        result = smoke.test_code_syntax()
        assert result.passed, result.message

    def test_truncated_output_is_lenient(self):
        # Truncated last line — should still pass due to leniency
        gen = MockGenerator("def hello():\n    retur")
        smoke = SmokeTest(gen)
        result = smoke.test_code_syntax()
        # Lenient path: truncated output counts as acceptable
        assert result.passed, result.message


class TestRepetition:
    def test_normal_output_passes(self):
        gen = MockGenerator("def hello():\n    x = 1\n    y = 2\n    return x + y\n")
        smoke = SmokeTest(gen)
        result = smoke.test_repetition()
        assert result.passed, result.message

    def test_repetitive_output_fails(self):
        # Same single word repeated many times
        repeated = "spam " * 50
        gen = MockGenerator(repeated)
        smoke = SmokeTest(gen)
        result = smoke.test_repetition()
        assert not result.passed, "Repetitive output should fail"
        assert "spam" in result.message.lower() or "repetiti" in result.message.lower()

    def test_exception_fails_gracefully(self):
        smoke = SmokeTest(ExplodingGenerator())
        result = smoke.test_repetition()
        assert not result.passed


class TestDiversity:
    def test_different_prompts_produce_different_outputs(self):
        # Cycle through distinct strings so different prompts get different results
        gen = MockGenerator([
            "def hello():\n    pass\n",
            "function add(a, b) { return a + b; }",
            "class User:\n    pass\n",
        ])
        smoke = SmokeTest(gen)
        result = smoke.test_diversity()
        assert result.passed, result.message

    def test_identical_outputs_fail(self):
        gen = MockGenerator("same output every time")
        smoke = SmokeTest(gen)
        result = smoke.test_diversity()
        assert not result.passed


class TestSpecialTokens:
    def test_clean_output_passes(self):
        gen = MockGenerator("def hello():\n    return 'world'\n")
        smoke = SmokeTest(gen)
        result = smoke.test_special_tokens()
        assert result.passed, result.message

    def test_raw_special_token_fails(self):
        gen = MockGenerator("def hello():\n    <|endoftext|> return 42\n")
        smoke = SmokeTest(gen)
        result = smoke.test_special_tokens()
        assert not result.passed
        assert "endoftext" in result.message


class TestCodeKeywords:
    def test_code_output_passes(self):
        gen = MockGenerator("def hello():\n    return 42\n")
        smoke = SmokeTest(gen)
        result = smoke.test_code_keywords()
        assert result.passed, result.message

    def test_non_code_output_fails(self):
        # Prose with no programming keywords
        gen = MockGenerator("The quick brown fox jumped over the lazy dog")
        smoke = SmokeTest(gen)
        result = smoke.test_code_keywords()
        assert not result.passed


class TestTemperatureSensitivity:
    def test_temperature_sensitivity_with_varying_outputs(self):
        """High-temperature calls produce at least as many unique outputs."""
        call_count = [0]

        def _outputs(prompt: str, temperature: float = 0.8) -> str:
            call_count[0] += 1
            if temperature >= 1.0:
                # High temp → unique output each time
                return f"def unique_{call_count[0]}():\n    pass\n"
            else:
                # Low temp → same output every time
                return "def deterministic():\n    pass\n"

        gen = MockGenerator(_outputs)
        smoke = SmokeTest(gen)
        result = smoke.test_temperature_sensitivity()
        assert result.passed, result.message

    def test_exception_fails_gracefully(self):
        smoke = SmokeTest(ExplodingGenerator())
        result = smoke.test_temperature_sensitivity()
        assert not result.passed


# ── Tests: run_all aggregation ────────────────────────────────────────────────


class TestRunAll:
    def test_run_all_returns_report(self):
        """run_all() returns a SmokeTestReport with results for every test."""
        gen = MockGenerator("def hello():\n    return 42\n")
        smoke = SmokeTest(gen)
        report = smoke.run_all()
        assert isinstance(report, SmokeTestReport)
        assert len(report.results) == 8  # 8 test methods
        assert report.total_duration_ms > 0

    def test_run_all_aggregates_pass_fail(self):
        """run_all passed property reflects individual results."""
        gen = MockGenerator("def hello():\n    return 42\n")
        smoke = SmokeTest(gen)
        report = smoke.run_all()
        # All individual results determine overall pass
        individual_passed = all(r.passed for r in report.results)
        assert report.passed == individual_passed

    def test_run_all_with_good_generator_mostly_passes(self):
        """A well-formed code generator passes most smoke tests."""
        good_outputs = [
            "def hello():\n    return 42\n",
            "function add(a, b) {\n    return a + b;\n}\n",
            "class User:\n    def __init__(self):\n        pass\n",
            "import os\nimport sys\n",
            "// Calculate the sum\nfunction sum(a, b) {\n    return a + b;\n}\n",
        ]
        gen = MockGenerator(good_outputs)
        smoke = SmokeTest(gen)
        report = smoke.run_all()
        # At least 5 of 8 tests should pass with well-formed output
        assert report.num_passed >= 5, (
            f"Expected >=5 passed, got {report.num_passed}. "
            f"Failures: {[(r.name, r.message) for r in report.results if not r.passed]}"
        )

    def test_run_all_no_tokenizer_skips_perplexity(self):
        """Without a tokenizer, perplexity test is skipped (passes with a note)."""
        gen = MockGenerator("def hello():\n    return 42\n")
        smoke = SmokeTest(gen, tokenizer=None)
        report = smoke.run_all()
        ppl_result = next((r for r in report.results if r.name == "test_perplexity_range"), None)
        assert ppl_result is not None
        assert ppl_result.passed  # skipped → True
        assert "skip" in ppl_result.message.lower()

    def test_run_all_records_durations(self):
        """Every test result records a non-negative duration."""
        gen = MockGenerator("def x():\n    pass\n")
        smoke = SmokeTest(gen)
        report = smoke.run_all()
        for r in report.results:
            assert r.duration_ms >= 0, f"{r.name} has negative duration"
