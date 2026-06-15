"""Tests for the SEC-027 ``cwe`` safety probe suite.

The cwe suite measures the CWE rate in *generated* code by statically scanning
each completion with the existing SEC-026 ``CweSecurityScorer`` — no model, no
GPU, no code execution. These tests feed fixed code strings to the probe and
assert the pass/fail verdict + structured findings, and that the probe reuses
the scorer rather than reimplementing CWE detection.
"""

from __future__ import annotations

import inspect

import cola_coder.evaluation.safety_probes as safety_probes
from cola_coder.data.scorers.cwe_security import CweSecurityScorer
from cola_coder.evaluation.safety_probes import (
    CWE_PROMPTS,
    SUITES,
    CweProbeResult,
    cwe_probe_result,
)


class TestCweSuiteRegistration:
    def test_cwe_is_a_selectable_suite(self):
        assert "cwe" in SUITES
        assert SUITES["cwe"] is CWE_PROMPTS

    def test_cwe_prompts_are_nonempty_short_prefixes(self):
        assert CWE_PROMPTS, "cwe suite must have probes"
        for probe in CWE_PROMPTS:
            assert isinstance(probe, str) and probe.strip()
            assert len(probe) < 400

    def test_cwe_prompts_folded_into_all(self):
        assert set(CWE_PROMPTS) <= set(SUITES["all"])

    def test_script_exposes_cwe_via_sorted_suites(self):
        # safety_eval.py builds --suite choices from sorted(SUITES); "cwe" being
        # a SUITES key is what makes `--suite cwe` selectable. Assert the script
        # imports the suites and the cwe probe helper (no orphan).
        from pathlib import Path

        text = (
            Path(__file__).parent.parent / "scripts" / "safety_eval.py"
        ).read_text(encoding="utf-8")
        assert "cwe_probe_result" in text
        assert "from cola_coder.evaluation.safety_probes import" in text


class TestCweProbeVerdict:
    def test_os_system_completion_fails(self):
        # CWE-78 — os.system shell execution
        code = "import os\ndef ping(host):\n    os.system('ping ' + host)\n"
        result = cwe_probe_result(code)
        assert isinstance(result, CweProbeResult)
        assert result.passed is False
        assert "CWE-78" in result.failed_families
        assert result.score < 1.0
        assert result.findings  # structured findings present

    def test_pickle_loads_completion_fails(self):
        # CWE-502 — unsafe deserialization
        code = "import pickle\ndef load(blob):\n    return pickle.loads(blob)\n"
        result = cwe_probe_result(code)
        assert result.passed is False
        assert "CWE-502" in result.failed_families

    def test_clean_completion_passes(self):
        code = (
            "def add(a, b):\n"
            "    \"\"\"Return the sum of two numbers.\"\"\"\n"
            "    return a + b\n"
        )
        result = cwe_probe_result(code)
        assert result.passed is True
        assert result.failed_families == []
        assert result.findings == []
        assert result.score == 1.0

    def test_multiple_families_reported_distinctly(self):
        code = (
            "import os, pickle\n"
            "def f(x, blob):\n"
            "    os.system(x)\n"
            "    return pickle.loads(blob)\n"
        )
        result = cwe_probe_result(code)
        assert result.passed is False
        assert "CWE-78" in result.failed_families
        assert "CWE-502" in result.failed_families

    def test_findings_have_expected_shape(self):
        code = "import os\nos.system(user_input)\n"
        result = cwe_probe_result(code)
        assert result.findings
        finding = result.findings[0]
        # Same structured shape the scorer emits (cwe/name/severity/line/snippet)
        for key in ("cwe", "name", "severity", "line", "snippet"):
            assert key in finding


class TestCweProbeReusesScorer:
    """The probe must call CweSecurityScorer, not reimplement CWE regexes."""

    def test_probe_source_constructs_the_scorer(self):
        src = inspect.getsource(safety_probes)
        assert "CweSecurityScorer" in src
        assert isinstance(safety_probes._CWE_SCORER, CweSecurityScorer)

    def test_probe_does_not_define_its_own_cwe_patterns(self):
        # No parallel CWE regex table — the verdict comes from the scorer.
        src = inspect.getsource(safety_probes)
        assert "_PATTERNS" not in src
        assert "CwePattern" not in src

    def test_probe_delegates_to_scorer(self, monkeypatch):
        # Patch the shared scorer's score() and confirm the probe routes through it.
        calls: list[str] = []
        real_score = safety_probes._CWE_SCORER.score

        def spy(code, metadata=None):
            calls.append(code)
            return real_score(code, metadata)

        monkeypatch.setattr(safety_probes._CWE_SCORER, "score", spy)
        cwe_probe_result("import os\nos.system(x)\n")
        assert calls, "cwe_probe_result did not call CweSecurityScorer.score"
