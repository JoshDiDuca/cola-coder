"""Tests for the safety probe prompt suites."""

from cola_coder.evaluation.safety_probes import (
    BASIC_PROMPTS,
    CWE_PROMPTS,
    EXTENDED_EXTRA,
    INJECTION_PROMPTS,
    LICENSE_PROMPTS,
    PII_PROMPTS,
    SUITES,
)


class TestSuiteStructure:
    def test_all_expected_suites_exist(self):
        assert set(SUITES) == {"basic", "extended", "pii", "license",
                               "injection", "cwe", "all"}

    def test_minimum_suite_sizes(self):
        # Floors, not exact counts — suites should only ever grow
        assert len(SUITES["basic"]) >= 25
        assert len(SUITES["extended"]) >= 65
        assert len(SUITES["pii"]) >= 24
        assert len(SUITES["license"]) >= 18
        assert len(SUITES["injection"]) >= 16
        assert len(SUITES["all"]) >= 123

    def test_extended_is_superset_of_basic(self):
        assert SUITES["extended"][: len(BASIC_PROMPTS)] == BASIC_PROMPTS
        assert len(SUITES["extended"]) == len(BASIC_PROMPTS) + len(EXTENDED_EXTRA)

    def test_all_contains_every_suite(self):
        combined = set(SUITES["all"])
        for suite in (BASIC_PROMPTS, EXTENDED_EXTRA, PII_PROMPTS,
                      LICENSE_PROMPTS, INJECTION_PROMPTS, CWE_PROMPTS):
            assert set(suite) <= combined


class TestProbeQuality:
    def test_no_duplicate_probes_within_all(self):
        probes = SUITES["all"]
        assert len(probes) == len(set(probes)), "duplicate probe prompts"

    def test_probes_are_nonempty_strings(self):
        for name, probes in SUITES.items():
            for probe in probes:
                assert isinstance(probe, str) and probe.strip(), (
                    f"empty probe in suite {name!r}"
                )

    def test_probes_are_short_prefixes(self):
        # Probes are prefixes, not full programs — keep generation fast
        for probe in SUITES["all"]:
            assert len(probe) < 400

    def test_injection_probes_embed_instructions_in_code_context(self):
        # Every injection probe must pair an instruction with real code
        # context (comment/docstring + a def/function/const to complete)
        for probe in INJECTION_PROMPTS:
            has_code = any(
                marker in probe
                for marker in ("def ", "function ", "const ", " = ")
            )
            assert has_code, f"injection probe lacks code context: {probe!r}"


class TestScriptIntegration:
    def test_script_imports_library_suites(self):
        text = (
            __import__("pathlib").Path(__file__).parent.parent
            / "scripts" / "safety_eval.py"
        ).read_text(encoding="utf-8")
        assert "from cola_coder.evaluation.safety_probes import SUITES" in text
        # The old inline prompt lists must be gone
        assert "_BASIC_PROMPTS = [" not in text
