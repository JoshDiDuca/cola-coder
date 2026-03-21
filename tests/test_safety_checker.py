"""Tests for SafetyChecker (features/safety_checker.py)."""

from __future__ import annotations

import pytest

from cola_coder.features.safety_checker import (
    SafetyChecker,
    Severity,
)


@pytest.fixture()
def checker() -> SafetyChecker:
    return SafetyChecker()


class TestIsEnabled:
    def test_feature_enabled(self):
        from cola_coder.features.safety_checker import FEATURE_ENABLED, is_enabled

        assert FEATURE_ENABLED is True
        assert is_enabled() is True


class TestSafeCode:
    def test_safe_code_passes(self, checker):
        code = "def add(a, b):\n    return a + b\n"
        report = checker.check(code)
        assert report.passed is True
        assert report.verdict == Severity.SAFE

    def test_empty_code_safe(self, checker):
        report = checker.check("")
        assert report.passed is True

    def test_normal_print_safe(self, checker):
        report = checker.check("print('hello world')\n")
        assert report.passed is True

    def test_is_safe_method(self, checker):
        assert checker.is_safe("x = 1 + 2\n") is True


class TestEvalExec:
    def test_eval_critical(self, checker):
        report = checker.check("result = eval(user_input)\n")
        assert report.verdict == Severity.CRITICAL
        assert report.passed is False
        rules = [f.rule for f in report.findings]
        assert "eval_call" in rules

    def test_exec_critical(self, checker):
        report = checker.check("exec(untrusted_code)\n")
        assert report.verdict == Severity.CRITICAL
        assert report.critical_count >= 1

    def test_compile_high(self, checker):
        report = checker.check("code = compile(src, '<str>', 'exec')\n")
        assert report.verdict >= Severity.HIGH


class TestShellCommands:
    def test_os_system_critical(self, checker):
        report = checker.check("import os\nos.system('rm -rf /')\n")
        assert report.verdict == Severity.CRITICAL

    def test_subprocess_run_high(self, checker):
        report = checker.check("import subprocess\nsubprocess.run(['ls'])\n")
        assert report.verdict >= Severity.HIGH

    def test_shutil_rmtree_high(self, checker):
        report = checker.check("import shutil\nshutil.rmtree('/tmp/test')\n")
        assert report.verdict >= Severity.HIGH


class TestHardcodedSecrets:
    def test_password_in_string(self, checker):
        code = 'password = "supersecret123"\n'
        report = checker.check(code)
        assert report.verdict >= Severity.HIGH

    def test_api_key_literal(self, checker):
        code = 'api_key = "sk-abcdefghij1234567890"\n'
        report = checker.check(code)
        assert report.verdict >= Severity.HIGH

    def test_token_format_critical(self, checker):
        code = 'token = "ghp_ABCDEFGHIJ1234567890ABCDEF"\n'
        report = checker.check(code)
        assert report.verdict == Severity.CRITICAL


class TestInfiniteLoops:
    def test_while_true_without_break_medium(self, checker):
        code = "while True:\n    print('looping')\n"
        report = checker.check(code)
        # Should have a MEDIUM finding from the infinite loop check
        medium_plus = [
            f for f in report.findings if f.severity >= Severity.MEDIUM
        ]
        assert len(medium_plus) >= 1

    def test_while_true_with_break_ok(self, checker):
        code = "while True:\n    if done:\n        break\n"
        # Should only have LOW finding from regex (while True match), not MEDIUM from AST
        report = checker.check(code)
        medium_plus = [
            f for f in report.findings
            if f.rule == "infinite_loop_no_break"
        ]
        assert len(medium_plus) == 0  # AST check should not fire (has break)


class TestPickleYaml:
    def test_pickle_load_high(self, checker):
        report = checker.check("data = pickle.loads(raw_bytes)\n")
        assert report.verdict >= Severity.HIGH

    def test_yaml_load_medium(self, checker):
        report = checker.check("data = yaml.load(stream)\n")
        assert report.verdict >= Severity.MEDIUM


class TestFindings:
    def test_findings_have_line_number(self, checker):
        code = "line1 = 1\nresult = eval(x)\nline3 = 3\n"
        report = checker.check(code)
        for f in report.findings:
            if f.rule == "eval_call":
                assert f.line == 2

    def test_findings_have_snippet(self, checker):
        report = checker.check("eval(x)\n")
        for f in report.findings:
            if f.rule == "eval_call":
                assert "eval" in f.snippet

    def test_summary_method(self, checker):
        report = checker.check("eval(x)\n")
        s = report.summary()
        assert "CRITICAL" in s

    def test_severity_ordering(self):
        assert Severity.CRITICAL > Severity.HIGH
        assert Severity.HIGH > Severity.MEDIUM
        assert Severity.MEDIUM > Severity.LOW
        assert Severity.LOW > Severity.SAFE
