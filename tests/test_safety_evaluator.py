"""Tests for the SafetyEvaluator (evaluation/safety_eval.py).

EVAL-002: the JS/TS import regex now matches dotted package names so a
hallucinated dotted package can be detected (previously the match failed
entirely at the dot, a silent false negative).

EVAL-003 (verified, NOT a bug): the hardcoded-secret regex flags
quoted-key secret-shaped assignments in generated code — that is the
intended behavior for a safety probe, so these tests document it rather
than "fixing" it.
"""

from cola_coder.evaluation import safety_eval
from cola_coder.evaluation.safety_eval import SafetyEvaluator


class TestDottedPackageDetection:
    def test_dotted_fake_package_is_detected(self, monkeypatch):
        # A hypothetical dotted fake — patched in so we don't pollute the
        # production set, but it exercises the regex's dotted-name support.
        monkeypatch.setattr(safety_eval, "FAKE_PACKAGES", {"totally.fake.pkg"})
        ev = SafetyEvaluator()
        issues = ev._check_fake_packages('const x = require("totally.fake.pkg");')
        assert any("totally.fake.pkg" in i for i in issues)

    def test_dotted_real_import_not_flagged(self, monkeypatch):
        monkeypatch.setattr(safety_eval, "FAKE_PACKAGES", {"totally.fake.pkg"})
        ev = SafetyEvaluator()
        # A dotted name that is NOT in the fake set must not be flagged
        issues = ev._check_fake_packages('import memo from "lodash.memoize";')
        assert issues == []

    def test_known_nondotted_fake_still_detected(self):
        ev = SafetyEvaluator()
        issues = ev._check_fake_packages('const u = require("react-utils");')
        assert any("react-utils" in i for i in issues)


class TestSecretAndDangerousDetection:
    def test_quoted_key_secret_flagged(self):
        # Intended behavior (EVAL-003): secret-shaped quoted assignment flagged.
        ev = SafetyEvaluator()
        issues = ev._check_secrets('config = {"api_key": "abcdef123456"}')
        assert any("Hardcoded Secret" in i for i in issues)

    def test_aws_key_pattern_flagged(self):
        ev = SafetyEvaluator()
        assert ev._check_secrets("key = 'AKIAIOSFODNN7EXAMPLE'")

    def test_clean_code_has_no_secret(self):
        ev = SafetyEvaluator()
        assert ev._check_secrets("def add(a, b):\n    return a + b") == []

    def test_dangerous_eval_flagged(self):
        ev = SafetyEvaluator()
        assert ev._check_dangerous("result = eval(user_input)")

    def test_dangerous_rm_rf_flagged(self):
        ev = SafetyEvaluator()
        assert ev._check_dangerous("os.system('rm -rf /tmp/x')")
