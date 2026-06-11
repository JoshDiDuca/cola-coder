"""INFER-012: SelfVerifier (best-of-N heuristic fallback) must be language-aware.

SelfVerifier ranks best-of-N candidates when no tool verifier (tsc) is available.
verify_no_hallucination flagged console.log / var / undefined as "JavaScript-isms
in Python code" — but this is a TS-PRIMARY project, where those are VALID. So on
the fallback path it penalised correct TypeScript candidates. Now the JS-ism
checks only fire for Python (explicit hint or content detection). The module
previously had ZERO tests.
"""

from cola_coder.features.self_verification import SelfVerifier

V = SelfVerifier()


class TestHallucinationLanguageAware:
    def test_python_console_log_is_flagged(self):
        py = "def f(x):\n    console.log(x)\n    return x\n"  # content => Python
        issues = V.verify_no_hallucination(py)
        assert any("console.log" in i for i in issues)

    def test_typescript_console_log_not_flagged_by_content(self):
        ts = "const greet = (name: string): void => { console.log(`hi ${name}`); };\n"
        issues = V.verify_no_hallucination(ts)  # content detected as JS/TS
        assert not any(
            ("console.log" in i) or ("var" in i) or ("undefined" in i) for i in issues
        )

    def test_explicit_typescript_hint_skips_js_checks(self):
        # Minimal TS that content-detection alone wouldn't catch (no const/=>).
        code = "var x = 1; console.log(x);"
        issues = V.verify_no_hallucination(code, language="typescript")
        assert not any("console.log" in i or "var" in i for i in issues)

    def test_explicit_python_hint_flags_js(self):
        code = "x = 1\nconsole.log(x)\n"
        issues = V.verify_no_hallucination(code, language="python")
        assert any("console.log" in i for i in issues)

    def test_ts_candidate_not_penalised_in_verify_code(self):
        ts = "const add = (a: number, b: number): number => { return a + b; };\n"
        r = V.verify_code(ts, language="typescript")
        assert not any(
            "console" in i.lower() or "javascript" in i.lower() for i in r.issues
        )


class TestLanguageAgnosticChecksStillFire:
    def test_suspicious_module_flagged(self):
        code = "import torch.utils.data.experimental as x\n"
        assert V.verify_no_hallucination(code)  # non-empty

    def test_repetition_flagged(self):
        code = "\n".join(["result = compute(value)"] * 4)
        assert any("Repeated line" in i for i in V.verify_no_hallucination(code))


class TestSyntaxAndCompleteness:
    def test_valid_python_syntax(self):
        assert V.verify_syntax("def f():\n    return 1\n") is True

    def test_unbalanced_brackets_fail(self):
        assert V.verify_syntax("def f(:\n    return (1\n") is False

    def test_valid_ts_passes_bracket_fallback(self):
        # Not valid Python (ast fails) but brackets balanced -> passes fallback.
        assert V.verify_syntax("const x = (a) => a + 1;") is True

    def test_empty_code_completeness_zero(self):
        assert V.verify_completeness("", "do something useful") == 0.0
