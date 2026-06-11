"""Security + behavior tests for the agent ToolExecutor.

The tools/ module had ZERO coverage. The headline guard is path-traversal
containment: _validate_path used str.startswith, which a SIBLING directory
with a shared name prefix could bypass (".../proj-secrets" starts with
".../proj"), letting read_file/run_tests/lint escape the project root.
"""

import pytest

from cola_coder.tools.executor import ToolExecutor, ToolResult


@pytest.fixture()
def project(tmp_path):
    """A project root plus a sibling 'secret' dir sharing a name prefix."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "ok.txt").write_text("inside the project", encoding="utf-8")
    sub = root / "src"
    sub.mkdir()
    (sub / "main.py").write_text("print('hi')\n", encoding="utf-8")

    sibling = tmp_path / "proj-secrets"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
    return root


class TestPathTraversal:
    def test_sibling_prefix_bypass_blocked(self, project):
        ex = ToolExecutor(project_root=str(project))
        res = ex.execute("read_file", {"path": "../proj-secrets/secret.txt"})
        assert res.success is False
        assert "SECRET" not in res.output
        assert "traversal" in res.error.lower()

    def test_parent_escape_blocked(self, project):
        ex = ToolExecutor(project_root=str(project))
        res = ex.execute("read_file", {"path": "../../etc/passwd"})
        assert res.success is False
        assert "traversal" in res.error.lower()

    def test_absolute_path_outside_blocked(self, project, tmp_path):
        ex = ToolExecutor(project_root=str(project))
        outside = tmp_path / "proj-secrets" / "secret.txt"
        res = ex.execute("read_file", {"path": str(outside)})
        assert res.success is False

    def test_valid_in_root_allowed(self, project):
        ex = ToolExecutor(project_root=str(project))
        res = ex.execute("read_file", {"path": "ok.txt"})
        assert res.success is True
        assert "inside the project" in res.output

    def test_valid_nested_allowed(self, project):
        ex = ToolExecutor(project_root=str(project))
        res = ex.execute("read_file", {"path": "src/main.py"})
        assert res.success is True
        assert "print" in res.output

    def test_validate_path_helper_rejects_sibling(self, project):
        ex = ToolExecutor(project_root=str(project))
        with pytest.raises(ValueError, match="traversal"):
            ex._validate_path("../proj-secrets/secret.txt")
        # Root itself and nested paths are fine.
        assert ex._validate_path("ok.txt") == (project / "ok.txt").resolve()


class TestUnknownTool:
    def test_unknown_tool_returns_error(self, tmp_path):
        res = ToolExecutor(project_root=str(tmp_path)).execute("rm_rf", {})
        assert isinstance(res, ToolResult)
        assert res.success is False
        assert "Unknown tool" in res.error


class TestGitRefValidation:
    # git_diff signals a bad ref by returning the "invalid git ref" message
    # BEFORE running any subprocess (the handler validates and returns early).
    def test_flag_injection_ref_rejected(self, project):
        ex = ToolExecutor(project_root=str(project))
        for bad in ("--ext-diff", "-R", "--output=x"):
            res = ex.execute("git_diff", {"ref": bad})
            assert "invalid git ref" in res.output

    def test_ref_with_shell_metachars_rejected(self, project):
        ex = ToolExecutor(project_root=str(project))
        res = ex.execute("git_diff", {"ref": "HEAD; rm -rf /"})
        # Space/; not in the allowed char set → rejected before any subprocess.
        assert "invalid git ref" in res.output

    def test_plain_ref_passes_validation(self, project):
        # A normal ref isn't rejected by the validator (it reaches git, which
        # may or may not succeed — we only assert it's not the validation error).
        ex = ToolExecutor(project_root=str(project))
        res = ex.execute("git_diff", {"ref": "HEAD"})
        assert "invalid git ref" not in res.output


class TestReadFileMissing:
    def test_missing_file_reports_not_found(self, project):
        ex = ToolExecutor(project_root=str(project))
        res = ex.execute("read_file", {"path": "does_not_exist.txt"})
        # Path is valid (inside root) but file absent.
        assert "not found" in res.output.lower()


class _FakeTscRunner:
    """Stand-in for TscRunner so typecheck tests don't need tsc on PATH."""

    def __init__(self, errors, available=True):
        self._errors = errors
        self._available = available
        self.seen = []

    def is_available(self):
        return self._available

    def check(self, code):
        self.seen.append(code)
        return list(self._errors)


class TestTypecheckRoutesThroughTscRunner:
    """typecheck runs UNTRUSTED model code, so it must go through the sandboxed
    TscRunner (hardened tsconfig, no network) — never ad-hoc `npx tsc`."""

    def test_no_code_returns_error(self, project):
        ex = ToolExecutor(project_root=str(project))
        res = ex.execute("typecheck", {"code": ""})
        assert "no code" in res.output.lower()

    def test_clean_code_reports_ok(self, project):
        from cola_coder.reasoning.rewards.tsc_runner import TscError

        ex = ToolExecutor(project_root=str(project))
        fake = _FakeTscRunner(errors=[])
        ex.__dict__["_tsc_runners"] = {True: fake}
        res = ex.execute("typecheck", {"code": "const x: number = 1;\n"})
        assert res.success is True
        assert "OK" in res.output
        assert fake.seen == ["const x: number = 1;\n"]  # routed through runner
        assert TscError  # symbol exists for the error path below

    def test_type_errors_are_formatted(self, project):
        from cola_coder.reasoning.rewards.tsc_runner import TscError

        ex = ToolExecutor(project_root=str(project))
        err = TscError(
            file="check.ts", line=1, col=7, severity="error",
            code="TS2322", message="Type 'string' is not assignable to 'number'.",
        )
        ex.__dict__["_tsc_runners"] = {True: _FakeTscRunner(errors=[err])}
        res = ex.execute("typecheck", {"code": "const x: number = 'a';\n"})
        assert "TS2322" in res.output
        assert "(1,7)" in res.output

    def test_warnings_are_ignored(self, project):
        from cola_coder.reasoning.rewards.tsc_runner import TscError

        ex = ToolExecutor(project_root=str(project))
        warn = TscError(
            file="check.ts", line=2, col=1, severity="warning",
            code="TS6133", message="'y' is declared but never used.",
        )
        ex.__dict__["_tsc_runners"] = {True: _FakeTscRunner(errors=[warn])}
        res = ex.execute("typecheck", {"code": "const y = 1;\n"})
        assert "OK" in res.output  # only severity == "error" counts

    def test_unavailable_tsc_reports_gracefully(self, project):
        ex = ToolExecutor(project_root=str(project))
        ex.__dict__["_tsc_runners"] = {True: _FakeTscRunner(errors=[], available=False)}
        res = ex.execute("typecheck", {"code": "const x = 1;\n"})
        assert "not available" in res.output.lower()

    def test_strict_flag_selects_distinct_runner(self, project):
        ex = ToolExecutor(project_root=str(project))
        strict_runner = _FakeTscRunner(errors=[])
        loose_runner = _FakeTscRunner(errors=[])
        ex.__dict__["_tsc_runners"] = {True: strict_runner, False: loose_runner}
        ex.execute("typecheck", {"code": "a", "strict": False})
        assert loose_runner.seen == ["a"] and strict_runner.seen == []
