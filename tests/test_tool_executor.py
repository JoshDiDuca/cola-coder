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
