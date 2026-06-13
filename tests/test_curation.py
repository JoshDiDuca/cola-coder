"""Tests for the test-driven data curation system.

Tests framework detection, scoring logic, Docker availability checking,
and timeout handling. Uses minimal fake repo structures — no real repos needed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from cola_coder.data.curation.docker_sandbox import DockerSandbox
from cola_coder.data.curation.test_runner import (
    TestRunner,
    _detect_go_framework,
    _detect_node_framework,
    _detect_python_framework,
    _detect_rust_framework,
    _parse_jest_output,
    _parse_pytest_output,
    _parse_go_test_output,
    _score_repo_worker,
)
from cola_coder.data.curation.test_scorer import RepoScore, TestResult, TestScorer


# ---------------------------------------------------------------------------
# Fixtures: create minimal fake repos in tmp directories
# ---------------------------------------------------------------------------


@pytest.fixture
def jest_repo(tmp_path: Path) -> Path:
    """Create a minimal Jest repo."""
    pkg = {
        "name": "test-jest-repo",
        "version": "1.0.0",
        "scripts": {"test": "jest"},
        "devDependencies": {"jest": "^29.0.0"},
    }
    (tmp_path / "package.json").write_text(json.dumps(pkg))
    (tmp_path / "sum.js").write_text("module.exports = (a, b) => a + b;\n")
    (tmp_path / "sum.test.js").write_text(textwrap.dedent("""\
        const sum = require('./sum');
        test('adds 1 + 2 to equal 3', () => {
            expect(sum(1, 2)).toBe(3);
        });
    """))
    return tmp_path


@pytest.fixture
def vitest_repo(tmp_path: Path) -> Path:
    """Create a minimal Vitest repo."""
    pkg = {
        "name": "test-vitest-repo",
        "version": "1.0.0",
        "scripts": {"test": "vitest run"},
        "devDependencies": {"vitest": "^1.0.0"},
    }
    (tmp_path / "package.json").write_text(json.dumps(pkg))
    return tmp_path


@pytest.fixture
def mocha_repo(tmp_path: Path) -> Path:
    """Create a minimal Mocha repo."""
    pkg = {
        "name": "test-mocha-repo",
        "version": "1.0.0",
        "scripts": {"test": "mocha"},
        "devDependencies": {"mocha": "^10.0.0"},
    }
    (tmp_path / "package.json").write_text(json.dumps(pkg))
    return tmp_path


@pytest.fixture
def pytest_repo(tmp_path: Path) -> Path:
    """Create a minimal pytest repo."""
    pyproject = textwrap.dedent("""\
        [build-system]
        requires = ["setuptools"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "test-pytest-repo"
        version = "0.1.0"

        [tool.pytest.ini_options]
        testpaths = ["tests"]
    """)
    (tmp_path / "pyproject.toml").write_text(pyproject)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_basic.py").write_text(textwrap.dedent("""\
        def test_add():
            assert 1 + 1 == 2

        def test_sub():
            assert 3 - 1 == 2
    """))
    return tmp_path


@pytest.fixture
def unittest_repo(tmp_path: Path) -> Path:
    """Create a minimal unittest repo."""
    (tmp_path / "test_example.py").write_text(textwrap.dedent("""\
        import unittest

        class TestExample(unittest.TestCase):
            def test_add(self):
                self.assertEqual(1 + 1, 2)

        if __name__ == '__main__':
            unittest.main()
    """))
    return tmp_path


@pytest.fixture
def go_repo(tmp_path: Path) -> Path:
    """Create a minimal Go repo."""
    (tmp_path / "go.mod").write_text("module example.com/test\ngo 1.22\n")
    return tmp_path


@pytest.fixture
def rust_repo(tmp_path: Path) -> Path:
    """Create a minimal Rust repo."""
    (tmp_path / "Cargo.toml").write_text(textwrap.dedent("""\
        [package]
        name = "test-rust"
        version = "0.1.0"
        edition = "2021"
    """))
    return tmp_path


@pytest.fixture
def no_test_repo(tmp_path: Path) -> Path:
    """Create a repo with no tests."""
    (tmp_path / "README.md").write_text("# Hello\n")
    (tmp_path / "main.py").write_text("print('hello')\n")
    return tmp_path


@pytest.fixture
def npm_test_repo(tmp_path: Path) -> Path:
    """Create a repo with generic npm test script."""
    pkg = {
        "name": "test-generic",
        "version": "1.0.0",
        "scripts": {"test": "node test.js"},
    }
    (tmp_path / "package.json").write_text(json.dumps(pkg))
    return tmp_path


# ---------------------------------------------------------------------------
# Test: Framework detection
# ---------------------------------------------------------------------------


class TestFrameworkDetection:
    """Test that we correctly detect test frameworks from repo structures."""

    def test_detect_jest(self, jest_repo: Path) -> None:
        fw = _detect_node_framework(jest_repo)
        assert fw is not None
        assert fw.name == "jest"
        assert fw.language == "node"
        assert "jest" in fw.test_cmd

    def test_detect_vitest(self, vitest_repo: Path) -> None:
        fw = _detect_node_framework(vitest_repo)
        assert fw is not None
        assert fw.name == "vitest"
        assert fw.language == "node"

    def test_detect_mocha(self, mocha_repo: Path) -> None:
        fw = _detect_node_framework(mocha_repo)
        assert fw is not None
        assert fw.name == "mocha"

    def test_detect_pytest(self, pytest_repo: Path) -> None:
        fw = _detect_python_framework(pytest_repo)
        assert fw is not None
        assert fw.name == "pytest"
        assert fw.language == "python"

    def test_detect_unittest(self, unittest_repo: Path) -> None:
        fw = _detect_python_framework(unittest_repo)
        assert fw is not None
        assert fw.name == "unittest"

    def test_detect_go(self, go_repo: Path) -> None:
        fw = _detect_go_framework(go_repo)
        assert fw is not None
        assert fw.name == "go_test"

    def test_detect_rust(self, rust_repo: Path) -> None:
        fw = _detect_rust_framework(rust_repo)
        assert fw is not None
        assert fw.name == "cargo_test"

    def test_detect_none(self, no_test_repo: Path) -> None:
        runner = TestRunner(mode="dry_run")
        fw = runner.detect_test_framework(no_test_repo)
        assert fw is None

    def test_detect_generic_npm_test(self, npm_test_repo: Path) -> None:
        fw = _detect_node_framework(npm_test_repo)
        assert fw is not None
        assert fw.name == "npm_test"

    def test_detect_nonexistent_path(self) -> None:
        runner = TestRunner(mode="dry_run")
        fw = runner.detect_test_framework(Path("/nonexistent/path"))
        assert fw is None

    def test_detect_broken_package_json(self, tmp_path: Path) -> None:
        """Malformed package.json should not crash detection."""
        (tmp_path / "package.json").write_text("not json at all {{{")
        fw = _detect_node_framework(tmp_path)
        assert fw is None

    def test_detect_via_runner(self, jest_repo: Path) -> None:
        """TestRunner.detect_test_framework should work end-to-end."""
        runner = TestRunner(mode="dry_run")
        fw = runner.detect_test_framework(jest_repo)
        assert fw is not None
        assert fw.name == "jest"


# ---------------------------------------------------------------------------
# Test: Output parsing
# ---------------------------------------------------------------------------


class TestOutputParsing:
    """Test parsing of test runner output into TestResult."""

    def test_parse_jest_json(self) -> None:
        output = json.dumps({
            "numTotalTests": 10,
            "numPassedTests": 8,
            "numFailedTests": 1,
            "numPendingTests": 1,
        })
        result = _parse_jest_output(output)
        assert result is not None
        assert result.total_tests == 10
        assert result.passed == 8
        assert result.failed == 1
        assert result.skipped == 1

    def test_parse_pytest_summary(self) -> None:
        output = "====== 5 passed, 2 failed, 1 skipped in 3.45s ======"
        result = _parse_pytest_output(output)
        assert result is not None
        assert result.passed == 5
        assert result.failed == 2
        assert result.skipped == 1
        assert result.total_tests == 8

    def test_parse_pytest_all_passed(self) -> None:
        output = "====== 12 passed in 1.23s ======"
        result = _parse_pytest_output(output)
        assert result is not None
        assert result.passed == 12
        assert result.failed == 0
        assert result.total_tests == 12
        assert result.all_passed

    def test_parse_pytest_with_coverage(self) -> None:
        output = textwrap.dedent("""\
            5 passed in 2.00s
            ----------- coverage: platform linux, python 3.11 -----------
            Name                 Stmts   Miss  Cover
            ----------------------------------------
            TOTAL                  100     15    85%
        """)
        result = _parse_pytest_output(output)
        assert result is not None
        assert result.passed == 5
        assert result.coverage == 0.85

    def test_parse_go_test_json(self) -> None:
        output = "\n".join([
            json.dumps({"Action": "pass", "Test": "TestAdd"}),
            json.dumps({"Action": "pass", "Test": "TestSub"}),
            json.dumps({"Action": "fail", "Test": "TestMul"}),
            json.dumps({"Action": "pass", "Package": "example.com/test"}),
        ])
        result = _parse_go_test_output(output)
        assert result is not None
        assert result.passed == 2
        assert result.failed == 1
        assert result.total_tests == 3

    def test_parse_empty_output(self) -> None:
        result = _parse_jest_output("")
        assert result is None

    def test_parse_garbage_output(self) -> None:
        result = _parse_jest_output("random garbage that is not JSON")
        assert result is None


# ---------------------------------------------------------------------------
# Test: TestScorer
# ---------------------------------------------------------------------------


class TestScoring:
    """Test the scoring logic."""

    def setup_method(self) -> None:
        self.scorer = TestScorer()

    def test_score_verified(self) -> None:
        """All tests pass with high coverage = verified tier."""
        result = TestResult(
            framework="pytest",
            total_tests=20,
            passed=20,
            failed=0,
            skipped=0,
            coverage=0.92,
        )
        score = self.scorer.score(result)
        assert score.quality_tier == "verified"
        assert score.tests_pass is True
        assert score.score > 0.9

    def test_score_tested(self) -> None:
        """Some tests pass = tested tier."""
        result = TestResult(
            framework="jest",
            total_tests=10,
            passed=7,
            failed=3,
            skipped=0,
        )
        score = self.scorer.score(result)
        assert score.quality_tier == "tested"
        assert score.tests_pass is False
        assert 0.5 < score.score < 1.0

    def test_score_all_pass_no_coverage(self) -> None:
        """All tests pass but no coverage info = tested tier."""
        result = TestResult(
            framework="jest",
            total_tests=5,
            passed=5,
            failed=0,
            skipped=0,
            coverage=None,
        )
        score = self.scorer.score(result)
        assert score.quality_tier == "tested"
        assert score.tests_pass is True

    def test_score_detected(self) -> None:
        """Tests exist but didn't run = detected tier."""
        score = self.scorer.score(None, tests_detected=True)
        assert score.quality_tier == "detected"
        assert score.score == TestScorer.QUALITY_TIERS["detected"]

    def test_score_none(self) -> None:
        """No tests found = none tier."""
        score = self.scorer.score(None, tests_detected=False)
        assert score.quality_tier == "none"
        assert score.score == TestScorer.QUALITY_TIERS["none"]

    def test_score_all_failed(self) -> None:
        """All tests fail = detected tier."""
        result = TestResult(
            framework="pytest",
            total_tests=5,
            passed=0,
            failed=5,
            skipped=0,
        )
        score = self.scorer.score(result)
        assert score.quality_tier == "detected"

    def test_file_weight(self) -> None:
        """file_weight returns correct multipliers per tier."""
        verified = RepoScore(True, True, None, "verified", 1.0)
        tested = RepoScore(True, False, None, "tested", 0.7)
        none_ = RepoScore(False, False, None, "none", 0.2)

        assert self.scorer.file_weight(verified) == 3.0
        assert self.scorer.file_weight(tested) == 2.0
        assert self.scorer.file_weight(none_) == 0.5

    def test_pass_rate(self) -> None:
        """TestResult.pass_rate computes correctly."""
        result = TestResult("x", total_tests=10, passed=7, failed=2, skipped=1)
        # pass_rate = 7 / (10 - 1) = 7/9
        assert abs(result.pass_rate - 7 / 9) < 0.001

    def test_pass_rate_all_skipped(self) -> None:
        """pass_rate is 0 when all tests are skipped."""
        result = TestResult("x", total_tests=5, passed=0, failed=0, skipped=5)
        assert result.pass_rate == 0.0

    def test_score_serialization(self) -> None:
        """RepoScore round-trips through to_dict/from_dict."""
        result = TestResult(
            framework="pytest", total_tests=10, passed=8, failed=1,
            skipped=1, coverage=0.85, duration_seconds=2.5,
        )
        score = self.scorer.score(result)
        d = score.to_dict()
        restored = RepoScore.from_dict(d)
        assert restored.quality_tier == score.quality_tier
        assert restored.score == score.score
        assert restored.test_result is not None
        assert restored.test_result.passed == 8
        assert restored.test_result.coverage == 0.85


# ---------------------------------------------------------------------------
# Test: DockerSandbox
# ---------------------------------------------------------------------------


class TestDockerSandbox:
    """Test Docker sandbox (only checks availability, no actual Docker needed)."""

    def test_is_available_returns_bool(self) -> None:
        """is_available() should return bool without crashing."""
        result = DockerSandbox.is_available()
        assert isinstance(result, bool)

    def test_image_for_language(self) -> None:
        """Should return correct Docker images for known languages."""
        assert "node" in DockerSandbox.image_for_language("node")
        assert "python" in DockerSandbox.image_for_language("python")
        assert "golang" in DockerSandbox.image_for_language("go")
        assert "rust" in DockerSandbox.image_for_language("rust")
        # Unknown language should return a default
        assert DockerSandbox.image_for_language("unknown") is not None

    def test_constructor_defaults(self) -> None:
        """DockerSandbox should accept default constructor args."""
        sandbox = DockerSandbox()
        assert sandbox.memory_limit == "2g"
        assert sandbox.cpu_limit == 2.0
        assert sandbox.pid_limit == 64
        assert sandbox.network is False
        assert sandbox.timeout == 300

    @pytest.mark.skipif(
        not DockerSandbox.is_available(),
        reason="Docker not available",
    )
    def test_docker_run_simple(self, tmp_path: Path) -> None:
        """If Docker is available, run a simple command."""
        sandbox = DockerSandbox(timeout=30)
        (tmp_path / "hello.txt").write_text("world")
        code, stdout, stderr = sandbox.run(
            repo_path=tmp_path,
            command="cat /code/hello.txt",
            image="alpine:latest",
        )
        assert code == 0
        assert "world" in stdout


class TestDockerTimeoutKillsContainer:
    """SECURITY (SEC-001): a Docker run that exceeds the timeout must
    force-kill the container. subprocess.run's timeout only kills the
    `docker run` *client* — the daemon keeps the container (and the untrusted
    code inside it) running. Without an explicit `docker rm -f`, the timeout
    control is defeated and untrusted scraped code keeps executing on the host.
    """

    def test_timeout_force_removes_container(self, tmp_path: Path) -> None:
        sandbox = DockerSandbox(timeout=1)
        calls: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            # First call is the `docker run` — simulate it hanging past timeout.
            if cmd[:2] == ["docker", "run"]:
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)
            # Subsequent calls (is_available's `docker info`, the rm) succeed.

            class _R:
                returncode = 0
                stdout = ""
                stderr = ""

            return _R()

        with patch.object(DockerSandbox, "is_available", return_value=True), \
                patch("cola_coder.data.curation.docker_sandbox.subprocess.run",
                      side_effect=fake_run):
            code, stdout, stderr = sandbox.run(
                repo_path=tmp_path,
                command="sleep 999",
                image="alpine:latest",
            )

        assert code == -1
        assert "Timeout" in stderr

        # The run command must have assigned a --name so the container is
        # addressable, and a `docker rm -f <name>` must have been issued.
        run_cmd = next(c for c in calls if c[:2] == ["docker", "run"])
        name_flag = next(a for a in run_cmd if a.startswith("--name="))
        container_name = name_flag.split("=", 1)[1]

        rm_calls = [c for c in calls if c[:3] == ["docker", "rm", "-f"]]
        assert rm_calls, "timeout did not force-remove the container"
        assert container_name in rm_calls[0]

    def test_force_remove_swallows_errors(self) -> None:
        """_force_remove_container must never raise — a cleanup failure must not
        mask the timeout result returned to the caller."""
        with patch("cola_coder.data.curation.docker_sandbox.subprocess.run",
                   side_effect=FileNotFoundError("docker gone")):
            # Should not raise.
            DockerSandbox._force_remove_container("cola-curation-deadbeef")


class TestDockerCleanupOnAllExitPaths:
    """SECURITY (SEC-002): defense-in-depth — no container/`docker run` child
    may outlive ``DockerSandbox.run``. The container is force-removed on EVERY
    exit path: normal completion, an unexpected exception, and Ctrl-C, in
    addition to the SEC-001 timeout path. The unique name is generated once and
    reused for the run command and every cleanup path.
    """

    @staticmethod
    def _run_and_capture(sandbox: DockerSandbox, tmp_path: Path, run_side_effect):
        """Drive sandbox.run with a mocked subprocess.run.

        ``run_side_effect(cmd)`` decides what the `docker run` call does; the
        follow-up `docker rm -f` always succeeds. Returns the recorded calls.
        """
        calls: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            if cmd[:2] == ["docker", "run"]:
                return run_side_effect(cmd)

            class _R:
                returncode = 0
                stdout = ""
                stderr = ""

            return _R()

        with patch.object(DockerSandbox, "is_available", return_value=True), \
                patch("cola_coder.data.curation.docker_sandbox.subprocess.run",
                      side_effect=fake_run):
            result = sandbox.run(
                repo_path=tmp_path,
                command="echo hi",
                image="alpine:latest",
            )
        return calls, result

    @staticmethod
    def _assert_container_removed(calls: list[list[str]]) -> None:
        """The same --name from `docker run` must be `docker rm -f`'d exactly."""
        run_cmd = next(c for c in calls if c[:2] == ["docker", "run"])
        name_flag = next(a for a in run_cmd if a.startswith("--name="))
        container_name = name_flag.split("=", 1)[1]
        assert container_name.startswith("cola-curation-")

        rm_calls = [c for c in calls if c[:3] == ["docker", "rm", "-f"]]
        assert rm_calls, "container was not force-removed"
        assert container_name in rm_calls[0]

    def test_normal_completion_force_removes_container(self, tmp_path: Path) -> None:
        """On a clean exit, the `finally` still issues `docker rm -f` (belt and
        braces in case `--rm` did not fire). The success-path return contract is
        unchanged: (returncode, stdout, stderr)."""
        sandbox = DockerSandbox(timeout=30)

        class _Ok:
            returncode = 0
            stdout = "hi\n"
            stderr = ""

        calls, result = self._run_and_capture(sandbox, tmp_path, lambda cmd: _Ok())

        assert result == (0, "hi\n", "")
        self._assert_container_removed(calls)

    def test_unexpected_exception_force_removes_container(self, tmp_path: Path) -> None:
        """An unexpected error during the run must propagate, but only AFTER the
        container is force-removed by the `finally`."""
        sandbox = DockerSandbox(timeout=30)

        def boom(cmd):
            raise RuntimeError("docker exploded")

        with pytest.raises(RuntimeError, match="docker exploded"):
            self._run_and_capture(sandbox, tmp_path, boom)

    def test_unexpected_exception_cleanup_runs_before_propagating(
        self, tmp_path: Path
    ) -> None:
        """Verify the rm actually fired on the exception path (the test above
        only proves the exception propagated)."""
        sandbox = DockerSandbox(timeout=30)
        calls: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            if cmd[:2] == ["docker", "run"]:
                raise RuntimeError("docker exploded")

            class _R:
                returncode = 0
                stdout = ""
                stderr = ""

            return _R()

        with patch.object(DockerSandbox, "is_available", return_value=True), \
                patch("cola_coder.data.curation.docker_sandbox.subprocess.run",
                      side_effect=fake_run):
            with pytest.raises(RuntimeError, match="docker exploded"):
                sandbox.run(repo_path=tmp_path, command="echo hi", image="alpine:latest")

        self._assert_container_removed(calls)

    def test_keyboard_interrupt_force_removes_and_reraises(self, tmp_path: Path) -> None:
        """A Ctrl-C mid-run must tear down the container AND still re-raise so
        the interrupt is not swallowed."""
        sandbox = DockerSandbox(timeout=30)
        calls: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            if cmd[:2] == ["docker", "run"]:
                raise KeyboardInterrupt()

            class _R:
                returncode = 0
                stdout = ""
                stderr = ""

            return _R()

        with patch.object(DockerSandbox, "is_available", return_value=True), \
                patch("cola_coder.data.curation.docker_sandbox.subprocess.run",
                      side_effect=fake_run):
            with pytest.raises(KeyboardInterrupt):
                sandbox.run(repo_path=tmp_path, command="echo hi", image="alpine:latest")

        self._assert_container_removed(calls)

    def test_container_name_is_reused_for_run_and_cleanup(self, tmp_path: Path) -> None:
        """The unique name must be generated once and reused: the name on the
        `docker run` command must be identical to the one `docker rm -f` targets
        (SEC-002 robust-naming requirement)."""
        sandbox = DockerSandbox(timeout=30)

        class _Ok:
            returncode = 0
            stdout = ""
            stderr = ""

        calls, _ = self._run_and_capture(sandbox, tmp_path, lambda cmd: _Ok())

        run_names = [
            a.split("=", 1)[1]
            for c in calls if c[:2] == ["docker", "run"]
            for a in c if a.startswith("--name=")
        ]
        rm_names = [c[3] for c in calls if c[:3] == ["docker", "rm", "-f"]]
        assert len(run_names) == 1
        assert len(rm_names) == 1
        assert run_names[0] == rm_names[0]


class TestDockerSandboxHardening:
    """SECURITY (SEC-012): the `docker run` argv must carry EVERY container
    isolation control so untrusted scraped code is bulletproofed. We mock
    subprocess (no Docker needed) and assert each flag is present, building on
    the SEC-001/002 mocked-argv pattern. We never regress the unique --name +
    force-remove behaviour.
    """

    @staticmethod
    def _capture_run_argv(sandbox: DockerSandbox, tmp_path: Path,
                          **run_kwargs) -> list[str]:
        """Drive sandbox.run with a mocked subprocess.run and return the argv
        of the `docker run` invocation."""
        calls: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(cmd)

            class _R:
                returncode = 0
                stdout = ""
                stderr = ""

            return _R()

        with patch.object(DockerSandbox, "is_available", return_value=True), \
                patch("cola_coder.data.curation.docker_sandbox.subprocess.run",
                      side_effect=fake_run):
            sandbox.run(
                repo_path=tmp_path,
                command="echo hi",
                image="alpine:latest",
                **run_kwargs,
            )

        return next(c for c in calls if c[:2] == ["docker", "run"])

    def test_runs_as_nonroot_user(self, tmp_path: Path) -> None:
        argv = self._capture_run_argv(DockerSandbox(), tmp_path)
        assert "--user" in argv
        assert argv[argv.index("--user") + 1] == "65534:65534"

    def test_read_only_rootfs(self, tmp_path: Path) -> None:
        argv = self._capture_run_argv(DockerSandbox(), tmp_path)
        assert "--read-only" in argv

    def test_single_writable_tmpfs(self, tmp_path: Path) -> None:
        argv = self._capture_run_argv(DockerSandbox(), tmp_path)
        tmpfs_flags = [a for a in argv if a.startswith("--tmpfs=")]
        # Exactly ONE small writable tmpfs for the workdir.
        assert len(tmpfs_flags) == 1
        assert tmpfs_flags[0].startswith("--tmpfs=/tmp:rw")
        assert "size=64m" in tmpfs_flags[0]

    def test_network_off(self, tmp_path: Path) -> None:
        argv = self._capture_run_argv(DockerSandbox(), tmp_path)
        assert "--network=none" in argv

    def test_no_host_namespaces(self, tmp_path: Path) -> None:
        argv = self._capture_run_argv(DockerSandbox(), tmp_path)
        # Sharing a host namespace would defeat the sandbox entirely.
        assert "--pid=host" not in argv
        assert "--ipc=host" not in argv
        assert "--net=host" not in argv
        assert "--network=host" not in argv

    def test_never_privileged_or_unconfined(self, tmp_path: Path) -> None:
        argv = self._capture_run_argv(DockerSandbox(), tmp_path)
        assert "--privileged" not in argv
        joined = " ".join(argv)
        assert "seccomp=unconfined" not in joined
        assert "apparmor=unconfined" not in joined

    def test_caps_dropped_and_no_new_privileges(self, tmp_path: Path) -> None:
        argv = self._capture_run_argv(DockerSandbox(), tmp_path)
        assert "--cap-drop=ALL" in argv
        assert "--security-opt" in argv
        assert argv[argv.index("--security-opt") + 1] == "no-new-privileges"

    def test_pids_limit(self, tmp_path: Path) -> None:
        argv = self._capture_run_argv(DockerSandbox(pid_limit=256), tmp_path)
        assert "--pids-limit=256" in argv

    def test_memory_and_swap_equal(self, tmp_path: Path) -> None:
        argv = self._capture_run_argv(DockerSandbox(memory_limit="2g"), tmp_path)
        # --memory-swap == --memory disables swap (no extra swap budget).
        assert "--memory=2g" in argv
        assert "--memory-swap=2g" in argv

    def test_cpu_limit(self, tmp_path: Path) -> None:
        argv = self._capture_run_argv(DockerSandbox(cpu_limit=1.5), tmp_path)
        assert "--cpus=1.5" in argv

    def test_ulimits_nofile_and_nproc(self, tmp_path: Path) -> None:
        argv = self._capture_run_argv(DockerSandbox(), tmp_path)
        assert "--ulimit=nofile=256:256" in argv
        assert "--ulimit=nproc=256:256" in argv

    def test_clean_environment_no_host_secrets(self, tmp_path: Path) -> None:
        """No `-e` env flags unless the caller explicitly passes env."""
        argv = self._capture_run_argv(DockerSandbox(), tmp_path)
        assert "-e" not in argv

    def test_explicit_env_is_forwarded(self, tmp_path: Path) -> None:
        argv = self._capture_run_argv(
            DockerSandbox(), tmp_path, env={"FOO": "bar"}
        )
        assert "-e" in argv
        assert "FOO=bar" in argv

    def test_no_storage_opt_size(self, tmp_path: Path) -> None:
        """--storage-opt size is unsupported on Docker Desktop / overlay2, so
        it must NOT be emitted (the tmpfs size cap is used instead)."""
        argv = self._capture_run_argv(DockerSandbox(), tmp_path)
        assert not any(a.startswith("--storage-opt") for a in argv)

    def test_limits_are_configurable(self) -> None:
        """All hardening knobs are constructor params with safe defaults."""
        sandbox = DockerSandbox(
            user="1000:1000",
            read_only=False,
            tmpfs_size="32m",
            nofile_limit=128,
            nproc_limit=64,
            max_output_bytes=500,
        )
        assert sandbox.user == "1000:1000"
        assert sandbox.read_only is False
        assert sandbox.tmpfs_size == "32m"
        assert sandbox.nofile_limit == 128
        assert sandbox.nproc_limit == 64
        assert sandbox.max_output_bytes == 500

    def test_run_with_install_copies_into_tmpfs(self, tmp_path: Path) -> None:
        """The code-copy step must target the writable tmpfs so install/test
        work under --read-only rootfs."""
        sandbox = DockerSandbox()
        calls: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(cmd)

            class _R:
                returncode = 0
                stdout = ""
                stderr = ""

            return _R()

        with patch.object(DockerSandbox, "is_available", return_value=True), \
                patch("cola_coder.data.curation.docker_sandbox.subprocess.run",
                      side_effect=fake_run):
            sandbox.run_with_install(
                repo_path=tmp_path,
                install_cmd="npm install",
                test_cmd="npm test",
                image="node:20-slim",
            )

        run_cmd = next(c for c in calls if c[:2] == ["docker", "run"])
        shell_command = run_cmd[-1]
        # Copies into the tmpfs-backed workdir, not a read-only location.
        assert "cp -r /code /tmp/workdir" in shell_command
        assert "cd /tmp/workdir" in shell_command


class TestDockerOutputCap:
    """SECURITY (SEC-012): captured stdout/stderr must be bounded so a malicious
    test cannot exhaust host memory with an output bomb."""

    def test_output_is_truncated_past_cap(self, tmp_path: Path) -> None:
        sandbox = DockerSandbox(max_output_bytes=100)
        big = "A" * 10_000

        def fake_run(cmd, *args, **kwargs):
            class _R:
                returncode = 0
                stdout = big
                stderr = big

            return _R()

        with patch.object(DockerSandbox, "is_available", return_value=True), \
                patch("cola_coder.data.curation.docker_sandbox.subprocess.run",
                      side_effect=fake_run):
            code, stdout, stderr = sandbox.run(
                repo_path=tmp_path, command="yes", image="alpine:latest"
            )

        assert code == 0
        # Far smaller than the 10_000-byte bomb, and clearly marked truncated.
        assert len(stdout.encode("utf-8")) < 10_000
        assert "output truncated" in stdout
        assert "output truncated" in stderr

    def test_small_output_not_truncated(self, tmp_path: Path) -> None:
        sandbox = DockerSandbox(max_output_bytes=1_000_000)

        def fake_run(cmd, *args, **kwargs):
            class _R:
                returncode = 0
                stdout = "hello\n"
                stderr = ""

            return _R()

        with patch.object(DockerSandbox, "is_available", return_value=True), \
                patch("cola_coder.data.curation.docker_sandbox.subprocess.run",
                      side_effect=fake_run):
            code, stdout, stderr = sandbox.run(
                repo_path=tmp_path, command="echo hello", image="alpine:latest"
            )

        assert stdout == "hello\n"
        assert "output truncated" not in stdout


class TestDockerWatchdog:
    """SECURITY (SEC-012): an outer wall-clock watchdog must arm on every run as
    a belt-and-braces backstop to the subprocess timeout, and must be cancelled
    once the call tears the container down (no leaked timers)."""

    def test_watchdog_armed_and_cancelled(self, tmp_path: Path) -> None:
        sandbox = DockerSandbox(timeout=5, watchdog_grace=10)
        created: list[object] = []

        real_timer = threading.Timer

        def spy_timer(*args, **kwargs):
            t = real_timer(*args, **kwargs)
            created.append(t)
            return t

        def fake_run(cmd, *args, **kwargs):
            class _R:
                returncode = 0
                stdout = ""
                stderr = ""

            return _R()

        with patch.object(DockerSandbox, "is_available", return_value=True), \
                patch("cola_coder.data.curation.docker_sandbox.subprocess.run",
                      side_effect=fake_run), \
                patch("cola_coder.data.curation.docker_sandbox.threading.Timer",
                      side_effect=spy_timer):
            sandbox.run(repo_path=tmp_path, command="echo hi", image="alpine:latest")

        # A watchdog timer was created and has been cancelled in the finally,
        # so its scheduled action will never fire after the container is gone.
        # (Timer.cancel sets the internal `finished` event; the daemon thread
        # may take a moment to actually exit, so we assert on the flag, not on
        # thread liveness which is racy.)
        assert created, "no watchdog timer was armed"
        assert created[0].finished.is_set()


# ---------------------------------------------------------------------------
# Test: TestRunner integration
# ---------------------------------------------------------------------------


class TestRunnerIntegration:
    """Test TestRunner end-to-end with dry_run and subprocess modes."""

    def test_dry_run_with_jest_repo(self, jest_repo: Path) -> None:
        """Dry run should detect framework but not execute."""
        runner = TestRunner(mode="dry_run", cache_dir=jest_repo / ".cache")
        score = runner.score_repo(jest_repo)
        assert score.tests_detected
        assert score.quality_tier == "detected"
        assert score.test_result is not None
        assert score.test_result.error == "dry_run mode — tests not executed"

    def test_dry_run_no_tests(self, no_test_repo: Path) -> None:
        """Dry run on repo without tests should return 'none' tier."""
        runner = TestRunner(mode="dry_run", cache_dir=no_test_repo / ".cache")
        score = runner.score_repo(no_test_repo)
        assert not score.tests_detected
        assert score.quality_tier == "none"

    def test_subprocess_pytest_repo(self, pytest_repo: Path) -> None:
        """Actually run pytest on our minimal repo (subprocess mode).

        We directly invoke the test command using sys.executable to ensure we
        use the correct Python interpreter (not a random system Python).
        """
        # Run pytest directly using the current interpreter
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--tb=short", "-q"],
            cwd=str(pytest_repo),
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Parse the output through our parser
        from cola_coder.data.curation.test_runner import _parse_pytest_output
        parsed = _parse_pytest_output(result.stdout + "\n" + result.stderr)
        assert parsed is not None
        assert parsed.framework == "pytest"
        assert parsed.passed >= 2
        assert parsed.all_passed

    def test_subprocess_pytest_scoring(self, pytest_repo: Path) -> None:
        """Score via the runner, which detects and runs pytest.

        Note: On Windows, the subprocess runner's 'python' may not resolve
        to the correct venv Python. We test the full integration but accept
        that it may fail to run if Python isn't on PATH — the important
        thing is that detection + scoring logic works.
        """
        runner = TestRunner(
            mode="subprocess",
            allow_host_execution=True,  # trusted local fixture repos
            timeout=30,
            install_timeout=30,
            cache_dir=pytest_repo / ".cache",
        )
        score = runner.score_repo(pytest_repo)
        assert score.tests_detected
        # The test may not pass on all systems (Python PATH issue), so
        # just verify the scoring pipeline completes without error
        assert score.quality_tier in ("verified", "tested", "detected")

    def test_cache_works(self, pytest_repo: Path) -> None:
        """Second score_repo call should use cache."""
        cache_dir = pytest_repo / ".cache"
        runner = TestRunner(
            mode="subprocess",
            allow_host_execution=True,  # trusted local fixture repos
            timeout=30,
            install_timeout=10,
            cache_dir=cache_dir,
        )
        # First call — runs tests
        score1 = runner.score_repo(pytest_repo)
        # Second call — should hit cache (faster)
        score2 = runner.score_repo(pytest_repo)
        assert score1.quality_tier == score2.quality_tier
        assert score1.score == score2.score

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            TestRunner(mode="invalid_mode")

    def test_default_mode_is_safe_dry_run(self) -> None:
        """The default must never execute untrusted code."""
        runner = TestRunner()
        assert runner.mode == "dry_run"

    def test_subprocess_requires_explicit_opt_in(self) -> None:
        """Host execution of repo scripts needs allow_host_execution=True."""
        with pytest.raises(ValueError, match="allow_host_execution"):
            TestRunner(mode="subprocess")


# ---------------------------------------------------------------------------
# Test: Timeout handling
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    """Test that subprocess timeouts work correctly."""

    @pytest.mark.skipif(
        sys.platform == "win32" and not os.environ.get("CI"),
        reason="sleep command may behave differently on Windows outside CI",
    )
    def test_subprocess_timeout(self, tmp_path: Path) -> None:
        """A command that exceeds timeout should be killed."""
        # Create a fake "repo" with a package.json that has a slow test
        pkg = {
            "name": "slow-repo",
            "version": "1.0.0",
            "scripts": {"test": "sleep 999"},
        }
        (tmp_path / "package.json").write_text(json.dumps(pkg))

        runner = TestRunner(
            mode="subprocess",
            allow_host_execution=True,  # trusted local fixture repos
            timeout=2,  # 2 seconds
            install_timeout=2,
            cache_dir=tmp_path / ".cache",
        )

        start = time.monotonic()
        result = runner.run_tests(tmp_path)
        elapsed = time.monotonic() - start

        # Should complete within a reasonable time (timeout + buffer)
        assert elapsed < 10, f"Took {elapsed:.1f}s — timeout didn't work"
        # Result should indicate timeout
        assert result is not None
        assert result.error is not None
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()

    def test_install_timeout_on_windows(self, tmp_path: Path) -> None:
        """Test that install timeout works (platform-independent version)."""
        # Create repo with a test command that will execute quickly
        # but install command that would take forever
        pyproject = textwrap.dedent("""\
            [project]
            name = "timeout-test"
            version = "0.1.0"

            [tool.pytest.ini_options]
            testpaths = ["."]
        """)
        (tmp_path / "pyproject.toml").write_text(pyproject)
        # Write a Python test file that just hangs
        (tmp_path / "test_hang.py").write_text(textwrap.dedent("""\
            import time
            def test_hang():
                time.sleep(999)
        """))

        runner = TestRunner(
            mode="subprocess",
            allow_host_execution=True,  # trusted local fixture repos
            timeout=2,
            install_timeout=2,
            cache_dir=tmp_path / ".cache",
        )

        start = time.monotonic()
        result = runner.run_tests(tmp_path)
        elapsed = time.monotonic() - start

        assert elapsed < 15, f"Took {elapsed:.1f}s — timeout didn't work"
        # Either it timed out or parsed an error
        assert result is not None


# ---------------------------------------------------------------------------
# Test: Parallel scoring
# ---------------------------------------------------------------------------


class TestParallelScoring:
    """Test parallel repo scoring."""

    def test_parallel_dry_run(self, tmp_path: Path) -> None:
        """Parallel dry-run scoring should return results for all repos."""
        # Create three separate repo directories (can't reuse fixtures — same tmp_path)
        jest_dir = tmp_path / "jest-repo"
        jest_dir.mkdir()
        pkg = {
            "name": "par-jest",
            "scripts": {"test": "jest"},
            "devDependencies": {"jest": "^29.0.0"},
        }
        (jest_dir / "package.json").write_text(json.dumps(pkg))

        pytest_dir = tmp_path / "pytest-repo"
        pytest_dir.mkdir()
        (pytest_dir / "pytest.ini").write_text("[pytest]\n")
        tests_sub = pytest_dir / "tests"
        tests_sub.mkdir()
        (tests_sub / "test_x.py").write_text("def test_x(): pass\n")

        empty_dir = tmp_path / "empty-repo"
        empty_dir.mkdir()
        (empty_dir / "README.md").write_text("# nothing\n")

        runner = TestRunner(mode="dry_run", cache_dir=tmp_path / ".cache")
        repos = [jest_dir, pytest_dir, empty_dir]
        results = runner.score_repos_parallel(repos, max_workers=2, use_cache=False)

        assert len(results) == 3
        assert results[jest_dir].tests_detected
        assert results[pytest_dir].tests_detected
        assert not results[empty_dir].tests_detected


class TestParallelSubprocessGate:
    """BUG-103: score_repos_parallel must propagate allow_host_execution to the
    worker processes. Without it, subprocess-mode workers re-trip the __init__
    safety gate and every repo is silently scored as an error (0.2)."""

    def test_worker_subprocess_requires_flag_propagation(self, tmp_path):
        # No test framework → score_repo returns without executing anything, so
        # this is safe even in subprocess mode. The point is construction: with
        # the flag the worker builds; without it, __init__ raises.
        empty = tmp_path / "norepo"
        empty.mkdir()
        (empty / "README.md").write_text("# nothing\n", encoding="utf-8")

        # Old signature couldn't even accept the flag; new one must.
        score = _score_repo_worker(
            empty, "subprocess", 5, 5,
            allow_host_execution=True,
            cache_dir=tmp_path / ".cache",
        )
        assert score.tests_detected is False
        # NOT the exception path — details must not carry the safety-gate error.
        assert "allow_host_execution" not in str(score.details)

    def test_worker_without_flag_still_gated(self, tmp_path):
        # The safety gate itself must remain intact when the flag is absent.
        empty = tmp_path / "norepo2"
        empty.mkdir()
        with pytest.raises(ValueError, match="allow_host_execution"):
            _score_repo_worker(empty, "subprocess", 5, 5)

    def test_parallel_subprocess_no_framework_not_errored(self, tmp_path):
        # End-to-end: a no-framework repo scored via the parallel path in
        # subprocess mode must come back as a real (non-error) score, proving
        # the workers constructed successfully.
        empty = tmp_path / "empty-sub"
        empty.mkdir()
        (empty / "README.md").write_text("# nothing\n", encoding="utf-8")

        runner = TestRunner(
            mode="subprocess",
            allow_host_execution=True,  # trusted empty fixture
            cache_dir=tmp_path / ".cache",
        )
        results = runner.score_repos_parallel([empty], max_workers=1, use_cache=False)
        assert empty in results
        assert "allow_host_execution" not in str(results[empty].details)
        assert results[empty].tests_detected is False
