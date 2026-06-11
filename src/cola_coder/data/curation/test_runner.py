"""Test runner: detect frameworks and execute test suites.

Supports:
    Node.js/TypeScript: jest, vitest, mocha (detected from package.json)
    Python: pytest, unittest (detected from pyproject.toml, setup.cfg, pytest.ini)
    Go: go test (detected from go.mod)
    Rust: cargo test (detected from Cargo.toml)

Execution modes:
    docker     — Safest. Runs in isolated Docker container (requires Docker).
    subprocess — Faster, less safe. For trusted repos only.
    dry_run    — Just detect test framework, don't execute.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from cola_coder.data.curation.docker_sandbox import DockerSandbox
from cola_coder.data.curation.test_scorer import RepoScore, TestResult, TestScorer

logger = logging.getLogger(__name__)


@dataclass
class TestFramework:
    """Detected test framework for a repository.

    Commands are stored as lists (safe for subprocess without shell=True).
    The shell_cmd properties join them for Docker's ``sh -c`` execution.
    """

    name: str          # "jest", "vitest", "mocha", "pytest", "unittest", "go_test", "cargo_test"
    language: str      # "node", "python", "go", "rust"
    install_args: list[str]   # ["npm", "install", "--ignore-scripts"]
    test_args: list[str]      # ["npx", "jest", "--json", "--forceExit"]
    coverage_args: list[str] | None = None

    @property
    def install_cmd(self) -> str:
        """Shell string form for Docker execution."""
        return shlex.join(self.install_args)

    @property
    def test_cmd(self) -> str:
        """Shell string form for Docker execution."""
        return shlex.join(self.test_args)

    @property
    def coverage_cmd(self) -> str | None:
        """Shell string form for Docker execution."""
        return shlex.join(self.coverage_args) if self.coverage_args else None

    def __str__(self) -> str:
        return f"{self.name} ({self.language})"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _repo_cache_key(repo_path: Path) -> str:
    """Generate a cache key from the repo path and latest commit (if git)."""
    repo_str = str(repo_path.resolve())
    # Try to get git commit hash for more precise caching
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            repo_str += f"@{result.stdout.strip()}"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return hashlib.sha256(repo_str.encode()).hexdigest()[:16]


class _ResultCache:
    """Simple JSON file cache for test results."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> RepoScore | None:
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                return RepoScore.from_dict(data)
            except (json.JSONDecodeError, KeyError, TypeError):
                return None
        return None

    def put(self, key: str, score: RepoScore) -> None:
        cache_file = self.cache_dir / f"{key}.json"
        cache_file.write_text(
            json.dumps(score.to_dict(), indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Framework detection
# ---------------------------------------------------------------------------

def _detect_node_framework(repo_path: Path) -> TestFramework | None:
    """Detect Node.js test framework from package.json."""
    pkg_json = repo_path / "package.json"
    if not pkg_json.exists():
        return None

    try:
        pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    scripts = pkg.get("scripts", {})
    test_script = scripts.get("test", "")
    dev_deps = pkg.get("devDependencies", {})
    deps = pkg.get("dependencies", {})
    all_deps = {**deps, **dev_deps}

    # Detect vitest (check first — vitest configs often include "vitest" in test script)
    if "vitest" in test_script or "vitest" in all_deps:
        return TestFramework(
            name="vitest",
            language="node",
            install_args=["npm", "install", "--ignore-scripts"],
            test_args=["npx", "vitest", "run", "--reporter=json"],
            coverage_args=["npx", "vitest", "run", "--coverage", "--reporter=json"],
        )

    # Detect jest
    if "jest" in test_script or "jest" in all_deps:
        return TestFramework(
            name="jest",
            language="node",
            install_args=["npm", "install", "--ignore-scripts"],
            test_args=["npx", "jest", "--json", "--forceExit"],
            coverage_args=["npx", "jest", "--coverage", "--json", "--forceExit"],
        )

    # Detect mocha
    if "mocha" in test_script or "mocha" in all_deps:
        return TestFramework(
            name="mocha",
            language="node",
            install_args=["npm", "install", "--ignore-scripts"],
            test_args=["npx", "mocha", "--reporter", "json"],
        )

    # Generic npm test (if test script exists and isn't the default "echo error")
    # WARNING: npm test executes the "test" script from package.json verbatim.
    # --ignore-scripts only protects install lifecycle hooks, NOT the test script.
    # Only use subprocess mode with trusted repos; prefer docker mode for scraped data.
    if test_script and "no test specified" not in test_script:
        return TestFramework(
            name="npm_test",
            language="node",
            install_args=["npm", "install", "--ignore-scripts"],
            test_args=["npm", "test"],
        )

    return None


def _detect_python_framework(repo_path: Path) -> TestFramework | None:
    """Detect Python test framework from config files."""
    # Check for pytest indicators
    pytest_indicators = [
        repo_path / "pytest.ini",
        repo_path / "conftest.py",
    ]

    # Check pyproject.toml for [tool.pytest]
    pyproject = repo_path / "pyproject.toml"
    has_pytest_config = False
    if pyproject.exists():
        try:
            content = pyproject.read_text(encoding="utf-8")
            if "[tool.pytest" in content or "pytest" in content.lower():
                has_pytest_config = True
        except UnicodeDecodeError:
            pass

    # Check setup.cfg for [tool:pytest]
    setup_cfg = repo_path / "setup.cfg"
    if setup_cfg.exists():
        try:
            content = setup_cfg.read_text(encoding="utf-8")
            if "[tool:pytest]" in content:
                has_pytest_config = True
        except UnicodeDecodeError:
            pass

    if any(p.exists() for p in pytest_indicators) or has_pytest_config:
        return TestFramework(
            name="pytest",
            language="python",
            install_args=["pip", "install", "-e", ".[dev]"],
            test_args=["python", "-m", "pytest", "--tb=short", "-q"],
            coverage_args=[
                "python", "-m", "pytest", "--cov", "--cov-report=term-missing",
                "--tb=short", "-q",
            ],
        )

    # Check for unittest (look for test_*.py files)
    test_files = list(repo_path.glob("test_*.py")) + list(
        repo_path.glob("tests/test_*.py")
    )
    if test_files:
        return TestFramework(
            name="unittest",
            language="python",
            install_args=["pip", "install", "-e", "."],
            test_args=["python", "-m", "unittest", "discover", "-s", ".", "-p",
                        "test_*.py"],
        )

    return None


def _detect_go_framework(repo_path: Path) -> TestFramework | None:
    """Detect Go test framework from go.mod."""
    if (repo_path / "go.mod").exists():
        return TestFramework(
            name="go_test",
            language="go",
            install_args=["go", "mod", "download"],
            test_args=["go", "test", "./...", "-json"],
            coverage_args=["go", "test", "./...", "-coverprofile=coverage.out",
                           "-json"],
        )
    return None


def _detect_rust_framework(repo_path: Path) -> TestFramework | None:
    """Detect Rust test framework from Cargo.toml."""
    if (repo_path / "Cargo.toml").exists():
        return TestFramework(
            name="cargo_test",
            language="rust",
            install_args=["cargo", "fetch"],
            test_args=["cargo", "test"],
        )
    return None


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def _parse_jest_output(stdout: str) -> TestResult | None:
    """Parse Jest JSON output into TestResult."""
    # Jest --json outputs a JSON blob; find it in the output
    try:
        # Look for the JSON object in stdout
        json_match = re.search(r'\{[\s\S]*"numTotalTests"[\s\S]*\}', stdout)
        if not json_match:
            return None
        data = json.loads(json_match.group())
        return TestResult(
            framework="jest",
            total_tests=data.get("numTotalTests", 0),
            passed=data.get("numPassedTests", 0),
            failed=data.get("numFailedTests", 0),
            skipped=data.get("numPendingTests", 0),
        )
    except (json.JSONDecodeError, KeyError):
        return None


def _parse_vitest_output(stdout: str) -> TestResult | None:
    """Parse Vitest JSON output into TestResult."""
    try:
        json_match = re.search(r'\{[\s\S]*"numTotalTests"[\s\S]*\}', stdout)
        if not json_match:
            return None
        data = json.loads(json_match.group())
        return TestResult(
            framework="vitest",
            total_tests=data.get("numTotalTests", 0),
            passed=data.get("numPassedTests", 0),
            failed=data.get("numFailedTests", 0),
            skipped=data.get("numPendingTests", 0) + data.get("numTodoTests", 0),
        )
    except (json.JSONDecodeError, KeyError):
        return None


def _parse_pytest_output(stdout: str) -> TestResult | None:
    """Parse pytest output (short format) into TestResult."""
    # Look for summary line like: "5 passed, 1 failed, 2 skipped in 1.23s"
    # or: "=== 5 passed in 1.23s ==="
    passed = failed = skipped = errors = 0

    # Pattern: "N passed"
    m = re.search(r"(\d+)\s+passed", stdout)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+)\s+failed", stdout)
    if m:
        failed = int(m.group(1))
    m = re.search(r"(\d+)\s+skipped", stdout)
    if m:
        skipped = int(m.group(1))
    m = re.search(r"(\d+)\s+error", stdout)
    if m:
        errors = int(m.group(1))

    total = passed + failed + skipped + errors
    if total == 0:
        return None

    # Try to get coverage from output
    coverage = None
    cov_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", stdout)
    if cov_match:
        coverage = int(cov_match.group(1)) / 100.0

    return TestResult(
        framework="pytest",
        total_tests=total,
        passed=passed,
        failed=failed + errors,
        skipped=skipped,
        coverage=coverage,
    )


def _parse_go_test_output(stdout: str) -> TestResult | None:
    """Parse go test -json output into TestResult."""
    passed = failed = skipped = 0
    try:
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            action = event.get("Action", "")
            test_name = event.get("Test", "")
            if not test_name:
                continue
            if action == "pass":
                passed += 1
            elif action == "fail":
                failed += 1
            elif action == "skip":
                skipped += 1
    except Exception:
        return None

    total = passed + failed + skipped
    if total == 0:
        return None

    return TestResult(
        framework="go_test",
        total_tests=total,
        passed=passed,
        failed=failed,
        skipped=skipped,
    )


def _parse_generic_output(stdout: str, stderr: str, exit_code: int) -> TestResult | None:
    """Best-effort parse of generic test output."""
    combined = stdout + "\n" + stderr

    # Try pytest-style first
    result = _parse_pytest_output(combined)
    if result:
        return result

    # Try jest-style
    result = _parse_jest_output(combined)
    if result:
        return result

    # Last resort: just look for pass/fail keywords
    pass_count = len(re.findall(r"(?:PASS|passing|passed|ok)", combined, re.IGNORECASE))
    fail_count = len(re.findall(r"(?:FAIL|failing|failed|error)", combined, re.IGNORECASE))

    if pass_count + fail_count > 0:
        return TestResult(
            framework="unknown",
            total_tests=pass_count + fail_count,
            passed=pass_count,
            failed=fail_count,
            skipped=0,
            error=f"exit_code={exit_code}" if exit_code != 0 else None,
        )

    return None


def _parse_output(framework_name: str, stdout: str, stderr: str, exit_code: int) -> TestResult | None:
    """Route to the correct parser based on framework."""
    parsers = {
        "jest": _parse_jest_output,
        "vitest": _parse_vitest_output,
        "pytest": _parse_pytest_output,
        "go_test": _parse_go_test_output,
    }
    combined = stdout + "\n" + stderr

    parser = parsers.get(framework_name)
    if parser:
        result = parser(combined)
        if result:
            return result

    # Fallback to generic parsing
    return _parse_generic_output(stdout, stderr, exit_code)


# ---------------------------------------------------------------------------
# Main TestRunner class
# ---------------------------------------------------------------------------

class TestRunner:
    """Run test suites in sandboxed environments.

    Supports three execution modes:
        docker     — Runs in an isolated Docker container (safest)
        subprocess — Runs directly via subprocess (faster, for trusted repos)
        dry_run    — Only detects framework, doesn't execute tests

    Usage:
        runner = TestRunner(mode="subprocess", timeout=300)
        framework = runner.detect_test_framework(Path("./my-repo"))
        result = runner.run_tests(Path("./my-repo"))
        score = runner.score_repo(Path("./my-repo"))
    """

    DETECTORS = [
        _detect_node_framework,
        _detect_python_framework,
        _detect_go_framework,
        _detect_rust_framework,
    ]

    def __init__(
        self,
        mode: str = "dry_run",
        timeout: int = 300,
        install_timeout: int = 120,
        cache_dir: Path | None = None,
        allow_host_execution: bool = False,
    ):
        """Initialize the test runner.

        Args:
            mode: Execution mode — "docker", "subprocess", or "dry_run".
                Defaults to the safe "dry_run" (framework detection only) —
                running scraped repos' install/test scripts executes
                arbitrary third-party code.
            timeout: Max seconds for running tests.
            install_timeout: Max seconds for installing dependencies.
            cache_dir: Directory for caching results. Defaults to data/test_cache/.
            allow_host_execution: Required to use "subprocess" mode. The
                explicit flag prevents host execution of untrusted code from
                being reachable by accident (e.g. a forgotten mode kwarg).
        """
        if mode not in ("docker", "subprocess", "dry_run"):
            raise ValueError(f"Invalid mode: {mode!r}. Use 'docker', 'subprocess', or 'dry_run'.")

        if mode == "subprocess" and not allow_host_execution:
            raise ValueError(
                "mode='subprocess' runs repo install/test scripts directly on "
                "this machine — that is arbitrary code execution for scraped "
                "repos. Pass allow_host_execution=True only for repos you "
                "trust, or use mode='docker' for isolation."
            )

        self.mode = mode
        self.timeout = timeout
        self.install_timeout = install_timeout
        # Stored so score_repos_parallel can faithfully reconstruct this runner
        # in each worker process — without propagating allow_host_execution the
        # subprocess-mode workers would re-trip the safety gate above and every
        # repo would silently score as an error.
        self._allow_host_execution = allow_host_execution

        if mode == "subprocess":
            logger.warning(
                "TestRunner using 'subprocess' mode — only safe for trusted local repos. "
                "Use 'docker' mode when scoring repos cloned from GitHub."
            )
        self._scorer = TestScorer()

        # Set up cache
        if cache_dir is None:
            cache_dir = Path("data/test_cache")
        self._cache_dir = Path(cache_dir)
        self._cache = _ResultCache(cache_dir)

        # Check Docker availability if docker mode requested
        if mode == "docker":
            if not DockerSandbox.is_available():
                raise RuntimeError(
                    "Docker is not available but mode='docker' was requested. "
                    "Install and start Docker, or explicitly pass mode='subprocess' "
                    "(only safe for trusted local repos, NOT for cloned GitHub repos)."
                )
            self._sandbox = DockerSandbox(timeout=timeout)

    def detect_test_framework(self, repo_path: Path) -> TestFramework | None:
        """Detect which test framework a repo uses.

        Checks for package.json (jest/vitest/mocha), pyproject.toml/pytest.ini
        (pytest), go.mod (go test), Cargo.toml (cargo test).

        Returns:
            TestFramework if detected, None otherwise.
        """
        repo_path = Path(repo_path)
        if not repo_path.is_dir():
            return None

        for detector in self.DETECTORS:
            framework = detector(repo_path)
            if framework is not None:
                logger.info("Detected %s in %s", framework, repo_path)
                return framework

        return None

    def run_tests(self, repo_path: Path, use_cache: bool = True) -> TestResult | None:
        """Run detected test suite and return results.

        Args:
            repo_path: Path to the repository.
            use_cache: Whether to check/use cached results.

        Returns:
            TestResult if tests ran, None if no tests detected or couldn't run.
        """
        repo_path = Path(repo_path)
        framework = self.detect_test_framework(repo_path)
        if framework is None:
            return None

        if self.mode == "dry_run":
            # Return a placeholder result indicating tests were detected
            return TestResult(
                framework=framework.name,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                error="dry_run mode — tests not executed",
            )

        # Execute tests
        if self.mode == "docker":
            return self._run_docker(repo_path, framework)
        else:
            return self._run_subprocess(repo_path, framework)

    def score_repo(self, repo_path: Path, use_cache: bool = True) -> RepoScore:
        """Score a repo based on test results + coverage.

        Args:
            repo_path: Path to the repository.
            use_cache: Whether to check/use cached results.

        Returns:
            RepoScore with quality tier and composite score.
        """
        repo_path = Path(repo_path)

        # Check cache
        if use_cache:
            cache_key = _repo_cache_key(repo_path)
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info("Cache hit for %s", repo_path)
                return cached

        framework = self.detect_test_framework(repo_path)
        tests_detected = framework is not None

        result = self.run_tests(repo_path, use_cache=False)
        score = self._scorer.score(result, tests_detected=tests_detected)

        # Cache the result
        if use_cache:
            cache_key = _repo_cache_key(repo_path)
            self._cache.put(cache_key, score)

        return score

    def score_repos_parallel(
        self,
        repo_paths: list[Path],
        max_workers: int = 4,
        use_cache: bool = True,
    ) -> dict[Path, RepoScore]:
        """Score multiple repos in parallel using ProcessPoolExecutor.

        Args:
            repo_paths: List of repo paths to score.
            max_workers: Number of parallel workers.
            use_cache: Whether to use cached results.

        Returns:
            Dict mapping repo path to its score.
        """
        results: dict[Path, RepoScore] = {}

        # For dry_run mode, we can use threads (no subprocess overhead)
        # For subprocess/docker, use process pool to avoid GIL
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {}
            for path in repo_paths:
                # Check cache first to avoid unnecessary work
                if use_cache:
                    cache_key = _repo_cache_key(path)
                    cached = self._cache.get(cache_key)
                    if cached is not None:
                        results[path] = cached
                        continue

                future = executor.submit(
                    _score_repo_worker, path, self.mode, self.timeout,
                    self.install_timeout, self._allow_host_execution,
                    str(self._cache_dir),
                )
                future_to_path[future] = path

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    score = future.result(timeout=self.timeout + 60)
                    results[path] = score
                    # Cache the result
                    if use_cache:
                        cache_key = _repo_cache_key(path)
                        self._cache.put(cache_key, score)
                except Exception as exc:
                    logger.error("Failed to score %s: %s", path, exc)
                    results[path] = RepoScore(
                        tests_detected=False,
                        tests_pass=False,
                        test_result=None,
                        quality_tier="none",
                        score=0.2,
                        details={"error": str(exc)},
                    )

        return results

    # -----------------------------------------------------------------------
    # Private execution methods
    # -----------------------------------------------------------------------

    def _run_subprocess(self, repo_path: Path, framework: TestFramework) -> TestResult | None:
        """Run tests via subprocess (no Docker).

        Uses list-form commands (no shell=True) to prevent shell injection
        from malicious package.json or config files in cloned repos.
        """
        import time

        start = time.monotonic()

        # Step 1: Install dependencies (with timeout)
        try:
            install_result = subprocess.run(
                framework.install_args,
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=self.install_timeout,
            )
            if install_result.returncode != 0:
                logger.warning(
                    "Install failed for %s (exit %d): %s",
                    repo_path, install_result.returncode,
                    install_result.stderr[:500],
                )
                # Don't bail — some repos work without explicit install
        except subprocess.TimeoutExpired:
            logger.warning("Install timed out after %ds for %s",
                           self.install_timeout, repo_path)
            return TestResult(
                framework=framework.name,
                total_tests=0, passed=0, failed=0, skipped=0,
                error=f"Install timed out after {self.install_timeout}s",
            )

        # Step 2: Run tests (with timeout)
        try:
            test_proc = subprocess.run(
                framework.test_args,
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Tests timed out after %ds for %s", self.timeout, repo_path)
            return TestResult(
                framework=framework.name,
                total_tests=0, passed=0, failed=0, skipped=0,
                error=f"Tests timed out after {self.timeout}s",
            )

        elapsed = time.monotonic() - start

        # Parse results
        result = _parse_output(
            framework.name,
            test_proc.stdout or "",
            test_proc.stderr or "",
            test_proc.returncode,
        )

        if result is not None:
            result.duration_seconds = elapsed
        else:
            # Could not parse output — create minimal result from exit code
            result = TestResult(
                framework=framework.name,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                error=f"Could not parse test output (exit_code={test_proc.returncode})",
                duration_seconds=elapsed,
            )

        return result

    def _run_docker(self, repo_path: Path, framework: TestFramework) -> TestResult | None:
        """Run tests inside a Docker container."""
        import time

        start = time.monotonic()
        image = DockerSandbox.image_for_language(framework.language)

        exit_code, stdout, stderr = self._sandbox.run_with_install(
            repo_path=repo_path,
            install_cmd=framework.install_cmd,
            test_cmd=framework.test_cmd,
            image=image,
            install_timeout=self.install_timeout,
            test_timeout=self.timeout,
        )

        elapsed = time.monotonic() - start

        result = _parse_output(framework.name, stdout, stderr, exit_code)
        if result is not None:
            result.duration_seconds = elapsed
        else:
            result = TestResult(
                framework=framework.name,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                error=f"Could not parse output (exit_code={exit_code})",
                duration_seconds=elapsed,
            )

        return result


# ---------------------------------------------------------------------------
# Worker function for parallel execution (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _score_repo_worker(
    repo_path: Path,
    mode: str,
    timeout: int,
    install_timeout: int,
    allow_host_execution: bool = False,
    cache_dir: str | Path | None = None,
) -> RepoScore:
    """Score a single repo. Used by ProcessPoolExecutor.

    ``allow_host_execution`` and ``cache_dir`` MUST be forwarded from the parent
    runner: subprocess mode re-trips the __init__ safety gate without the flag,
    and a hardcoded cache_dir would ignore the caller's configured location.
    """
    runner = TestRunner(
        mode=mode,
        timeout=timeout,
        install_timeout=install_timeout,
        cache_dir=Path(cache_dir) if cache_dir is not None else Path("data/test_cache"),
        allow_host_execution=allow_host_execution,
    )
    return runner.score_repo(repo_path, use_cache=False)
