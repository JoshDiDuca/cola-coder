# Research: Test-Driven Data Curation

## Status: Novel — Nobody has done this systematically

## The Idea

What if we scored training data quality by whether the code ACTUALLY WORKS?

Most data curation is static analysis: parse checks, lint scores, dedup. But the
ultimate quality signal is: **does this code produce correct behavior?**

## The Approach

For repos that include tests:

1. Clone the repo
2. Install dependencies
3. Run the test suite
4. If tests pass: this repo has VERIFIED working code
5. Extract code + test pairs as high-quality training data
6. Weight these files 2-5x higher in training

### Why This Is Revolutionary

Current data quality approaches:
- **Static**: Parse check, lint, pattern matching → catches syntax errors
- **Heuristic**: Star count, fork count, recency → proxy for quality
- **LLM judge**: GPT/Claude rates code → expensive, subjective

This approach:
- **Dynamic**: Does the code actually run and produce correct output?
- **Objective**: Tests pass or they don't — no ambiguity
- **Teaches correctness**: Model learns code that is PROVEN to work

### Data Format: Code-Test Pairs

```yaml
# A single training example
code_file: "src/utils/validator.ts"
test_file: "tests/utils/validator.test.ts"
test_result: "pass"  # or "fail"
test_count: 12
pass_count: 12
coverage: 0.94  # 94% of code_file lines are executed by tests

# Training signal: this code is HIGH QUALITY because:
# 1. It has dedicated tests
# 2. All tests pass
# 3. Tests cover 94% of the code
```

### Training Strategy

Option A: **Weighted sampling** — Files from test-passing repos get 3x weight
Option B: **Curriculum learning** — Train on verified code first, noisy code second
Option C: **Code-test interleaving** — Train on [code → test] pairs so model learns
          what good tests look like AND what well-tested code looks like

### Implementation Plan

```python
class TestVerifiedSource(DataSource):
    """Clone repos, run tests, extract verified code."""

    def __init__(
        self,
        repos: list[str],
        languages: list[str],
        timeout_per_repo: int = 300,  # 5 min max per repo
        docker: bool = True,  # Run in container for safety
    ):
        ...

    def stream(self) -> Iterator[DataRecord]:
        for repo in self.repos:
            result = self._clone_and_test(repo)
            if result.tests_pass:
                for file_record in self._extract_files(result):
                    file_record.metadata["test_verified"] = True
                    file_record.metadata["test_coverage"] = result.coverage
                    file_record.metadata["quality_tier"] = "verified"
                    yield file_record

    def _clone_and_test(self, repo: str) -> TestResult:
        """Clone repo, detect test framework, run tests.

        Detects:
        - Node.js: jest, vitest, mocha (look for test script in package.json)
        - Python: pytest, unittest (look for pytest.ini, setup.cfg)
        - Go: go test
        - Rust: cargo test

        Runs in Docker container for isolation. Timeout after 5 min.
        """
        ...
```

### Safety: Docker Isolation

Running arbitrary code from GitHub is dangerous. Every test run must be sandboxed:

```python
class DockerSandbox:
    """Run code in an isolated Docker container.

    - No network access (--network none)
    - Read-only filesystem except /tmp
    - Memory limit (2GB)
    - CPU limit (2 cores)
    - Timeout (5 minutes)
    - No host mounts
    """

    def run(self, repo_path: Path, command: str, timeout: int = 300) -> TestResult:
        # docker run --rm --network none --memory 2g --cpus 2
        #   -v {repo_path}:/code:ro -w /code
        #   node:20-slim npm test
        ...
```

### Scale Considerations

- **Small scale (100-1000 repos)**: Clone, test, extract locally. ~1-2 days.
- **Medium scale (10k repos)**: Parallelize across multiple Docker containers. ~1 week.
- **Large scale (100k+ repos)**: Use cloud CI (GitHub Actions, AWS CodeBuild). $$$

### Quality Tiers

Based on test verification, we can create quality tiers for data mixing:

| Tier | Criteria | Typical % of Data | Training Weight |
|------|----------|-------------------|----------------|
| Verified | Tests pass, >80% coverage | 5-10% | 3-5x |
| Tested | Has tests, some pass | 10-20% | 2x |
| Linted | Passes linter, valid syntax | 30-40% | 1x |
| Parsed | Valid AST, basic checks | 60-80% | 0.5x |
| Raw | Everything else | 100% | 0.2x |

### Connection to GRPO/Reasoning

This connects beautifully to the reasoning training (GRPO) already in the project:

1. **Pre-training**: Use test-verified data for higher quality base model
2. **GRPO fine-tuning**: Generate code, run tests, reward passing solutions
3. **Virtuous cycle**: Better base model → better GRPO starting point → better model

The test infrastructure built here (Docker sandbox, test runner) can be reused
directly as GRPO reward functions.

## Feasibility

| Aspect | Assessment |
|--------|-----------|
| Novelty | Very high — no public system does this at scale |
| Difficulty | Medium-high (Docker orchestration, framework detection) |
| Expected impact | High — verified code is objectively better training data |
| Cost | Moderate — compute for running tests, Docker overhead |
| Risk | Medium — many repos have flaky tests, version conflicts |

## Prior Art

- AlphaCode (2022): Generates code, runs tests, but only for evaluation, not data curation
- CodeContests (2022): Uses test cases for filtering, but only for competition problems
- APPS dataset (2021): Code + tests, but curated by hand, not auto-discovered
- **Nobody has used test execution as a DATA QUALITY signal for pre-training**
