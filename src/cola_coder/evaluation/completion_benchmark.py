"""Code Completion Benchmark: evaluate model on prefix-based code completion.

Unlike HumanEval (which generates from docstrings), this benchmark gives the
model a partial function and asks it to complete it.  Scoring checks for
required patterns (identifiers, keywords, return statements) in the output.

For a TS dev: think of it like a Jest suite where each test fixture is a
half-written function and the assertion checks the AI's continuation.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CompletionProblem:
    """A single code-completion problem."""

    task_id: str
    prefix: str  # The code the model sees (truncated mid-function)
    required_patterns: list[str]  # Regexes that must match in the completion
    forbidden_patterns: list[str] = field(default_factory=list)  # Must NOT match
    description: str = ""
    difficulty: str = "medium"  # easy | medium | hard
    category: str = "general"
    language: str = "python"


@dataclass
class CompletionResult:
    """Result for a single problem."""

    task_id: str
    completion: str
    passed: bool
    matched_patterns: list[str]
    missed_patterns: list[str]
    forbidden_found: list[str]
    latency_ms: float = 0.0


@dataclass
class BenchmarkReport:
    """Aggregated benchmark report."""

    total: int
    passed: int
    failed: int
    pass_rate: float
    results: list[CompletionResult]
    total_latency_ms: float = 0.0
    by_difficulty: dict[str, dict[str, int]] = field(default_factory=dict)
    by_category: dict[str, dict[str, int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 30 built-in problems
# ---------------------------------------------------------------------------

PROBLEMS: list[CompletionProblem] = [
    # ── Easy ─────────────────────────────────────────────────────────────────
    CompletionProblem(
        task_id="complete_add",
        description="Complete a simple add function",
        prefix="def add(a: int, b: int) -> int:\n    ",
        required_patterns=[r"return\s+a\s*\+\s*b"],
        difficulty="easy",
        category="math",
    ),
    CompletionProblem(
        task_id="complete_len_check",
        description="Complete a function that checks if a list is empty",
        prefix="def is_empty(lst: list) -> bool:\n    ",
        required_patterns=[r"return\s+(?:len\(lst\)\s*==\s*0|not\s+lst)"],
        difficulty="easy",
        category="list",
    ),
    CompletionProblem(
        task_id="complete_absolute",
        description="Complete an absolute value function",
        prefix="def absolute(x: float) -> float:\n    if x < 0:\n        ",
        required_patterns=[r"return\s+-\s*x|return\s+x\s*\*\s*-1"],
        difficulty="easy",
        category="math",
    ),
    CompletionProblem(
        task_id="complete_string_reverse",
        description="Complete a string-reversal function",
        prefix="def reverse_string(s: str) -> str:\n    ",
        required_patterns=[r"return\s+s\[::-1\]|return\s+reversed\b|''.join"],
        difficulty="easy",
        category="string",
    ),
    CompletionProblem(
        task_id="complete_max_of_two",
        description="Complete a function returning the larger of two numbers",
        prefix="def max_of_two(a, b):\n    if a > b:\n        ",
        required_patterns=[r"return\s+a"],
        difficulty="easy",
        category="math",
    ),
    CompletionProblem(
        task_id="complete_list_sum",
        description="Complete a function that sums a list",
        prefix="def list_sum(numbers: list[int]) -> int:\n    total = 0\n    for n in numbers:\n        ",
        required_patterns=[r"total\s*\+=\s*n"],
        difficulty="easy",
        category="list",
    ),
    CompletionProblem(
        task_id="complete_factorial_base",
        description="Complete factorial base case",
        prefix="def factorial(n: int) -> int:\n    if n <= 1:\n        ",
        required_patterns=[r"return\s+1"],
        difficulty="easy",
        category="math",
    ),
    CompletionProblem(
        task_id="complete_contains",
        description="Complete a membership-check function",
        prefix="def contains(lst: list, item) -> bool:\n    ",
        required_patterns=[r"return\s+item\s+in\s+lst|item\s+in\s+lst"],
        difficulty="easy",
        category="list",
    ),
    CompletionProblem(
        task_id="complete_greet",
        description="Complete a greeting function",
        prefix='def greet(name: str) -> str:\n    ',
        required_patterns=[r"return\s+f?[\"'].*{?name}?.*[\"']|return\s+\"Hello"],
        difficulty="easy",
        category="string",
    ),
    CompletionProblem(
        task_id="complete_square",
        description="Complete a squaring function",
        prefix="def square(x: float) -> float:\n    ",
        required_patterns=[r"return\s+x\s*\*\*\s*2|return\s+x\s*\*\s*x"],
        difficulty="easy",
        category="math",
    ),
    # ── Medium ────────────────────────────────────────────────────────────────
    CompletionProblem(
        task_id="complete_fibonacci",
        description="Complete Fibonacci with memoization",
        prefix=(
            "def fibonacci(n: int, memo: dict | None = None) -> int:\n"
            "    if memo is None:\n"
            "        memo = {}\n"
            "    if n in memo:\n"
            "        return memo[n]\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    "
        ),
        # Require the recursive call — these cannot appear in the prefix above
        required_patterns=[r"fibonacci\(n\s*-\s*1\)", r"fibonacci\(n\s*-\s*2\)"],
        difficulty="medium",
        category="math",
    ),
    CompletionProblem(
        task_id="complete_binary_search",
        description="Complete binary search loop body",
        prefix=(
            "def binary_search(arr: list[int], target: int) -> int:\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            "
        ),
        required_patterns=[r"lo\s*=\s*mid\s*\+\s*1"],
        difficulty="medium",
        category="algorithm",
    ),
    CompletionProblem(
        task_id="complete_flatten",
        description="Complete a list-flattening function",
        prefix=(
            "def flatten(nested: list) -> list:\n"
            "    result = []\n"
            "    for item in nested:\n"
            "        if isinstance(item, list):\n"
            "            "
        ),
        required_patterns=[r"result\.extend|flatten\(item\)"],
        difficulty="medium",
        category="list",
    ),
    CompletionProblem(
        task_id="complete_count_words",
        description="Complete a word-count function",
        prefix=(
            "def count_words(text: str) -> dict[str, int]:\n"
            "    counts: dict[str, int] = {}\n"
            "    for word in text.split():\n"
            "        word = word.lower()\n"
            "        "
        ),
        required_patterns=[r"counts\[word\]|counts\.get\(word"],
        difficulty="medium",
        category="string",
    ),
    CompletionProblem(
        task_id="complete_is_palindrome",
        description="Complete a palindrome check",
        prefix=(
            "def is_palindrome(s: str) -> bool:\n"
            "    s = s.lower().replace(' ', '')\n"
            "    "
        ),
        required_patterns=[r"return\s+s\s*==\s*s\[::-1\]|s\[::-1\]"],
        difficulty="medium",
        category="string",
    ),
    CompletionProblem(
        task_id="complete_class_init",
        description="Complete a Stack class __init__",
        prefix=(
            "class Stack:\n"
            "    def __init__(self):\n"
            "        "
        ),
        required_patterns=[r"self\._?items\s*=\s*\[\]|self\._?stack\s*=\s*\[\]"],
        difficulty="medium",
        category="oop",
    ),
    CompletionProblem(
        task_id="complete_stack_push",
        description="Complete Stack.push method",
        prefix=(
            "class Stack:\n"
            "    def __init__(self):\n"
            "        self.items = []\n\n"
            "    def push(self, item):\n"
            "        "
        ),
        required_patterns=[r"self\.items\.append\(item\)"],
        difficulty="medium",
        category="oop",
    ),
    CompletionProblem(
        task_id="complete_decorator",
        description="Complete a timing decorator wrapper",
        prefix=(
            "import time\n"
            "from functools import wraps\n\n"
            "def timeit(func):\n"
            "    @wraps(func)\n"
            "    def wrapper(*args, **kwargs):\n"
            "        start = time.perf_counter()\n"
            "        result = func(*args, **kwargs)\n"
            "        "
        ),
        required_patterns=[r"time\.perf_counter\(\)\s*-\s*start|elapsed"],
        difficulty="medium",
        category="patterns",
    ),
    CompletionProblem(
        task_id="complete_context_manager",
        description="Complete __exit__ of a context manager",
        prefix=(
            "class Timer:\n"
            "    def __enter__(self):\n"
            "        self.start = time.perf_counter()\n"
            "        return self\n\n"
            "    def __exit__(self, *args):\n"
            "        "
        ),
        required_patterns=[r"self\.elapsed|perf_counter\(\)\s*-\s*self\.start"],
        difficulty="medium",
        category="patterns",
    ),
    CompletionProblem(
        task_id="complete_generator",
        description="Complete a range-of-squares generator",
        prefix=(
            "def squares(n: int):\n"
            "    \"\"\"Yield squares of 0..n-1.\"\"\"\n"
            "    for i in range(n):\n"
            "        "
        ),
        required_patterns=[r"yield\s+i\s*\*\*\s*2|yield\s+i\s*\*\s*i"],
        difficulty="medium",
        category="python",
    ),
    # ── Hard ─────────────────────────────────────────────────────────────────
    CompletionProblem(
        task_id="complete_lru_cache",
        description="Complete an LRU get method",
        prefix=(
            "from collections import OrderedDict\n\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity: int):\n"
            "        self.capacity = capacity\n"
            "        self.cache: OrderedDict = OrderedDict()\n\n"
            "    def get(self, key: int) -> int:\n"
            "        if key not in self.cache:\n"
            "            return -1\n"
            "        self.cache.move_to_end(key)\n"
            "        "
        ),
        required_patterns=[r"return\s+self\.cache\[key\]"],
        difficulty="hard",
        category="algorithm",
    ),
    CompletionProblem(
        task_id="complete_merge_sort",
        description="Complete merge-sort merge step",
        prefix=(
            "def merge_sort(arr: list[int]) -> list[int]:\n"
            "    if len(arr) <= 1:\n"
            "        return arr\n"
            "    mid = len(arr) // 2\n"
            "    left = merge_sort(arr[:mid])\n"
            "    right = merge_sort(arr[mid:])\n"
            "    return merge(left, right)\n\n"
            "def merge(left: list[int], right: list[int]) -> list[int]:\n"
            "    result = []\n"
            "    i = j = 0\n"
            "    while i < len(left) and j < len(right):\n"
            "        if left[i] <= right[j]:\n"
            "            result.append(left[i])\n"
            "            "
        ),
        required_patterns=[r"i\s*\+=\s*1"],
        difficulty="hard",
        category="algorithm",
    ),
    CompletionProblem(
        task_id="complete_trie_insert",
        description="Complete Trie insert method",
        prefix=(
            "class TrieNode:\n"
            "    def __init__(self):\n"
            "        self.children: dict[str, TrieNode] = {}\n"
            "        self.is_end = False\n\n"
            "class Trie:\n"
            "    def __init__(self):\n"
            "        self.root = TrieNode()\n\n"
            "    def insert(self, word: str) -> None:\n"
            "        node = self.root\n"
            "        for char in word:\n"
            "            if char not in node.children:\n"
            "                "
        ),
        required_patterns=[r"node\.children\[char\]\s*=\s*TrieNode\(\)"],
        difficulty="hard",
        category="algorithm",
    ),
    CompletionProblem(
        task_id="complete_bfs",
        description="Complete BFS graph traversal body",
        prefix=(
            "from collections import deque\n\n"
            "def bfs(graph: dict[str, list[str]], start: str) -> list[str]:\n"
            "    visited = set()\n"
            "    queue = deque([start])\n"
            "    order = []\n"
            "    while queue:\n"
            "        node = queue.popleft()\n"
            "        if node not in visited:\n"
            "            visited.add(node)\n"
            "            order.append(node)\n"
            "            "
        ),
        required_patterns=[r"queue\.extend|for\s+\w+\s+in\s+graph\[node\]"],
        difficulty="hard",
        category="algorithm",
    ),
    CompletionProblem(
        task_id="complete_decorator_with_args",
        description="Complete a retry decorator",
        prefix=(
            "import functools\n\n"
            "def retry(max_attempts: int = 3):\n"
            "    def decorator(func):\n"
            "        @functools.wraps(func)\n"
            "        def wrapper(*args, **kwargs):\n"
            "            for attempt in range(max_attempts):\n"
            "                try:\n"
            "                    return func(*args, **kwargs)\n"
            "                except Exception:\n"
            "                    if attempt == max_attempts - 1:\n"
            "                        "
        ),
        required_patterns=[r"raise"],
        difficulty="hard",
        category="patterns",
    ),
    CompletionProblem(
        task_id="complete_dataclass_property",
        description="Complete a dataclass with a computed property",
        prefix=(
            "from dataclasses import dataclass\n\n"
            "@dataclass\n"
            "class Rectangle:\n"
            "    width: float\n"
            "    height: float\n\n"
            "    @property\n"
            "    def area(self) -> float:\n"
            "        "
        ),
        required_patterns=[r"return\s+self\.width\s*\*\s*self\.height"],
        difficulty="hard",
        category="oop",
    ),
    CompletionProblem(
        task_id="complete_abstract_method",
        description="Complete an abstract base class method",
        prefix=(
            "from abc import ABC, abstractmethod\n\n"
            "class Shape(ABC):\n"
            "    @abstractmethod\n"
            "    def area(self) -> float:\n"
            "        "
        ),
        required_patterns=[r"\.\.\.|\bpass\b|raise\s+NotImplementedError"],
        difficulty="hard",
        category="oop",
    ),
    CompletionProblem(
        task_id="complete_type_guard",
        description="Complete a TypeGuard function",
        prefix=(
            "from typing import TypeGuard\n\n"
            "def is_string_list(val: list) -> TypeGuard[list[str]]:\n"
            "    "
        ),
        required_patterns=[r"return\s+all\(|isinstance.*str"],
        difficulty="hard",
        category="typing",
    ),
    CompletionProblem(
        task_id="complete_async_function",
        description="Complete an async HTTP-fetch wrapper",
        prefix=(
            "import asyncio\n\n"
            "async def fetch_with_timeout(coro, timeout: float):\n"
            "    try:\n"
            "        return await asyncio.wait_for(coro, timeout=timeout)\n"
            "    except asyncio.TimeoutError:\n"
            "        "
        ),
        required_patterns=[r"raise|return\s+None|None"],
        difficulty="hard",
        category="async",
    ),
    CompletionProblem(
        task_id="complete_protocol",
        description="Complete a Protocol definition",
        prefix=(
            "from typing import Protocol, runtime_checkable\n\n"
            "@runtime_checkable\n"
            "class Drawable(Protocol):\n"
            "    def draw(self) -> None:\n"
            "        "
        ),
        required_patterns=[r"\.\.\.|\bpass\b"],
        difficulty="hard",
        category="typing",
    ),
]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class CompletionBenchmark:
    """Run a code-completion benchmark against a generator callable.

    The generator callable receives a prompt string and returns the completion
    (just the new tokens, not including the prompt).

    Example usage::

        def mock_generate(prompt: str) -> str:
            return "    return a + b\\n"

        bench = CompletionBenchmark()
        report = bench.run(mock_generate)
        print(bench.to_markdown(report))
    """

    def __init__(self, problems: list[CompletionProblem] | None = None) -> None:
        self.problems = problems if problems is not None else PROBLEMS

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def run(
        self,
        generator,  # callable(prompt: str) -> str
        *,
        timeout_ms: float = 30_000,
    ) -> BenchmarkReport:
        """Run all problems through the generator and return a report."""
        results: list[CompletionResult] = []

        for problem in self.problems:
            t0 = time.perf_counter()
            try:
                completion = generator(problem.prefix)
            except Exception as exc:
                completion = f"<ERROR: {exc}>"
            latency_ms = (time.perf_counter() - t0) * 1000

            result = self.score_single(problem, completion, latency_ms=latency_ms)
            results.append(result)

        return self._aggregate(results)

    def score_single(
        self,
        problem: CompletionProblem,
        completion: str,
        *,
        latency_ms: float = 0.0,
    ) -> CompletionResult:
        """Score a single completion against a problem."""
        full_text = problem.prefix + completion

        matched: list[str] = []
        missed: list[str] = []
        for pattern in problem.required_patterns:
            if re.search(pattern, full_text):
                matched.append(pattern)
            else:
                missed.append(pattern)

        forbidden_found: list[str] = []
        for pattern in problem.forbidden_patterns:
            if re.search(pattern, full_text):
                forbidden_found.append(pattern)

        passed = len(missed) == 0 and len(forbidden_found) == 0

        return CompletionResult(
            task_id=problem.task_id,
            completion=completion,
            passed=passed,
            matched_patterns=matched,
            missed_patterns=missed,
            forbidden_found=forbidden_found,
            latency_ms=latency_ms,
        )

    def score(self, problem_id: str, completion: str) -> CompletionResult:
        """Score a completion for a problem identified by task_id."""
        problem = self._get_problem(problem_id)
        return self.score_single(problem, completion)

    # ------------------------------------------------------------------
    # Output / reporting
    # ------------------------------------------------------------------

    def to_markdown(self, report: BenchmarkReport) -> str:
        """Render the benchmark report as a Markdown string."""
        lines: list[str] = []
        lines.append("# Code Completion Benchmark Results\n")
        lines.append(f"**Pass rate:** {report.pass_rate:.1%}  "
                     f"({report.passed}/{report.total})\n")
        if report.total_latency_ms > 0:
            avg_ms = report.total_latency_ms / max(report.total, 1)
            lines.append(f"**Avg latency:** {avg_ms:.1f} ms/problem\n")

        # By difficulty
        if report.by_difficulty:
            lines.append("\n## By Difficulty\n")
            lines.append("| Difficulty | Passed | Total | Rate |")
            lines.append("|-----------|--------|-------|------|")
            for diff in sorted(report.by_difficulty):
                d = report.by_difficulty[diff]
                rate = d["passed"] / max(d["total"], 1)
                lines.append(f"| {diff} | {d['passed']} | {d['total']} | {rate:.0%} |")

        # By category
        if report.by_category:
            lines.append("\n## By Category\n")
            lines.append("| Category | Passed | Total | Rate |")
            lines.append("|---------|----|----|----|")
            for cat in sorted(report.by_category):
                c = report.by_category[cat]
                rate = c["passed"] / max(c["total"], 1)
                lines.append(f"| {cat} | {c['passed']} | {c['total']} | {rate:.0%} |")

        # Per-problem
        lines.append("\n## Per-Problem Results\n")
        lines.append("| Task | Pass | Missed Patterns |")
        lines.append("|------|------|-----------------|")
        for r in report.results:
            status = "PASS" if r.passed else "FAIL"
            missed = ", ".join(r.missed_patterns) or "—"
            lines.append(f"| {r.task_id} | {status} | {missed} |")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_problem(self, task_id: str) -> CompletionProblem:
        for p in self.problems:
            if p.task_id == task_id:
                return p
        available = [p.task_id for p in self.problems]
        raise KeyError(f"Unknown task_id '{task_id}'. Available: {available}")

    def _aggregate(self, results: list[CompletionResult]) -> BenchmarkReport:
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        total_latency = sum(r.latency_ms for r in results)

        # Look up difficulty / category from problems list
        meta: dict[str, tuple[str, str]] = {
            p.task_id: (p.difficulty, p.category) for p in self.problems
        }

        by_difficulty: dict[str, dict[str, int]] = {}
        by_category: dict[str, dict[str, int]] = {}

        for r in results:
            diff, cat = meta.get(r.task_id, ("unknown", "unknown"))
            for bucket, key in [(by_difficulty, diff), (by_category, cat)]:
                if key not in bucket:
                    bucket[key] = {"passed": 0, "total": 0}
                bucket[key]["total"] += 1
                if r.passed:
                    bucket[key]["passed"] += 1

        return BenchmarkReport(
            total=total,
            passed=passed,
            failed=total - passed,
            pass_rate=passed / max(total, 1),
            results=results,
            total_latency_ms=total_latency,
            by_difficulty=by_difficulty,
            by_category=by_category,
        )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def get_problems_by_difficulty(difficulty: str) -> list[CompletionProblem]:
    """Return all built-in problems for a given difficulty."""
    return [p for p in PROBLEMS if p.difficulty == difficulty]


def get_problems_by_category(category: str) -> list[CompletionProblem]:
    """Return all built-in problems for a given category."""
    return [p for p in PROBLEMS if p.category == category]
