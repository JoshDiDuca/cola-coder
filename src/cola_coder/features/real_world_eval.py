"""Real-World Eval: evaluate model on practical coding tasks beyond benchmarks.

Categories:
- bug_fix: Fix broken or incorrect code
- feature_addition: Add new functionality to existing code
- refactoring: Improve code structure without changing behavior
- documentation: Write docstrings, comments, or README content

Each task is scored on keyword presence in the model's output, providing a
lightweight signal for whether the output addresses the task requirements.
"""

from dataclasses import dataclass

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class RealWorldTask:
    """A single real-world coding task."""
    name: str
    category: str  # bug_fix | feature_addition | refactoring | documentation
    prompt: str
    expected_keywords: list[str]
    difficulty: int  # 1-5
    language: str = "python"


@dataclass
class EvalResult:
    """Result of evaluating one task."""
    task_name: str
    score: float  # 0.0 to 1.0
    keyword_hits: list[str]
    details: dict


# ---------------------------------------------------------------------------
# Default task bank
# ---------------------------------------------------------------------------

_DEFAULT_TASKS: list[RealWorldTask] = [
    # --- Bug fixes ---
    RealWorldTask(
        name="fix_off_by_one",
        category="bug_fix",
        prompt=(
            "The following Python function is supposed to return the last element "
            "of a list but raises an IndexError. Fix it:\n\n"
            "def last(lst):\n    return lst[len(lst)]\n"
        ),
        expected_keywords=["lst[-1]", "len(lst) - 1", "return lst"],
        difficulty=1,
        language="python",
    ),
    RealWorldTask(
        name="fix_mutable_default_arg",
        category="bug_fix",
        prompt=(
            "This function has a classic Python bug with mutable default arguments. "
            "Fix it so each call gets a fresh list:\n\n"
            "def append_to(element, to=[]):\n    to.append(element)\n    return to\n"
        ),
        expected_keywords=["None", "if to is None", "to = []"],
        difficulty=2,
        language="python",
    ),
    RealWorldTask(
        name="fix_integer_division",
        category="bug_fix",
        prompt=(
            "The function below should return the average of a list as a float "
            "but always returns an integer. Fix it:\n\n"
            "def average(nums):\n    return sum(nums) / len(nums)\n"
            "# In Python 2 this was a bug; ensure float division in Python 3 is explicit.\n"
        ),
        expected_keywords=["float", "sum", "len"],
        difficulty=1,
        language="python",
    ),
    RealWorldTask(
        name="fix_except_too_broad",
        category="bug_fix",
        prompt=(
            "The following code swallows all exceptions silently. Refactor it to "
            "catch only ValueError and re-raise anything else:\n\n"
            "def parse(s):\n    try:\n        return int(s)\n    except:\n        return None\n"
        ),
        expected_keywords=["except ValueError", "raise", "return None"],
        difficulty=2,
        language="python",
    ),
    RealWorldTask(
        name="fix_sql_injection",
        category="bug_fix",
        prompt=(
            "The following code is vulnerable to SQL injection. Fix it using "
            "parameterized queries:\n\n"
            "def get_user(conn, username):\n"
            "    cur = conn.cursor()\n"
            "    cur.execute(f\"SELECT * FROM users WHERE name = '{username}'\")\n"
            "    return cur.fetchone()\n"
        ),
        expected_keywords=["?", "%s", "parameterized", "(username,)", "(username)"],
        difficulty=3,
        language="python",
    ),
    RealWorldTask(
        name="fix_race_condition",
        category="bug_fix",
        prompt=(
            "The counter below has a race condition when used from multiple threads. "
            "Fix it with proper locking:\n\n"
            "import threading\n\nclass Counter:\n    def __init__(self):\n"
            "        self.value = 0\n\n    def increment(self):\n        self.value += 1\n"
        ),
        expected_keywords=["Lock", "threading.Lock", "with self", "acquire", "release"],
        difficulty=3,
        language="python",
    ),

    # --- Feature additions ---
    RealWorldTask(
        name="add_pagination",
        category="feature_addition",
        prompt=(
            "Add a paginate(items, page, page_size) function that returns the correct "
            "slice of items for the given page number (1-indexed) and page_size."
        ),
        expected_keywords=["def paginate", "page_size", "return items", "slice", "start", "end"],
        difficulty=2,
        language="python",
    ),
    RealWorldTask(
        name="add_retry_decorator",
        category="feature_addition",
        prompt=(
            "Write a retry(max_attempts, exceptions) decorator that retries the "
            "decorated function up to max_attempts times when one of the listed "
            "exceptions is raised, with exponential backoff."
        ),
        expected_keywords=["def retry", "decorator", "wrapper", "attempts", "sleep", "except"],
        difficulty=3,
        language="python",
    ),
    RealWorldTask(
        name="add_lru_cache",
        category="feature_addition",
        prompt=(
            "Implement a simple LRU cache class with get(key) and put(key, value) "
            "methods and a configurable capacity. Use an OrderedDict internally."
        ),
        expected_keywords=["OrderedDict", "capacity", "def get", "def put", "move_to_end"],
        difficulty=3,
        language="python",
    ),
    RealWorldTask(
        name="add_typescript_generic_stack",
        category="feature_addition",
        prompt=(
            "Write a generic Stack<T> class in TypeScript with push, pop, peek, "
            "isEmpty, and size methods."
        ),
        expected_keywords=["class Stack", "<T>", "push", "pop", "peek", "isEmpty", "size"],
        difficulty=2,
        language="typescript",
    ),
    RealWorldTask(
        name="add_rate_limiter",
        category="feature_addition",
        prompt=(
            "Implement a token-bucket rate limiter class in Python. It should have "
            "allow_request() that returns True if a request is permitted."
        ),
        expected_keywords=["def allow_request", "tokens", "capacity", "rate", "time"],
        difficulty=4,
        language="python",
    ),

    # --- Refactoring ---
    RealWorldTask(
        name="refactor_nested_ifs",
        category="refactoring",
        prompt=(
            "Refactor the following deeply nested if-else chain into early returns "
            "to reduce indentation:\n\n"
            "def process(x):\n    if x is not None:\n        if x > 0:\n"
            "            if x < 100:\n                return x * 2\n"
            "            else:\n                return 100\n"
            "        else:\n            return 0\n    else:\n        return -1\n"
        ),
        expected_keywords=["return -1", "return 0", "return 100", "if x is None", "if x <= 0"],
        difficulty=2,
        language="python",
    ),
    RealWorldTask(
        name="refactor_extract_function",
        category="refactoring",
        prompt=(
            "The following function is too long. Extract the email validation logic "
            "into a separate is_valid_email(email) helper function:\n\n"
            "def register_user(name, email, age):\n"
            "    if not name or len(name) < 2:\n        raise ValueError('bad name')\n"
            "    if '@' not in email or '.' not in email.split('@')[-1]:\n"
            "        raise ValueError('bad email')\n"
            "    if age < 18:\n        raise ValueError('too young')\n"
            "    return {'name': name, 'email': email, 'age': age}\n"
        ),
        expected_keywords=["def is_valid_email", "def register_user", "@", "return"],
        difficulty=2,
        language="python",
    ),
    RealWorldTask(
        name="refactor_list_comprehension",
        category="refactoring",
        prompt=(
            "Rewrite the following loop using a list comprehension:\n\n"
            "result = []\nfor x in range(20):\n    if x % 2 == 0:\n        result.append(x ** 2)\n"
        ),
        expected_keywords=["[", "for x in", "if x % 2", "x ** 2", "]"],
        difficulty=1,
        language="python",
    ),
    RealWorldTask(
        name="refactor_dataclass",
        category="refactoring",
        prompt=(
            "Convert the following plain class to a Python dataclass:\n\n"
            "class Point:\n    def __init__(self, x, y, z=0):\n"
            "        self.x = x\n        self.y = y\n        self.z = z\n"
        ),
        expected_keywords=["@dataclass", "from dataclasses", "x:", "y:", "z: int = 0"],
        difficulty=1,
        language="python",
    ),

    # --- Documentation ---
    RealWorldTask(
        name="document_binary_search",
        category="documentation",
        prompt=(
            "Write a comprehensive docstring for the following binary search function:\n\n"
            "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n"
            "    while lo <= hi:\n        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n            return mid\n"
            "        elif arr[mid] < target:\n            lo = mid + 1\n"
            "        else:\n            hi = mid - 1\n    return -1\n"
        ),
        expected_keywords=["Args", "Returns", "sorted", "index", "-1", "O(log"],
        difficulty=1,
        language="python",
    ),
    RealWorldTask(
        name="document_rest_api_endpoint",
        category="documentation",
        prompt=(
            "Write a JSDoc comment for a TypeScript Express route handler:\n\n"
            "async function getUser(req: Request, res: Response): Promise<void> {\n"
            "    const user = await UserService.findById(req.params.id);\n"
            "    if (!user) { res.status(404).json({ error: 'Not found' }); return; }\n"
            "    res.json(user);\n}\n"
        ),
        expected_keywords=["@param", "@returns", "404", "user", "id"],
        difficulty=2,
        language="typescript",
    ),
    RealWorldTask(
        name="document_ml_training_loop",
        category="documentation",
        prompt=(
            "Add inline comments explaining each step of this PyTorch training loop:\n\n"
            "for batch in dataloader:\n    optimizer.zero_grad()\n"
            "    outputs = model(batch['input_ids'])\n"
            "    loss = criterion(outputs, batch['labels'])\n"
            "    loss.backward()\n    optimizer.step()\n"
        ),
        expected_keywords=["gradient", "forward", "backward", "loss", "optimizer", "#"],
        difficulty=2,
        language="python",
    ),
]


# ---------------------------------------------------------------------------
# Core classes
# ---------------------------------------------------------------------------

class RealWorldEval:
    """Evaluate a code-generation model on real-world coding tasks."""

    def __init__(self, tasks: list[RealWorldTask] | None = None) -> None:
        self.tasks: list[RealWorldTask] = tasks if tasks is not None else list(_DEFAULT_TASKS)

    def get_tasks(
        self,
        category: str | None = None,
        difficulty: int | None = None,
    ) -> list[RealWorldTask]:
        """Return tasks, optionally filtered by category and/or difficulty."""
        result = self.tasks
        if category is not None:
            result = [t for t in result if t.category == category]
        if difficulty is not None:
            result = [t for t in result if t.difficulty == difficulty]
        return result

    def evaluate(self, task: RealWorldTask, output: str) -> EvalResult:
        """Score a single model output against a task's expected keywords.

        Score = fraction of expected_keywords found in output (case-insensitive).
        A task with no keywords gets a score of 0.0.
        """
        if not task.expected_keywords:
            return EvalResult(
                task_name=task.name,
                score=0.0,
                keyword_hits=[],
                details={"reason": "no expected keywords defined"},
            )

        lower_output = output.lower()
        hits = [kw for kw in task.expected_keywords if kw.lower() in lower_output]
        score = len(hits) / len(task.expected_keywords)

        return EvalResult(
            task_name=task.name,
            score=round(score, 4),
            keyword_hits=hits,
            details={
                "total_keywords": len(task.expected_keywords),
                "hits": len(hits),
                "misses": [kw for kw in task.expected_keywords if kw not in hits],
                "category": task.category,
                "difficulty": task.difficulty,
                "language": task.language,
            },
        )

    def evaluate_all(self, outputs: dict[str, str]) -> list[EvalResult]:
        """Evaluate multiple tasks at once.

        Args:
            outputs: mapping of task_name -> model output string.
                     Tasks not present in the dict receive an empty output.

        Returns:
            List of EvalResult, one per task that has an entry in outputs.
        """
        results: list[EvalResult] = []
        task_map = {t.name: t for t in self.tasks}
        for task_name, output in outputs.items():
            task = task_map.get(task_name)
            if task is None:
                continue
            results.append(self.evaluate(task, output))
        return results

    def summary(self, results: list[EvalResult]) -> dict:
        """Aggregate statistics over a list of EvalResults.

        Returns a dict with:
        - total_tasks: number of results
        - mean_score: average score across all results
        - by_category: mean score per category
        - by_difficulty: mean score per difficulty level
        - passed (score >= 0.5): count
        - failed (score < 0.5): count
        """
        if not results:
            return {
                "total_tasks": 0,
                "mean_score": 0.0,
                "by_category": {},
                "by_difficulty": {},
                "passed": 0,
                "failed": 0,
            }

        total = len(results)
        mean_score = sum(r.score for r in results) / total
        passed = sum(1 for r in results if r.score >= 0.5)

        # Group by category
        cat_scores: dict[str, list[float]] = {}
        diff_scores: dict[int, list[float]] = {}
        for r in results:
            cat = r.details.get("category", "unknown")
            diff = r.details.get("difficulty", 0)
            cat_scores.setdefault(cat, []).append(r.score)
            diff_scores.setdefault(diff, []).append(r.score)

        by_category = {cat: round(sum(s) / len(s), 4) for cat, s in cat_scores.items()}
        by_difficulty = {d: round(sum(s) / len(s), 4) for d, s in diff_scores.items()}

        return {
            "total_tasks": total,
            "mean_score": round(mean_score, 4),
            "by_category": by_category,
            "by_difficulty": by_difficulty,
            "passed": passed,
            "failed": total - passed,
        }
