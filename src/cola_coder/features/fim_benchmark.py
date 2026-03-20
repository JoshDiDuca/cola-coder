"""Fill-in-the-Middle (FIM) Benchmark: evaluate code infilling ability.

Tests the model's ability to complete code given both prefix and suffix context.
FIM is crucial for IDE-style code completion where the cursor is in the middle
of existing code.

For a TS dev: like autocomplete in VS Code where you're typing in the middle
of a function and the AI needs to understand both what came before AND after.
"""

from dataclasses import dataclass, field

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class FIMProblem:
    """A single fill-in-the-middle problem."""
    id: str
    title: str
    difficulty: int  # 1-5
    prefix: str
    suffix: str
    expected_keywords: list[str]  # Keywords that should appear in the fill
    description: str = ""
    language: str = "typescript"


PROBLEMS: list[FIMProblem] = [
    FIMProblem(
        id="fim01",
        title="Return statement",
        difficulty=1,
        prefix="function add(a: number, b: number): number {\n",
        suffix="\n}",
        expected_keywords=["return", "a", "b"],
        description="Complete a simple addition function body",
    ),
    FIMProblem(
        id="fim02",
        title="Array method chain",
        difficulty=2,
        prefix="const result = numbers\n  .filter(n => n > 0)\n",
        suffix="\n  .reduce((sum, n) => sum + n, 0);",
        expected_keywords=[".map"],
        description="Add a map step in an array method chain",
    ),
    FIMProblem(
        id="fim03",
        title="Interface property",
        difficulty=1,
        prefix="interface User {\n  name: string;\n",
        suffix="\n  createdAt: Date;\n}",
        expected_keywords=[":"],
        description="Add a property to an interface",
    ),
    FIMProblem(
        id="fim04",
        title="Error handling",
        difficulty=2,
        prefix="async function fetchData(url: string) {\n  try {\n    const response = await fetch(url);\n",
        suffix="\n  } catch (error) {\n    console.error('Failed to fetch:', error);\n    throw error;\n  }\n}",
        expected_keywords=["return", "response"],
        description="Complete the try block of a fetch function",
    ),
    FIMProblem(
        id="fim05",
        title="Switch case",
        difficulty=2,
        prefix='function getStatusText(code: number): string {\n  switch (code) {\n    case 200:\n      return "OK";\n',
        suffix='\n    default:\n      return "Unknown";\n  }\n}',
        expected_keywords=["case", "return"],
        description="Add switch cases for HTTP status codes",
    ),
    FIMProblem(
        id="fim06",
        title="Generic constraint",
        difficulty=3,
        prefix="function getProperty<T extends object, K extends keyof T>(\n  obj: T,\n",
        suffix="\n): T[K] {\n  return obj[key];\n}",
        expected_keywords=["key", "K"],
        description="Complete a generic function parameter",
    ),
    FIMProblem(
        id="fim07",
        title="Promise.all pattern",
        difficulty=3,
        prefix="async function processAll(ids: string[]): Promise<Result[]> {\n  const promises = ids.map(id =>\n",
        suffix="\n  );\n  return Promise.all(promises);\n}",
        expected_keywords=["fetch", "id"],
        description="Complete the Promise.all mapping callback",
    ),
    FIMProblem(
        id="fim08",
        title="Class method",
        difficulty=2,
        prefix="class Stack<T> {\n  private items: T[] = [];\n\n  push(item: T): void {\n    this.items.push(item);\n  }\n\n",
        suffix="\n\n  peek(): T | undefined {\n    return this.items[this.items.length - 1];\n  }\n}",
        expected_keywords=["pop", "return", "this.items"],
        description="Add a pop method to a Stack class",
    ),
    FIMProblem(
        id="fim09",
        title="Conditional type",
        difficulty=4,
        prefix="type IsArray<T> = ",
        suffix=";\ntype Test1 = IsArray<string[]>;  // true\ntype Test2 = IsArray<number>;    // false",
        expected_keywords=["extends", "true", "false"],
        description="Define a conditional type that checks if T is an array",
    ),
    FIMProblem(
        id="fim10",
        title="Reducer logic",
        difficulty=3,
        prefix="function reducer(state: State, action: Action): State {\n  switch (action.type) {\n    case 'ADD_TODO':\n",
        suffix="\n    case 'REMOVE_TODO':\n      return {\n        ...state,\n        todos: state.todos.filter(t => t.id !== action.payload),\n      };\n    default:\n      return state;\n  }\n}",
        expected_keywords=["return", "state", "todos", "action"],
        description="Complete the ADD_TODO case in a reducer",
    ),
    FIMProblem(
        id="fim11",
        title="JSDoc parameter",
        difficulty=1,
        prefix="/**\n * Calculate the area of a rectangle.\n * @param width - The width of the rectangle\n",
        suffix="\n * @returns The area of the rectangle\n */\nfunction area(width: number, height: number): number {",
        expected_keywords=["@param", "height"],
        description="Complete a JSDoc comment with parameter documentation",
    ),
    FIMProblem(
        id="fim12",
        title="Destructuring assignment",
        difficulty=2,
        prefix="function processUser(user: User) {\n  const { name,",
        suffix=", ...rest } = user;\n  console.log(`${name} (${email})`);\n}",
        expected_keywords=["email"],
        description="Complete a destructuring pattern",
    ),
]


class FIMBenchmark:
    """Fill-in-the-Middle benchmark runner."""

    def __init__(self, problems: list[FIMProblem] | None = None):
        self.problems = problems or PROBLEMS

    def get_problem(self, problem_id: str) -> FIMProblem | None:
        for p in self.problems:
            if p.id == problem_id:
                return p
        return None

    def get_by_difficulty(self, min_diff: int = 1, max_diff: int = 5) -> list[FIMProblem]:
        return [p for p in self.problems if min_diff <= p.difficulty <= max_diff]

    def format_prompt(self, problem: FIMProblem, fim_format: str = "special_tokens") -> str:
        """Format a FIM problem as a model prompt.

        Args:
            problem: The FIM problem
            fim_format: How to format FIM:
                - "special_tokens": <PRE>prefix<SUF>suffix<MID> (standard FIM)
                - "natural": prefix + "// FILL HERE" + suffix
                - "prefix_only": just the prefix (left-to-right baseline)
        """
        if fim_format == "special_tokens":
            return f"<PRE>{problem.prefix}<SUF>{problem.suffix}<MID>"
        elif fim_format == "natural":
            return f"{problem.prefix}// FILL HERE\n{problem.suffix}"
        elif fim_format == "prefix_only":
            return problem.prefix
        else:
            return f"<PRE>{problem.prefix}<SUF>{problem.suffix}<MID>"

    def evaluate_fill(self, problem: FIMProblem, fill: str) -> dict:
        """Evaluate a fill against expected keywords.

        Args:
            problem: The problem being evaluated
            fill: The model's generated fill text

        Returns:
            Dict with keyword_hits, keyword_total, keyword_score, has_syntax_issue
        """
        fill_lower = fill.lower().strip()
        hits = 0
        for kw in problem.expected_keywords:
            if kw.lower() in fill_lower:
                hits += 1

        total = len(problem.expected_keywords)
        score = hits / total if total > 0 else 0.0

        # Basic syntax checks
        has_syntax_issue = False
        combined = problem.prefix + fill + problem.suffix
        # Check brace balance
        open_braces = combined.count("{")
        close_braces = combined.count("}")
        if abs(open_braces - close_braces) > 1:
            has_syntax_issue = True
        # Check paren balance
        open_parens = combined.count("(")
        close_parens = combined.count(")")
        if abs(open_parens - close_parens) > 1:
            has_syntax_issue = True

        return {
            "keyword_hits": hits,
            "keyword_total": total,
            "keyword_score": score,
            "has_syntax_issue": has_syntax_issue,
        }

    def difficulty_distribution(self) -> dict[int, int]:
        dist: dict[int, int] = {}
        for p in self.problems:
            dist[p.difficulty] = dist.get(p.difficulty, 0) + 1
        return dict(sorted(dist.items()))

    def print_summary(self) -> None:
        from cola_coder.cli import cli
        cli.header("FIM Benchmark", f"{len(self.problems)} problems")
        dist = self.difficulty_distribution()
        for diff, count in dist.items():
            cli.info(f"Difficulty {'*' * diff}", f"{count} problems")
