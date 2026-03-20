"""TypeScript Benchmark: 20 TypeScript coding problems for model evaluation.

Comprehensive benchmark testing TypeScript-specific capabilities:
- Type annotations and generics
- Interface/type definitions
- Async/await patterns
- Array methods and functional patterns
- Error handling with discriminated unions

Each problem has a signature, test code, and difficulty rating.
"""

from dataclasses import dataclass, field

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class TSProblem:
    """A single TypeScript benchmark problem."""
    id: str
    title: str
    difficulty: int  # 1-5
    signature: str
    description: str
    test_code: str
    tags: list[str] = field(default_factory=list)


# ── Problem Set ──────────────────────────────────────────────────────────

PROBLEMS: list[TSProblem] = [
    TSProblem(
        id="ts01",
        title="Type-safe identity",
        difficulty=1,
        signature="function identity<T>(value: T): T",
        description="Return the input value unchanged, preserving its type.",
        test_code='console.assert(identity(42) === 42); console.assert(identity("hi") === "hi"); console.log("ts01 ok");',
        tags=["generics"],
    ),
    TSProblem(
        id="ts02",
        title="Array last element",
        difficulty=1,
        signature="function last<T>(arr: T[]): T | undefined",
        description="Return the last element of an array, or undefined if empty.",
        test_code='console.assert(last([1,2,3]) === 3); console.assert(last([]) === undefined); console.log("ts02 ok");',
        tags=["arrays", "generics"],
    ),
    TSProblem(
        id="ts03",
        title="String reversal",
        difficulty=1,
        signature="function reverseString(s: string): string",
        description="Reverse the characters in a string.",
        test_code='console.assert(reverseString("hello") === "olleh"); console.assert(reverseString("") === ""); console.log("ts03 ok");',
        tags=["strings"],
    ),
    TSProblem(
        id="ts04",
        title="Flatten array",
        difficulty=2,
        signature="function flatten<T>(arr: (T | T[])[]): T[]",
        description="Flatten a nested array one level deep.",
        test_code='const r = flatten([[1,2],[3],4]); console.assert(JSON.stringify(r) === "[1,2,3,4]"); console.log("ts04 ok");',
        tags=["arrays", "generics"],
    ),
    TSProblem(
        id="ts05",
        title="Object pick",
        difficulty=2,
        signature="function pick<T extends object, K extends keyof T>(obj: T, keys: K[]): Pick<T, K>",
        description="Create a new object with only the specified keys from the source object.",
        test_code='const r = pick({a:1,b:2,c:3}, ["a","c"]); console.assert(r.a === 1 && r.c === 3 && !("b" in r)); console.log("ts05 ok");',
        tags=["objects", "generics", "utility-types"],
    ),
    TSProblem(
        id="ts06",
        title="Debounce function",
        difficulty=3,
        signature="function debounce<T extends (...args: any[]) => void>(fn: T, delay: number): T",
        description="Return a debounced version of the function that delays invocation until after delay ms have elapsed since the last call.",
        test_code='let count = 0; const inc = debounce(() => count++, 50); inc(); inc(); inc(); setTimeout(() => { console.assert(count === 1); console.log("ts06 ok"); }, 100);',
        tags=["functions", "generics", "async"],
    ),
    TSProblem(
        id="ts07",
        title="Group by key",
        difficulty=2,
        signature="function groupBy<T>(arr: T[], key: keyof T): Record<string, T[]>",
        description="Group array elements by a key property.",
        test_code='const r = groupBy([{a:"x",b:1},{a:"y",b:2},{a:"x",b:3}], "a"); console.assert(r["x"].length === 2 && r["y"].length === 1); console.log("ts07 ok");',
        tags=["arrays", "objects", "generics"],
    ),
    TSProblem(
        id="ts08",
        title="Safe JSON parse",
        difficulty=2,
        signature='function safeJsonParse<T>(json: string): { ok: true; value: T } | { ok: false; error: string }',
        description="Parse JSON safely, returning a discriminated union result.",
        test_code='const r1 = safeJsonParse<number>("42"); console.assert(r1.ok && r1.value === 42); const r2 = safeJsonParse("{bad}"); console.assert(!r2.ok); console.log("ts08 ok");',
        tags=["error-handling", "discriminated-unions"],
    ),
    TSProblem(
        id="ts09",
        title="Memoize function",
        difficulty=3,
        signature="function memoize<T extends (...args: any[]) => any>(fn: T): T",
        description="Return a memoized version that caches results based on arguments.",
        test_code='let calls = 0; const add = memoize((a: number, b: number) => { calls++; return a+b; }); add(1,2); add(1,2); console.assert(calls === 1); console.assert(add(1,2) === 3); console.log("ts09 ok");',
        tags=["functions", "generics", "caching"],
    ),
    TSProblem(
        id="ts10",
        title="Deep readonly",
        difficulty=3,
        signature="type DeepReadonly<T> = T extends object ? { readonly [K in keyof T]: DeepReadonly<T[K]> } : T;",
        description="Create a type that makes all properties and nested properties readonly.",
        test_code='const obj: DeepReadonly<{a: {b: number}}> = {a: {b: 1}}; console.log("ts10 ok");',
        tags=["types", "utility-types", "mapped-types"],
    ),
    TSProblem(
        id="ts11",
        title="Pipe functions",
        difficulty=3,
        signature="function pipe<T>(value: T, ...fns: ((arg: any) => any)[]): any",
        description="Apply a series of functions to a value, left to right.",
        test_code='const r = pipe(5, (x: number) => x * 2, (x: number) => x + 1, (x: number) => String(x)); console.assert(r === "11"); console.log("ts11 ok");',
        tags=["functions", "composition"],
    ),
    TSProblem(
        id="ts12",
        title="Event emitter",
        difficulty=3,
        signature="class TypedEmitter<Events extends Record<string, any[]>>",
        description="Create a type-safe event emitter with on, off, and emit methods.",
        test_code='const e = new TypedEmitter<{click: [x: number, y: number]}>(); let result = 0; const handler = (x: number, y: number) => result = x + y; e.on("click", handler); e.emit("click", 3, 4); console.assert(result === 7); e.off("click", handler); e.emit("click", 1, 1); console.assert(result === 7); console.log("ts12 ok");',
        tags=["classes", "generics", "events"],
    ),
    TSProblem(
        id="ts13",
        title="Promise retry",
        difficulty=3,
        signature="function retry<T>(fn: () => Promise<T>, maxRetries: number, delay: number): Promise<T>",
        description="Retry an async function up to maxRetries times with a delay between attempts.",
        test_code='let attempt = 0; const flaky = () => new Promise<number>((res, rej) => { attempt++; attempt >= 3 ? res(42) : rej("fail"); }); retry(flaky, 5, 10).then(r => { console.assert(r === 42); console.assert(attempt === 3); console.log("ts13 ok"); });',
        tags=["async", "promises", "error-handling"],
    ),
    TSProblem(
        id="ts14",
        title="Chunk array",
        difficulty=2,
        signature="function chunk<T>(arr: T[], size: number): T[][]",
        description="Split an array into chunks of the given size.",
        test_code='const r = chunk([1,2,3,4,5], 2); console.assert(JSON.stringify(r) === "[[1,2],[3,4],[5]]"); console.log("ts14 ok");',
        tags=["arrays", "generics"],
    ),
    TSProblem(
        id="ts15",
        title="Deep merge",
        difficulty=4,
        signature="function deepMerge<T extends object>(target: T, ...sources: Partial<T>[]): T",
        description="Deep merge multiple objects, with later sources overriding earlier ones.",
        test_code='const r = deepMerge({a:1,b:{c:2,d:3}}, {b:{c:4,e:5}} as any); console.assert(r.a === 1 && r.b.c === 4 && r.b.d === 3); console.log("ts15 ok");',
        tags=["objects", "recursion", "generics"],
    ),
    TSProblem(
        id="ts16",
        title="LRU Cache",
        difficulty=4,
        signature="class LRUCache<K, V>",
        description="Implement a Least Recently Used cache with get and set operations.",
        test_code='const c = new LRUCache<string,number>(2); c.set("a",1); c.set("b",2); console.assert(c.get("a") === 1); c.set("c",3); console.assert(c.get("b") === undefined); console.log("ts16 ok");',
        tags=["classes", "generics", "data-structures"],
    ),
    TSProblem(
        id="ts17",
        title="Observable pattern",
        difficulty=4,
        signature="class Observable<T>",
        description="Implement an Observable with subscribe, map, and filter operators.",
        test_code='const results: number[] = []; const obs = new Observable<number>((subscriber) => { subscriber.next(1); subscriber.next(2); subscriber.next(3); }); obs.filter(x => x > 1).map(x => x * 10).subscribe(x => results.push(x)); console.assert(JSON.stringify(results) === "[20,30]"); console.log("ts17 ok");',
        tags=["classes", "generics", "reactive"],
    ),
    TSProblem(
        id="ts18",
        title="Path type",
        difficulty=4,
        signature="type Path<T, Prefix extends string = ''> = T extends object ? { [K in keyof T & string]: K | `${K}.${Path<T[K]>}` }[keyof T & string] : never;",
        description="Create a type that extracts all dot-notation paths from an object type.",
        test_code='type Obj = {a: {b: {c: number}, d: string}}; const p: Path<Obj> = "a.b.c"; console.log("ts18 ok");',
        tags=["types", "template-literals", "recursion"],
    ),
    TSProblem(
        id="ts19",
        title="Async queue",
        difficulty=5,
        signature="class AsyncQueue<T>",
        description="Implement a queue where dequeue returns a Promise that resolves when an item is available.",
        test_code='const q = new AsyncQueue<number>(); setTimeout(() => q.enqueue(42), 50); q.dequeue().then(v => { console.assert(v === 42); console.log("ts19 ok"); });',
        tags=["classes", "async", "data-structures"],
    ),
    TSProblem(
        id="ts20",
        title="Type-safe builder",
        difficulty=5,
        signature="class QueryBuilder<T extends object>",
        description="Implement a type-safe query builder with where, select, and orderBy methods that return typed results.",
        test_code='type User = {name: string; age: number; email: string}; const q = new QueryBuilder<User>().select("name", "age").where("age", ">", 18).orderBy("name"); console.assert(q.build().select.length === 2); console.log("ts20 ok");',
        tags=["classes", "generics", "builder-pattern"],
    ),
]


class TypeScriptBenchmark:
    """Run TypeScript benchmark problems against a code generation model."""

    def __init__(self, problems: list[TSProblem] | None = None):
        self.problems = problems or PROBLEMS

    def get_problem(self, problem_id: str) -> TSProblem | None:
        """Get a problem by ID."""
        for p in self.problems:
            if p.id == problem_id:
                return p
        return None

    def get_by_difficulty(self, min_diff: int = 1, max_diff: int = 5) -> list[TSProblem]:
        """Filter problems by difficulty range."""
        return [p for p in self.problems if min_diff <= p.difficulty <= max_diff]

    def get_by_tag(self, tag: str) -> list[TSProblem]:
        """Filter problems by tag."""
        return [p for p in self.problems if tag in p.tags]

    def format_prompt(self, problem: TSProblem) -> str:
        """Format a problem as a prompt for the model."""
        return (
            f"// {problem.title}\n"
            f"// {problem.description}\n"
            f"{problem.signature} {{\n"
        )

    def all_tags(self) -> list[str]:
        """Get all unique tags across problems."""
        tags = set()
        for p in self.problems:
            tags.update(p.tags)
        return sorted(tags)

    def difficulty_distribution(self) -> dict[int, int]:
        """Count problems per difficulty level."""
        dist: dict[int, int] = {}
        for p in self.problems:
            dist[p.difficulty] = dist.get(p.difficulty, 0) + 1
        return dict(sorted(dist.items()))

    def print_summary(self) -> None:
        """Print a summary of the benchmark."""
        from cola_coder.cli import cli
        cli.header("TypeScript Benchmark", f"{len(self.problems)} problems")
        dist = self.difficulty_distribution()
        for diff, count in dist.items():
            stars = "*" * diff
            cli.info(f"Difficulty {stars}", f"{count} problems")
        cli.info("Tags", ", ".join(self.all_tags()))
