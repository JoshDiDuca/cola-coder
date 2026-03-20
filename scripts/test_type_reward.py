"""Quick test: score some TypeScript code with the type check reward.

Usage:
    python scripts/test_type_reward.py
    python scripts/test_type_reward.py --file path/to/file.ts
    python scripts/test_type_reward.py --batch  # test batch scoring
"""

import argparse
import sys
import time

# Add project root to path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cola_coder.reasoning.rewards import TypeCheckReward, BatchTypeChecker, CombinedReward


# --- Sample TypeScript snippets for testing ---

PERFECT_CODE = """\
interface User {
    id: number;
    name: string;
    email: string;
}

function greet(user: User): string {
    return `Hello, ${user.name}!`;
}

const users: User[] = [
    { id: 1, name: "Alice", email: "alice@example.com" },
    { id: 2, name: "Bob", email: "bob@example.com" },
];

const messages: string[] = users.map(greet);
"""

MINOR_ERRORS = """\
interface Config {
    host: string;
    port: number;
}

function createServer(config: Config) {
    // Missing return type annotation (not an error in strict, but port is wrong type)
    const url: string = `${config.host}:${config.port}`;
    const badPort: string = config.port;  // TS2322: number not assignable to string
    return url;
}
"""

MODERATE_ERRORS = """\
function process(data) {  // TS7006: implicit any
    const result = data.map(item => item.value);  // implicit any
    const sum: string = result.reduce((a, b) => a + b, 0);  // wrong type
    const flag: boolean = "hello";  // TS2322
    return { sum, flag, extra: data.missing.property };  // possible runtime error
}
"""

SYNTAX_ERROR = """\
function broken( {
    const x = ;
    if (true {
        console.log("missing paren")
    }
"""

EMPTY_CODE = ""


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def test_single_file(file_path: str | None = None):
    """Test scoring on built-in snippets or a provided file."""
    print_section("TypeScript Type Check Reward — Single File Mode")

    # Check availability
    available = TypeCheckReward.is_available()
    print(f"tsc available: {available}")

    if not available:
        print("\nWARNING: tsc not installed. Install with: npm install -g typescript")
        print("Falling back to CombinedReward without type checking.\n")
        reward = CombinedReward()
        snippets = {
            "Perfect code": PERFECT_CODE,
            "Syntax error": SYNTAX_ERROR,
            "Empty": EMPTY_CODE,
        }
        for name, code in snippets.items():
            result = reward.detailed_score(code)
            print(f"  {name}:")
            print(f"    Combined score: {result['combined_score']:.2f}")
            print(f"    Syntax: {result['syntax_score']:.2f}, "
                  f"Style: {result['style_score']:.2f}, "
                  f"Completeness: {result['completeness_score']:.2f}")
            print()
        return

    # Score a user-provided file
    if file_path:
        code = Path(file_path).read_text(encoding="utf-8")
        reward = TypeCheckReward()
        start = time.perf_counter()
        result = reward.detailed_score(code)
        elapsed = time.perf_counter() - start
        print(f"File: {file_path}")
        print(f"Score: {result['score']:.2f}")
        print(f"Errors: {result['num_errors']}")
        print(f"Time: {elapsed*1000:.1f}ms")
        if result["errors"]:
            print("\nErrors:")
            for e in result["errors"]:
                print(f"  Line {e['line']}: TS{e['code']} — {e['message']}")
        return

    # Score built-in snippets
    reward = TypeCheckReward()
    snippets = {
        "Perfect code (should be ~1.0)": PERFECT_CODE,
        "Minor errors (should be ~0.7)": MINOR_ERRORS,
        "Moderate errors (should be ~0.3)": MODERATE_ERRORS,
        "Syntax error (should be -0.5)": SYNTAX_ERROR,
        "Empty code (should be 0.0)": EMPTY_CODE,
    }

    for name, code in snippets.items():
        start = time.perf_counter()
        result = reward.detailed_score(code)
        elapsed = time.perf_counter() - start

        print(f"  {name}:")
        print(f"    Score: {result['score']:.2f}  "
              f"({result['num_errors']} errors, {elapsed*1000:.1f}ms)")
        if result["error_codes"]:
            print(f"    Error codes: {', '.join(result['error_codes'][:5])}")
        print()


def test_batch():
    """Test batch type checking."""
    print_section("TypeScript Type Check Reward — Batch Mode")

    if not BatchTypeChecker.is_available():
        print("tsc not available — skipping batch test")
        return

    checker = BatchTypeChecker()
    codes = [PERFECT_CODE, MINOR_ERRORS, MODERATE_ERRORS, SYNTAX_ERROR, EMPTY_CODE]

    start = time.perf_counter()
    results = checker.detailed_batch(codes)
    elapsed = time.perf_counter() - start

    names = ["Perfect", "Minor errors", "Moderate errors", "Syntax error", "Empty"]
    for name, result in zip(names, results):
        print(f"  {name}: score={result['score']:.2f}, errors={result['num_errors']}")

    print(f"\n  Batch time: {elapsed*1000:.1f}ms for {len(codes)} files "
          f"({elapsed*1000/len(codes):.1f}ms per file)")


def test_combined():
    """Test the combined multi-signal reward."""
    print_section("Combined Multi-Signal Reward")

    reward = CombinedReward()
    print(f"  tsc available: {reward.has_type_checker}")

    snippets = {
        "Perfect code": PERFECT_CODE,
        "Minor errors": MINOR_ERRORS,
        "Syntax error": SYNTAX_ERROR,
    }

    for name, code in snippets.items():
        result = reward.detailed_score(code)
        print(f"\n  {name}:")
        print(f"    Combined: {result['combined_score']:.3f}")
        type_str = f"{result['type_score']:.2f}" if result['type_score'] is not None else "N/A"
        print(f"    Type: {type_str}, "
              f"Syntax: {result['syntax_score']:.2f}, "
              f"Style: {result['style_score']:.2f}, "
              f"Completeness: {result['completeness_score']:.2f}")
        print(f"    Weights: {result['weights']}")


def main():
    parser = argparse.ArgumentParser(
        description="Test TypeScript type check reward function."
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to a .ts file to score.",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Also test batch scoring mode.",
    )
    args = parser.parse_args()

    test_single_file(args.file)
    if args.batch:
        test_batch()
    test_combined()


if __name__ == "__main__":
    main()
