"""Chain-of-thought training data generation.

This module generates training data where code solutions are preceded
by step-by-step reasoning. The model trains on this data to learn
"how to think" before writing code.

Two approaches:
1. Template-based: Programmatically generate reasoning traces for known solutions
2. Self-play: Use the model itself to generate and filter solutions (later phase)

For now, we use template-based generation with our HumanEval problems
as a starting point.
"""

from dataclasses import dataclass
from pathlib import Path

from .thinking_tokens import format_thinking_example


@dataclass
class CoTExample:
    """A single chain-of-thought training example."""
    task_id: str
    prompt: str  # The coding problem
    thinking: str  # Step-by-step reasoning
    solution: str  # The correct code


# Hand-crafted reasoning traces for a subset of problems
# These serve as "seed" examples to teach the model the reasoning format
COT_EXAMPLES: list[CoTExample] = [
    CoTExample(
        task_id="has_close_elements",
        prompt='def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    """Check if in given list of numbers, are any two numbers closer to each other than given threshold."""\n',
        thinking="""Let me think through this step by step:
1. I need to check if ANY two numbers in the list are closer than the threshold
2. "Closer" means the absolute difference between them is less than the threshold
3. I need to compare every pair of numbers - that's a nested loop
4. For each pair (i, j) where i != j, check if |numbers[i] - numbers[j]| < threshold
5. If I find any such pair, return True immediately
6. If no pair is found after checking all, return False
7. Edge case: empty list or single element -> no pairs possible -> False""",
        solution='''def has_close_elements(numbers: list[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers closer to each other than given threshold."""
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False''',
    ),
    CoTExample(
        task_id="below_zero",
        prompt='def below_zero(operations: list[int]) -> bool:\n    """Detect if bank account balance falls below zero."""\n',
        thinking="""Let me think through this:
1. Start with balance = 0
2. Process each operation (deposit or withdrawal) in order
3. After each operation, check if balance < 0
4. If it ever goes below zero, return True
5. If we get through all operations without going below zero, return False
6. Edge case: empty list -> balance stays at 0 -> return False""",
        solution='''def below_zero(operations: list[int]) -> bool:
    """Detect if bank account balance falls below zero."""
    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False''',
    ),
    CoTExample(
        task_id="fib",
        prompt='def fib(n: int) -> int:\n    """Return n-th Fibonacci number."""\n',
        thinking="""Let me think about Fibonacci:
1. Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, 21, ...
2. Each number is the sum of the two before it: fib(n) = fib(n-1) + fib(n-2)
3. Base cases: fib(1) = 1, fib(2) = 1
4. I could use recursion, but that's O(2^n) - very slow for large n
5. Better approach: iterate from the bottom up, keeping track of the last two values
6. Start with a=0, b=1, then for each step: a, b = b, a+b
7. After n iterations, b will be fib(n)""",
        solution='''def fib(n: int) -> int:
    """Return n-th Fibonacci number."""
    if n <= 0:
        return 0
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a''',
    ),
    CoTExample(
        task_id="rolling_max",
        prompt='def rolling_max(numbers: list[int]) -> list[int]:\n    """Generate list of rolling maximum element found until given moment."""\n',
        thinking="""Let me break this down:
1. For each position i, I need the maximum of all elements from index 0 to i
2. I can track a running maximum as I iterate through the list
3. Start with current_max = first element
4. For each subsequent element, update current_max = max(current_max, element)
5. Append current_max to the result list at each step
6. Edge case: empty list -> return empty list""",
        solution='''def rolling_max(numbers: list[int]) -> list[int]:
    """Generate list of rolling maximum element found until given moment."""
    if not numbers:
        return []
    result = []
    current_max = numbers[0]
    for n in numbers:
        current_max = max(current_max, n)
        result.append(current_max)
    return result''',
    ),
    CoTExample(
        task_id="correct_bracketing",
        prompt='def correct_bracketing(brackets: str) -> bool:\n    """Check if every opening bracket has a corresponding closing bracket."""\n',
        thinking="""Let me think about bracket matching:
1. Classic stack problem - but since we only have one type of bracket, a counter works
2. Initialize counter = 0
3. For each character:
   - '(' -> increment counter (opened a bracket)
   - ')' -> decrement counter (closed a bracket)
4. If counter ever goes negative, we have a ')' without a matching '(' -> return False
5. At the end, counter must be 0 (all brackets matched) -> return counter == 0
6. This handles nested brackets correctly because inner pairs cancel out first""",
        solution='''def correct_bracketing(brackets: str) -> bool:
    """Check if every opening bracket has a corresponding closing bracket."""
    depth = 0
    for b in brackets:
        if b == '(':
            depth += 1
        elif b == ')':
            depth -= 1
        if depth < 0:
            return False
    return depth == 0''',
    ),
]


TS_COT_EXAMPLES: list[CoTExample] = [
    CoTExample(
        task_id="ts_has_close_elements",
        prompt='function hasCloseElements(numbers: number[], threshold: number): boolean {\n  /** Return true if any two numbers are closer than threshold. */\n',
        thinking="""Let me think through this step by step:
1. I need to check if ANY two numbers in the array are closer than the threshold
2. "Closer" means the absolute difference is less than the threshold
3. Compare every pair (i, j) where j > i: check Math.abs(numbers[i] - numbers[j]) < threshold
4. Return true immediately on the first match; false if no pair qualifies
5. TypeScript types: numbers: number[], threshold: number, return boolean
6. Edge case: empty or single-element array — no pairs, always false""",
        solution='''  for (let i = 0; i < numbers.length; i++) {
    for (let j = i + 1; j < numbers.length; j++) {
      if (Math.abs(numbers[i] - numbers[j]) < threshold) return true;
    }
  }
  return false;
}''',
    ),
    CoTExample(
        task_id="ts_below_zero",
        prompt='function belowZero(operations: number[]): boolean {\n  /** Return true if running balance ever goes below zero. */\n',
        thinking="""Let me think through this:
1. Start with balance = 0
2. Apply each operation in sequence (positive = deposit, negative = withdrawal)
3. After each operation check if balance < 0
4. If it ever goes below zero, return true
5. If we finish without going negative, return false
6. TypeScript: operations: number[], return boolean
7. Edge case: empty operations list — balance stays 0, return false""",
        solution='''  let balance = 0;
  for (const op of operations) {
    balance += op;
    if (balance < 0) return true;
  }
  return false;
}''',
    ),
    CoTExample(
        task_id="ts_fibonacci",
        prompt='function fibonacci(n: number): number {\n  /** Return the n-th Fibonacci number (0-indexed). */\n',
        thinking="""Let me think about Fibonacci in TypeScript:
1. Sequence: 0, 1, 1, 2, 3, 5, 8, 13... (fib(0)=0, fib(1)=1)
2. Recursive is O(2^n) — too slow; use iterative bottom-up
3. Maintain two variables: a = fib(i-1), b = fib(i)
4. Each step: [a, b] = [b, a + b] (destructuring assignment)
5. After n iterations starting from i=1, b = fib(n)
6. TypeScript: n: number, return number
7. Base case: n <= 0 → return 0""",
        solution='''  if (n <= 0) return 0;
  let a = 0, b = 1;
  for (let i = 1; i < n; i++) {
    [a, b] = [b, a + b];
  }
  return b;
}''',
    ),
    CoTExample(
        task_id="ts_rolling_max",
        prompt='function rollingMax(numbers: number[]): number[] {\n  /** Return array where each element is the running maximum up to that index. */\n',
        thinking="""Let me break this down:
1. For each index i, I need max(numbers[0..i])
2. Track a running maximum and update as I iterate
3. Start: max = numbers[0], push it
4. For each subsequent element: update max = Math.max(max, n), push max
5. TypeScript: numbers: number[], return number[]
6. Edge case: empty array → return []""",
        solution='''  if (numbers.length === 0) return [];
  const result: number[] = [];
  let max = numbers[0];
  for (const n of numbers) {
    max = Math.max(max, n);
    result.push(max);
  }
  return result;
}''',
    ),
    CoTExample(
        task_id="ts_correct_bracketing",
        prompt='function correctBracketing(s: string): boolean {\n  /** Return true if every opening parenthesis has a matching closing one. */\n',
        thinking="""Let me think about bracket matching:
1. Classic counter approach: depth tracks unmatched '(' count
2. For each character: '(' → depth++, ')' → depth--
3. If depth goes negative at any point → unmatched ')' → return false
4. At the end, depth === 0 means all brackets matched
5. TypeScript: s: string, return boolean
6. Empty string → depth = 0 → return true (valid)""",
        solution='''  let depth = 0;
  for (const ch of s) {
    if (ch === "(") depth++;
    else if (ch === ")") depth--;
    if (depth < 0) return false;
  }
  return depth === 0;
}''',
    ),
    CoTExample(
        task_id="ts_pairs_sum_to_zero",
        prompt='function pairsSumToZero(l: number[]): boolean {\n  /** Return true if there are two distinct elements in l that sum to zero. */\n',
        thinking="""Let me think step by step.
1. We need two distinct elements x and y where x + y = 0, so y = -x.
2. A Set lets us check in O(1) whether -x has been seen already.
3. Scan left to right: for each x, check if -x is in seen.
4. If yes, return true immediately — we found a valid pair.
5. Otherwise, add x to seen and continue.
6. Return false if the loop completes without finding a pair.""",
        solution='''  const seen = new Set<number>();
  for (const x of l) {
    if (seen.has(-x)) return true;
    seen.add(x);
  }
  return false;
}''',
    ),
    CoTExample(
        task_id="ts_merge_sorted",
        prompt='function mergeSorted(l1: number[], l2: number[]): number[] {\n  /** Merge two sorted arrays into a single sorted array. */\n',
        thinking="""Let me think step by step.
1. Both arrays are already sorted, so I can use the two-pointer merge technique.
2. Maintain indices i (for l1) and j (for l2), both starting at 0.
3. Compare l1[i] and l2[j]: push the smaller value and advance that index.
4. Repeat until one array is exhausted.
5. Append all remaining elements from the non-exhausted array.
6. Return the merged result array.""",
        solution='''  const result: number[] = [];
  let i = 0, j = 0;
  while (i < l1.length && j < l2.length) {
    if (l1[i] <= l2[j]) result.push(l1[i++]);
    else result.push(l2[j++]);
  }
  while (i < l1.length) result.push(l1[i++]);
  while (j < l2.length) result.push(l2[j++]);
  return result;
}''',
    ),
    CoTExample(
        task_id="ts_longest_prefix",
        prompt='function longestCommonPrefix(strings: string[]): string {\n  /** Find the longest common prefix of an array of strings. */\n',
        thinking="""Let me think step by step.
1. If the array is empty, return "" immediately.
2. Start with the full first string as the candidate prefix.
3. For each subsequent string, shorten the prefix until it is a prefix of that string.
4. Shortening: remove the last character one at a time.
5. If prefix becomes "", no common prefix exists — return "" immediately.
6. After processing all strings, the remaining prefix is the answer.""",
        solution='''  if (strings.length === 0) return "";
  let prefix = strings[0];
  for (let i = 1; i < strings.length; i++) {
    while (!strings[i].startsWith(prefix)) {
      prefix = prefix.slice(0, -1);
      if (prefix === "") return "";
    }
  }
  return prefix;
}''',
    ),
    CoTExample(
        task_id="ts_even_odd_palindrome",
        prompt='function evenOddPalindrome(n: number): [number, number] {\n  /** Return [even_palindrome_count, odd_palindrome_count] in range [1, n]. */\n',
        thinking="""Let me think step by step.
1. A palindrome reads the same forwards and backwards.
2. For checking: convert number to string, reverse it, compare.
3. Iterate i from 1 to n inclusive.
4. For each palindrome: if i is even increment even counter, else odd counter.
5. Return [even, odd] as a tuple.
6. Helper function isPalin makes the palindrome check reusable.""",
        solution='''  const isPalin = (n: number): boolean => {
    const s = String(n);
    return s === s.split("").reverse().join("");
  };
  let even = 0, odd = 0;
  for (let i = 1; i <= n; i++) {
    if (isPalin(i)) {
      if (i % 2 === 0) even++;
      else odd++;
    }
  }
  return [even, odd];
}''',
    ),
    CoTExample(
        task_id="ts_move_one_ball",
        prompt='function moveOneBall(arr: number[]): boolean {\n  /** Return true if arr can be sorted non-decreasingly by right-shift rotations. */\n',
        thinking="""Let me think step by step.
1. Right-shifting rotates the array. A rotated sorted array has at most one descent.
2. A "descent" is where arr[i] < arr[i-1] — a drop in value.
3. Count descents in the array including the wrap-around (last → first).
4. If there is exactly 0 or 1 descent, the array is a rotation of a sorted array.
5. An empty array trivially returns true.
6. Return drops <= 1.""",
        solution='''  if (arr.length === 0) return true;
  let drops = 0;
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] < arr[i - 1]) drops++;
  }
  if (arr[arr.length - 1] > arr[0]) drops++;
  return drops <= 1;
}''',
    ),
    CoTExample(
        task_id="ts_deep_flatten",
        prompt='function deepFlatten(arr: unknown[]): number[] {\n  /** Recursively flatten a nested array of numbers to a flat array. */\n',
        thinking="""Let me think step by step.
1. We need to recursively unwrap arrays of arbitrary depth.
2. For each element: check if it's an array with Array.isArray().
3. If it's an array, recursively flatten it and spread into the result.
4. If it's not an array, it's a leaf number — push it directly.
5. The base case (empty array) works naturally — the loop doesn't execute.
6. Return the accumulated flat result array.""",
        solution='''  const result: number[] = [];
  for (const item of arr) {
    if (Array.isArray(item)) result.push(...deepFlatten(item));
    else result.push(item as number);
  }
  return result;
}''',
    ),
    CoTExample(
        task_id="ts_next_smallest",
        prompt='function nextSmallest(lst: number[]): number | null {\n  /** Return the second smallest distinct element, or null if it doesn\'t exist. */\n',
        thinking="""Let me think step by step.
1. We want the second distinct smallest value, so duplicates don't count twice.
2. Deduplicate with a Set, then sort numerically ascending.
3. If fewer than 2 unique values, there is no second smallest — return null.
4. Otherwise return unique[1] (index 1 in the sorted unique array).
5. Handles [1, 1] → only 1 unique value → null.
6. Handles [1, 2, 2] → 2 unique values → 2.""",
        solution='''  const unique = [...new Set(lst)].sort((a, b) => a - b);
  return unique.length >= 2 ? unique[1] : null;
}''',
    ),
    CoTExample(
        task_id="ts_separate_paren_groups",
        prompt='function separateParenGroups(paren_string: string): string[] {\n  /** Separate groups of balanced parentheses in the input string. */\n',
        thinking="""Let me think step by step.
1. Strip whitespace first to make scanning simpler.
2. Track depth (number of unmatched opens) and current group string.
3. For each character: append to current, depth++ on '(', depth-- on ')'.
4. When depth hits 0 after a ')', we've closed a complete group.
5. Push current to groups, reset current = "".
6. Return the list of completed groups.""",
        solution='''  const groups: string[] = [];
  let depth = 0, current = "";
  for (const ch of paren_string.replace(/\s/g, "")) {
    current += ch;
    if (ch === "(") depth++;
    else if (ch === ")") {
      depth--;
      if (depth === 0) {
        groups.push(current);
        current = "";
      }
    }
  }
  return groups;
}''',
    ),
    CoTExample(
        task_id="ts_count_up_to",
        prompt='function countUpTo(n: number): number[] {\n  /** Return an array of prime numbers less than n. */\n',
        thinking="""Let me think step by step.
1. We need all primes strictly less than n.
2. A prime is a number > 1 divisible only by 1 and itself.
3. For each candidate i from 2 to n-1, check if it's prime.
4. Prime check: try divisors from 2 to sqrt(i). If any divide evenly, not prime.
5. If no divisor found, it's prime — add to result.
6. Return the accumulated list of primes.""",
        solution='''  const isPrime = (x: number): boolean => {
    if (x < 2) return false;
    for (let i = 2; i <= Math.sqrt(x); i++) {
      if (x % i === 0) return false;
    }
    return true;
  };
  const result: number[] = [];
  for (let i = 2; i < n; i++) {
    if (isPrime(i)) result.push(i);
  }
  return result;
}''',
    ),
]


def get_cot_training_data(language: str = "python") -> list[dict[str, str]]:
    """Get chain-of-thought training examples formatted for training.

    Args:
        language: "python" (default) or "typescript".

    Returns:
        List of dicts with 'text' field containing the full training example
        in the format: prompt + <think>reasoning</think>\nsolution
    """
    examples = TS_COT_EXAMPLES if language == "typescript" else COT_EXAMPLES
    training_data = []
    for example in examples:
        text = example.prompt + format_thinking_example(
            thinking=example.thinking,
            code=example.solution,
        )
        training_data.append({"text": text})

    return training_data


def save_cot_data(output_dir: str = "./data/reasoning"):
    """Save chain-of-thought training data to disk.

    Args:
        output_dir: Directory to save the data files.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = get_cot_training_data()

    for i, example in enumerate(data):
        file_path = out_path / f"cot_{i:04d}.txt"
        file_path.write_text(example["text"], encoding="utf-8")

    print(f"Saved {len(data)} CoT training examples to {output_dir}")


def generate_cot_from_solutions(
    problems: list[dict],
    solutions: list[str],
) -> list[CoTExample]:
    """Generate chain-of-thought examples from successful solutions.

    This is used in the "self-play" approach: generate solutions,
    keep the ones that pass, and create reasoning traces for them.

    Args:
        problems: List of problem definitions.
        solutions: List of correct solutions (code strings).

    Returns:
        List of CoT examples with generated reasoning.
    """
    examples = []
    for problem, solution in zip(problems, solutions):
        # Generate a simple reasoning trace from the solution structure
        thinking = _generate_reasoning_trace(solution)
        examples.append(CoTExample(
            task_id=problem.get("task_id", "unknown"),
            prompt=problem.get("prompt", ""),
            thinking=thinking,
            solution=solution,
        ))
    return examples


def _generate_reasoning_trace(code: str) -> str:
    """Generate a basic reasoning trace from code structure.

    This is a simple heuristic that analyzes the code and produces
    a step-by-step description. It's not perfect, but it gives the
    model something to learn the format from.
    """
    lines = [ln.strip() for ln in code.split("\n") if ln.strip() and not ln.strip().startswith("#")]
    steps = ["Let me think through this step by step:"]

    step_num = 1
    for line in lines:
        if line.startswith("def "):
            continue  # Skip the function signature
        if line.startswith('"""') or line.startswith("'''"):
            continue  # Skip docstrings

        if line.startswith("if "):
            steps.append(f"{step_num}. Check the condition: {line}")
        elif line.startswith("for "):
            steps.append(f"{step_num}. Iterate: {line}")
        elif line.startswith("while "):
            steps.append(f"{step_num}. Loop: {line}")
        elif line.startswith("return "):
            steps.append(f"{step_num}. Return the result: {line}")
        elif "=" in line and not line.startswith("="):
            steps.append(f"{step_num}. Set up: {line}")
        else:
            steps.append(f"{step_num}. Execute: {line}")
        step_num += 1

    return "\n".join(steps)
