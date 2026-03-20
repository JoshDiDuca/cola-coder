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


def get_cot_training_data() -> list[dict[str, str]]:
    """Get chain-of-thought training examples formatted for training.

    Returns:
        List of dicts with 'text' field containing the full training example
        in the format: prompt + <think>reasoning</think>\nsolution
    """
    training_data = []
    for example in COT_EXAMPLES:
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
