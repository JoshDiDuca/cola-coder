"""HumanEval-style coding problems for evaluating code generation.

HumanEval is the standard benchmark for code generation models. Each problem
has a function signature + docstring, and the model must generate the
function body. The generated code is then tested against provided test cases.

These problems are defined inline (not pulled from any external source).
They cover a range of difficulty: string manipulation, list operations,
math, and basic algorithms.

For a TS dev: think of these like unit test fixtures — each one defines
a coding problem and its expected solution behavior.
"""

from dataclasses import dataclass


@dataclass
class CodingProblem:
    """A single coding problem for evaluation."""
    task_id: str  # Unique identifier
    prompt: str  # Function signature + docstring (model sees this)
    test_code: str  # Test cases (model doesn't see this)
    entry_point: str  # Function name to test


# Collection of coding problems
PROBLEMS: list[CodingProblem] = [
    CodingProblem(
        task_id="has_close_elements",
        prompt='''def has_close_elements(numbers: list[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers closer to each other
    than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
''',
        test_code='''
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0], 2.0) == True
assert has_close_elements([], 0.5) == False
''',
        entry_point="has_close_elements",
    ),
    CodingProblem(
        task_id="separate_paren_groups",
        prompt='''def separate_paren_groups(paren_string: str) -> list[str]:
    """Input to this function is a string containing multiple groups of nested
    parentheses. Your goal is to separate those groups into separate strings
    and return the list of those. Separate groups are balanced (each open
    brace is properly closed) and not nested within each other.
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
''',
        test_code='''
assert separate_paren_groups('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']
assert separate_paren_groups('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']
assert separate_paren_groups('(()(())((())))') == ['(()(())((())))']
assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']
''',
        entry_point="separate_paren_groups",
    ),
    CodingProblem(
        task_id="truncate_number",
        prompt='''def truncate_number(number: float) -> float:
    """Given a positive floating point number, it can be decomposed into
    an integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).
    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """
''',
        test_code='''
assert truncate_number(3.5) == 0.5
assert abs(truncate_number(1.33) - 0.33) < 1e-6
assert abs(truncate_number(123.456) - 0.456) < 1e-6
''',
        entry_point="truncate_number",
    ),
    CodingProblem(
        task_id="below_zero",
        prompt='''def below_zero(operations: list[int]) -> bool:
    """You're given a list of deposit and withdrawal operations on a bank
    account that starts with zero balance. Your task is to detect if at
    any point the balance of account falls below zero, and at that point
    function should return True. Otherwise it should return False.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    """
''',
        test_code='''
assert below_zero([]) == False
assert below_zero([1, 2, -3, 1, 2, -3]) == False
assert below_zero([1, 2, -4, 5, 6]) == True
assert below_zero([1, -1, 2, -2, 5, -5, 4, -4]) == False
assert below_zero([1, -1, 2, -2, 5, -5, 4, -5]) == True
assert below_zero([1, -2]) == True
''',
        entry_point="below_zero",
    ),
    CodingProblem(
        task_id="mean_absolute_deviation",
        prompt='''def mean_absolute_deviation(numbers: list[float]) -> float:
    """For a given list of input numbers, calculate Mean Absolute Deviation
    around the mean of this dataset.
    Mean Absolute Deviation is the average absolute difference between each
    element and a centerpoint (mean in this case):
    MAD = average | x - x_mean |
    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
    1.0
    """
''',
        test_code='''
assert abs(mean_absolute_deviation([1.0, 2.0, 3.0]) - 2/3) < 1e-6
assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6
assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0, 5.0]) - 1.2) < 1e-6
''',
        entry_point="mean_absolute_deviation",
    ),
    CodingProblem(
        task_id="intersperse",
        prompt='''def intersperse(numbers: list[int], delimiter: int) -> list[int]:
    """Insert a number 'delimiter' between every two consecutive elements
    of input list `numbers`.
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
''',
        test_code='''
assert intersperse([], 7) == []
assert intersperse([5, 6, 3, 2], 8) == [5, 8, 6, 8, 3, 8, 2]
assert intersperse([2, 2, 2], 2) == [2, 2, 2, 2, 2]
''',
        entry_point="intersperse",
    ),
    CodingProblem(
        task_id="parse_nested_parens",
        prompt='''def parse_nested_parens(paren_string: str) -> list[int]:
    """Input to this function is a string represented multiple groups for
    nested parentheses separated by spaces. For each of the group, output
    the deepest level of nesting of parentheses.
    E.g. (()()) has maximum two levels of nesting while ((())) has three.
    >>> parse_nested_parens('(()()) ((())) () ((())()())')
    [2, 3, 1, 3]
    """
''',
        test_code='''
assert parse_nested_parens('(()()) ((())) () ((())()())') == [2, 3, 1, 3]
assert parse_nested_parens('() (()) ((())) (((())))') == [1, 2, 3, 4]
assert parse_nested_parens('(()(())((())))') == [4]
''',
        entry_point="parse_nested_parens",
    ),
    CodingProblem(
        task_id="filter_by_substring",
        prompt='''def filter_by_substring(strings: list[str], substring: str) -> list[str]:
    """Filter an input list of strings only for ones that contain given substring.
    >>> filter_by_substring([], 'a')
    []
    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
    ['abc', 'bacd', 'array']
    """
''',
        test_code='''
assert filter_by_substring([], 'john') == []
assert filter_by_substring(['xxx', 'asd', 'xxy', 'john doe', 'xxxuj', 'xxx'], 'xxx') == ['xxx', 'xxxuj', 'xxx']
assert filter_by_substring(['xxx', 'asd', 'aaber', 'john doe', 'xxxuj', 'xxx'], 'xx') == ['xxx', 'xxxuj', 'xxx']
assert filter_by_substring(['grunt', 'hierarchies', 'cadeau', 'malign'], 'hi') == ['hierarchies']
''',
        entry_point="filter_by_substring",
    ),
    CodingProblem(
        task_id="sum_product",
        prompt='''def sum_product(numbers: list[int]) -> tuple[int, int]:
    """For a given list of integers, return a tuple consisting of a sum and
    a product of all the integers in a list. Empty sum should be equal to 0
    and empty product should be equal to 1.
    >>> sum_product([])
    (0, 1)
    >>> sum_product([1, 2, 3, 4])
    (10, 24)
    """
''',
        test_code='''
assert sum_product([]) == (0, 1)
assert sum_product([1, 1, 1]) == (3, 1)
assert sum_product([100, 0]) == (100, 0)
assert sum_product([3, 5, 7]) == (15, 105)
assert sum_product([10]) == (10, 10)
''',
        entry_point="sum_product",
    ),
    CodingProblem(
        task_id="rolling_max",
        prompt='''def rolling_max(numbers: list[int]) -> list[int]:
    """From a given list of integers, generate a list of rolling maximum
    element found until given moment in the sequence.
    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])
    [1, 2, 3, 3, 3, 4, 4]
    """
''',
        test_code='''
assert rolling_max([]) == []
assert rolling_max([1, 2, 3, 4]) == [1, 2, 3, 4]
assert rolling_max([4, 3, 2, 1]) == [4, 4, 4, 4]
assert rolling_max([3, 2, 3, 100, 3]) == [3, 3, 3, 100, 100]
''',
        entry_point="rolling_max",
    ),
    CodingProblem(
        task_id="is_palindrome",
        prompt='''def is_palindrome_string(string: str) -> bool:
    """Test if given string is a palindrome.
    >>> is_palindrome_string('')
    True
    >>> is_palindrome_string('aba')
    True
    >>> is_palindrome_string('abc')
    False
    """
''',
        test_code='''
assert is_palindrome_string('') == True
assert is_palindrome_string('aba') == True
assert is_palindrome_string('aaaaa') == True
assert is_palindrome_string('zbcd') == False
assert is_palindrome_string('xywyx') == True
assert is_palindrome_string('xywyz') == False
''',
        entry_point="is_palindrome_string",
    ),
    CodingProblem(
        task_id="remove_vowels",
        prompt='''def remove_vowels(text: str) -> str:
    """Remove all vowels from the given string.
    >>> remove_vowels('')
    ''
    >>> remove_vowels("abcdef\\nghijklm")
    'bcdf\\nghjklm'
    >>> remove_vowels('aeiou')
    ''
    """
''',
        test_code='''
assert remove_vowels('') == ''
assert remove_vowels("abcdef\\nghijklm") == 'bcdf\\nghjklm'
assert remove_vowels('fedcba') == 'fdcb'
assert remove_vowels('eeeee') == ''
assert remove_vowels('acBAA') == 'cB'
assert remove_vowels('EcBOO') == 'cB'
''',
        entry_point="remove_vowels",
    ),
    CodingProblem(
        task_id="below_threshold",
        prompt='''def below_threshold(l: list[int], t: int) -> bool:
    """Return True if all numbers in the list l are below threshold t.
    >>> below_threshold([1, 2, 4, 10], 100)
    True
    >>> below_threshold([1, 20, 4, 10], 5)
    False
    """
''',
        test_code='''
assert below_threshold([1, 2, 4, 10], 100) == True
assert below_threshold([1, 20, 4, 10], 5) == False
assert below_threshold([1, 20, 4, 10], 21) == True
assert below_threshold([1, 20, 4, 10], 22) == True
assert below_threshold([1, 8, 4, 10], 11) == True
assert below_threshold([1, 8, 4, 10], 10) == False
''',
        entry_point="below_threshold",
    ),
    CodingProblem(
        task_id="add_elements",
        prompt='''def add(x: int, y: int) -> int:
    """Add two numbers x and y.
    >>> add(2, 3)
    5
    >>> add(5, 7)
    12
    """
''',
        test_code='''
assert add(0, 1) == 1
assert add(1, 0) == 1
assert add(2, 3) == 5
assert add(5, 7) == 12
assert add(7, 5) == 12
''',
        entry_point="add",
    ),
    CodingProblem(
        task_id="same_chars",
        prompt='''def same_chars(s0: str, s1: str) -> bool:
    """Check if two words have the same characters.
    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc')
    True
    >>> same_chars('abcd', 'dddddddabc')
    True
    >>> same_chars('dddddddabc', 'abcd')
    True
    >>> same_chars('eabcd', 'dddddddabc')
    False
    """
''',
        test_code='''
assert same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc') == True
assert same_chars('abcd', 'dddddddabc') == True
assert same_chars('dddddddabc', 'abcd') == True
assert same_chars('eabcd', 'dddddddabc') == False
assert same_chars('abcd', 'dddddddabce') == False
assert same_chars('aabb', 'aaccc') == False
''',
        entry_point="same_chars",
    ),
    CodingProblem(
        task_id="fib",
        prompt='''def fib(n: int) -> int:
    """Return n-th Fibonacci number.
    >>> fib(10)
    55
    >>> fib(1)
    1
    >>> fib(8)
    21
    """
''',
        test_code='''
assert fib(10) == 55
assert fib(1) == 1
assert fib(8) == 21
assert fib(11) == 89
assert fib(12) == 144
''',
        entry_point="fib",
    ),
    CodingProblem(
        task_id="common_elements",
        prompt='''def common(l1: list[int], l2: list[int]) -> list[int]:
    """Return sorted unique common elements for two lists.
    >>> common([1, 4, 3, 34, 653, 2, 5], [5, 7, 1, 5, 9, 653, 121])
    [1, 5, 653]
    """
''',
        test_code='''
assert common([1, 4, 3, 34, 653, 2, 5], [5, 7, 1, 5, 9, 653, 121]) == [1, 5, 653]
assert common([5, 3, 2, 8], [3, 2]) == [2, 3]
assert common([4, 3, 2, 8], [3, 2, 4]) == [2, 3, 4]
assert common([4, 3, 2, 8], []) == []
''',
        entry_point="common",
    ),
    CodingProblem(
        task_id="largest_prime_factor",
        prompt='''def largest_prime_factor(n: int) -> int:
    """Return the largest prime factor of n. Assume n > 1 and is not a prime.
    >>> largest_prime_factor(13195)
    29
    >>> largest_prime_factor(2048)
    2
    """
''',
        test_code='''
assert largest_prime_factor(15) == 5
assert largest_prime_factor(27) == 3
assert largest_prime_factor(63) == 7
assert largest_prime_factor(330) == 11
assert largest_prime_factor(13195) == 29
''',
        entry_point="largest_prime_factor",
    ),
    CodingProblem(
        task_id="sum_to_n",
        prompt='''def sum_to_n(n: int) -> int:
    """Return the sum of numbers from 1 to n.
    >>> sum_to_n(30)
    465
    >>> sum_to_n(100)
    5050
    >>> sum_to_n(1)
    1
    """
''',
        test_code='''
assert sum_to_n(1) == 1
assert sum_to_n(6) == 21
assert sum_to_n(11) == 66
assert sum_to_n(30) == 465
assert sum_to_n(100) == 5050
''',
        entry_point="sum_to_n",
    ),
    CodingProblem(
        task_id="correct_bracketing",
        prompt='''def correct_bracketing(brackets: str) -> bool:
    """brackets is a string of "(" and ")".
    Return True if every opening bracket has a corresponding closing bracket.
    >>> correct_bracketing("()")
    True
    >>> correct_bracketing("(()())")
    True
    >>> correct_bracketing(")(()")
    False
    """
''',
        test_code='''
assert correct_bracketing("()") == True
assert correct_bracketing("(()())") == True
assert correct_bracketing("()()(()())()") == True
assert correct_bracketing("()()((()()())())(()()(()))") == True
assert correct_bracketing("((()())))") == False
assert correct_bracketing(")(()") == False
assert correct_bracketing("(") == False
assert correct_bracketing("((((") == False
assert correct_bracketing(")") == False
''',
        entry_point="correct_bracketing",
    ),
]


def get_all_problems() -> list[CodingProblem]:
    """Return all coding problems."""
    return PROBLEMS


def get_problem_by_id(task_id: str) -> CodingProblem | None:
    """Get a specific problem by its ID."""
    for p in PROBLEMS:
        if p.task_id == task_id:
            return p
    return None
