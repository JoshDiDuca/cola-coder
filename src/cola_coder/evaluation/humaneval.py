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

from dataclasses import dataclass, field


@dataclass
class CodingProblem:
    """A single coding problem for evaluation."""

    task_id: str  # Unique identifier
    prompt: str  # Function signature + docstring (model sees this)
    test_code: str  # Test cases (model doesn't see this)
    entry_point: str  # Function name to test
    difficulty: str = "medium"  # "easy" | "medium" | "hard"
    category: str = "general"  # "string" | "array" | "math" | "algorithm" | ...
    language: str = "python"  # "python" | "typescript"
    canonical_solution: str = field(default="", repr=False)  # Reference solution


# ---------------------------------------------------------------------------
# Original 20 problems (backward-compatible, now with difficulty/category tags)
# ---------------------------------------------------------------------------
PROBLEMS: list[CodingProblem] = [
    CodingProblem(
        task_id="has_close_elements",
        difficulty="easy",
        category="array",
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
        canonical_solution='''
    if len(numbers) < 2:
        return False
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
''',
    ),
    CodingProblem(
        task_id="separate_paren_groups",
        difficulty="medium",
        category="string",
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
        canonical_solution='''
    result = []
    current = ''
    depth = 0
    for ch in paren_string.replace(' ', ''):
        current += ch
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                result.append(current)
                current = ''
    return result
''',
    ),
    CodingProblem(
        task_id="truncate_number",
        difficulty="easy",
        category="math",
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
        canonical_solution='''
    return number % 1.0
''',
    ),
    CodingProblem(
        task_id="below_zero",
        difficulty="easy",
        category="array",
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
        canonical_solution='''
    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False
''',
    ),
    CodingProblem(
        task_id="mean_absolute_deviation",
        difficulty="easy",
        category="math",
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
        canonical_solution='''
    mean = sum(numbers) / len(numbers)
    return sum(abs(x - mean) for x in numbers) / len(numbers)
''',
    ),
    CodingProblem(
        task_id="intersperse",
        difficulty="easy",
        category="array",
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
        canonical_solution='''
    if not numbers:
        return []
    result = [numbers[0]]
    for n in numbers[1:]:
        result.append(delimiter)
        result.append(n)
    return result
''',
    ),
    CodingProblem(
        task_id="parse_nested_parens",
        difficulty="medium",
        category="string",
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
        canonical_solution='''
    def max_depth(s):
        depth = max_d = 0
        for ch in s:
            if ch == '(':
                depth += 1
                max_d = max(max_d, depth)
            elif ch == ')':
                depth -= 1
        return max_d
    return [max_depth(g) for g in paren_string.split()]
''',
    ),
    CodingProblem(
        task_id="filter_by_substring",
        difficulty="easy",
        category="string",
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
        canonical_solution='''
    return [s for s in strings if substring in s]
''',
    ),
    CodingProblem(
        task_id="sum_product",
        difficulty="easy",
        category="math",
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
        canonical_solution='''
    total = 0
    product = 1
    for n in numbers:
        total += n
        product *= n
    return (total, product)
''',
    ),
    CodingProblem(
        task_id="rolling_max",
        difficulty="easy",
        category="array",
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
        canonical_solution='''
    result = []
    current_max = None
    for n in numbers:
        if current_max is None or n > current_max:
            current_max = n
        result.append(current_max)
    return result
''',
    ),
    CodingProblem(
        task_id="is_palindrome",
        difficulty="easy",
        category="string",
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
        canonical_solution='''
    return string == string[::-1]
''',
    ),
    CodingProblem(
        task_id="remove_vowels",
        difficulty="easy",
        category="string",
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
        canonical_solution='''
    return ''.join(c for c in text if c not in 'aeiouAEIOU')
''',
    ),
    CodingProblem(
        task_id="below_threshold",
        difficulty="easy",
        category="array",
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
        canonical_solution='''
    return all(x < t for x in l)
''',
    ),
    CodingProblem(
        task_id="add_elements",
        difficulty="easy",
        category="math",
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
        canonical_solution='''
    return x + y
''',
    ),
    CodingProblem(
        task_id="same_chars",
        difficulty="easy",
        category="string",
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
        canonical_solution='''
    return set(s0) == set(s1)
''',
    ),
    CodingProblem(
        task_id="fib",
        difficulty="easy",
        category="math",
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
        canonical_solution='''
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
''',
    ),
    CodingProblem(
        task_id="common_elements",
        difficulty="easy",
        category="array",
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
        canonical_solution='''
    return sorted(set(l1) & set(l2))
''',
    ),
    CodingProblem(
        task_id="largest_prime_factor",
        difficulty="medium",
        category="math",
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
        canonical_solution='''
    largest = 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            largest = d
            n //= d
        d += 1
    if n > 1:
        largest = n
    return largest
''',
    ),
    CodingProblem(
        task_id="sum_to_n",
        difficulty="easy",
        category="math",
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
        canonical_solution='''
    return n * (n + 1) // 2
''',
    ),
    CodingProblem(
        task_id="correct_bracketing",
        difficulty="medium",
        category="string",
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
        canonical_solution='''
    depth = 0
    for ch in brackets:
        if ch == '(':
            depth += 1
        else:
            depth -= 1
        if depth < 0:
            return False
    return depth == 0
''',
    ),
]

# ---------------------------------------------------------------------------
# Extended problem set — 42 additional problems
# ---------------------------------------------------------------------------
EXTENDED_PROBLEMS: list[CodingProblem] = [
    # ---- Easy: string manipulation ----
    CodingProblem(
        task_id="count_vowels",
        difficulty="easy",
        category="string",
        prompt='''def count_vowels(s: str) -> int:
    """Count the number of vowels (a, e, i, o, u — case insensitive) in the string.
    >>> count_vowels("Hello World")
    3
    >>> count_vowels("bcdfg")
    0
    """
''',
        test_code='''
assert count_vowels("Hello World") == 3
assert count_vowels("bcdfg") == 0
assert count_vowels("AEIOU") == 5
assert count_vowels("") == 0
assert count_vowels("Python") == 2
''',
        entry_point="count_vowels",
        canonical_solution='''
    return sum(1 for c in s if c.lower() in 'aeiou')
''',
    ),
    CodingProblem(
        task_id="reverse_words",
        difficulty="easy",
        category="string",
        prompt='''def reverse_words(sentence: str) -> str:
    """Reverse the order of words in a sentence. Words are separated by spaces.
    >>> reverse_words("Hello World")
    'World Hello'
    >>> reverse_words("one two three")
    'three two one'
    """
''',
        test_code='''
assert reverse_words("Hello World") == "World Hello"
assert reverse_words("one two three") == "three two one"
assert reverse_words("single") == "single"
assert reverse_words("a b c d") == "d c b a"
''',
        entry_point="reverse_words",
        canonical_solution='''
    return ' '.join(reversed(sentence.split()))
''',
    ),
    CodingProblem(
        task_id="capitalize_words",
        difficulty="easy",
        category="string",
        prompt='''def capitalize_words(sentence: str) -> str:
    """Capitalize the first letter of each word in the sentence.
    >>> capitalize_words("hello world")
    'Hello World'
    >>> capitalize_words("python is great")
    'Python Is Great'
    """
''',
        test_code='''
assert capitalize_words("hello world") == "Hello World"
assert capitalize_words("python is great") == "Python Is Great"
assert capitalize_words("single") == "Single"
assert capitalize_words("already Capitalized") == "Already Capitalized"
''',
        entry_point="capitalize_words",
        canonical_solution='''
    return ' '.join(w.capitalize() for w in sentence.split())
''',
    ),
    CodingProblem(
        task_id="count_occurrences",
        difficulty="easy",
        category="string",
        prompt='''def count_occurrences(text: str, char: str) -> int:
    """Count how many times a character appears in a string (case sensitive).
    >>> count_occurrences("hello world", "l")
    3
    >>> count_occurrences("aabbcc", "a")
    2
    """
''',
        test_code='''
assert count_occurrences("hello world", "l") == 3
assert count_occurrences("aabbcc", "a") == 2
assert count_occurrences("", "x") == 0
assert count_occurrences("mississippi", "s") == 4
assert count_occurrences("Python", "p") == 0
assert count_occurrences("Python", "P") == 1
''',
        entry_point="count_occurrences",
        canonical_solution='''
    return text.count(char)
''',
    ),
    CodingProblem(
        task_id="is_anagram",
        difficulty="easy",
        category="string",
        prompt='''def is_anagram(s1: str, s2: str) -> bool:
    """Check if s1 and s2 are anagrams of each other (case insensitive, ignore spaces).
    >>> is_anagram("listen", "silent")
    True
    >>> is_anagram("hello", "world")
    False
    """
''',
        test_code='''
assert is_anagram("listen", "silent") == True
assert is_anagram("hello", "world") == False
assert is_anagram("Astronomer", "Moon starer") == True
assert is_anagram("abc", "cba") == True
assert is_anagram("abc", "abcd") == False
''',
        entry_point="is_anagram",
        canonical_solution='''
    def normalize(s):
        return sorted(s.lower().replace(' ', ''))
    return normalize(s1) == normalize(s2)
''',
    ),
    # ---- Easy: array / list ----
    CodingProblem(
        task_id="flatten_list",
        difficulty="easy",
        category="array",
        prompt='''def flatten_list(nested: list[list]) -> list:
    """Flatten a one-level nested list into a single list.
    >>> flatten_list([[1, 2], [3, 4], [5]])
    [1, 2, 3, 4, 5]
    >>> flatten_list([[], [1], [2, 3]])
    [1, 2, 3]
    """
''',
        test_code='''
assert flatten_list([[1, 2], [3, 4], [5]]) == [1, 2, 3, 4, 5]
assert flatten_list([[], [1], [2, 3]]) == [1, 2, 3]
assert flatten_list([]) == []
assert flatten_list([[1, 2, 3]]) == [1, 2, 3]
assert flatten_list([[1], [2], [3]]) == [1, 2, 3]
''',
        entry_point="flatten_list",
        canonical_solution='''
    return [item for sublist in nested for item in sublist]
''',
    ),
    CodingProblem(
        task_id="unique_elements",
        difficulty="easy",
        category="array",
        prompt='''def unique_elements(lst: list) -> list:
    """Return a list containing only the unique elements of lst, preserving order.
    >>> unique_elements([1, 2, 2, 3, 1, 4])
    [1, 2, 3, 4]
    >>> unique_elements([])
    []
    """
''',
        test_code='''
assert unique_elements([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]
assert unique_elements([]) == []
assert unique_elements([1, 1, 1]) == [1]
assert unique_elements([3, 1, 2]) == [3, 1, 2]
assert unique_elements(["a", "b", "a", "c"]) == ["a", "b", "c"]
''',
        entry_point="unique_elements",
        canonical_solution='''
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
''',
    ),
    CodingProblem(
        task_id="chunk_list",
        difficulty="easy",
        category="array",
        prompt='''def chunk_list(lst: list, size: int) -> list[list]:
    """Split a list into chunks of given size. The last chunk may be smaller.
    >>> chunk_list([1, 2, 3, 4, 5], 2)
    [[1, 2], [3, 4], [5]]
    >>> chunk_list([1, 2, 3], 3)
    [[1, 2, 3]]
    """
''',
        test_code='''
assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
assert chunk_list([1, 2, 3], 3) == [[1, 2, 3]]
assert chunk_list([], 2) == []
assert chunk_list([1], 5) == [[1]]
assert chunk_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]
''',
        entry_point="chunk_list",
        canonical_solution='''
    return [lst[i:i + size] for i in range(0, len(lst), size)]
''',
    ),
    CodingProblem(
        task_id="find_duplicates",
        difficulty="easy",
        category="array",
        prompt='''def find_duplicates(lst: list[int]) -> list[int]:
    """Return a sorted list of all elements that appear more than once in lst.
    >>> find_duplicates([1, 2, 3, 2, 4, 3])
    [2, 3]
    >>> find_duplicates([1, 2, 3])
    []
    """
''',
        test_code='''
assert find_duplicates([1, 2, 3, 2, 4, 3]) == [2, 3]
assert find_duplicates([1, 2, 3]) == []
assert find_duplicates([]) == []
assert find_duplicates([5, 5, 5]) == [5]
assert find_duplicates([1, 1, 2, 2, 3]) == [1, 2]
''',
        entry_point="find_duplicates",
        canonical_solution='''
    from collections import Counter
    counts = Counter(lst)
    return sorted(k for k, v in counts.items() if v > 1)
''',
    ),
    # ---- Easy: math ----
    CodingProblem(
        task_id="is_prime",
        difficulty="easy",
        category="math",
        prompt='''def is_prime(n: int) -> bool:
    """Return True if n is a prime number.
    >>> is_prime(7)
    True
    >>> is_prime(10)
    False
    >>> is_prime(1)
    False
    """
''',
        test_code='''
assert is_prime(2) == True
assert is_prime(3) == True
assert is_prime(4) == False
assert is_prime(7) == True
assert is_prime(10) == False
assert is_prime(1) == False
assert is_prime(97) == True
assert is_prime(100) == False
''',
        entry_point="is_prime",
        canonical_solution='''
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
''',
    ),
    CodingProblem(
        task_id="gcd",
        difficulty="easy",
        category="math",
        prompt='''def gcd(a: int, b: int) -> int:
    """Return the greatest common divisor of a and b using Euclid's algorithm.
    >>> gcd(48, 18)
    6
    >>> gcd(100, 75)
    25
    """
''',
        test_code='''
assert gcd(48, 18) == 6
assert gcd(100, 75) == 25
assert gcd(7, 3) == 1
assert gcd(0, 5) == 5
assert gcd(12, 12) == 12
''',
        entry_point="gcd",
        canonical_solution='''
    while b:
        a, b = b, a % b
    return a
''',
    ),
    CodingProblem(
        task_id="celsius_to_fahrenheit",
        difficulty="easy",
        category="math",
        prompt='''def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit.
    Formula: F = C * 9/5 + 32
    >>> celsius_to_fahrenheit(0)
    32.0
    >>> celsius_to_fahrenheit(100)
    212.0
    """
''',
        test_code='''
assert celsius_to_fahrenheit(0) == 32.0
assert celsius_to_fahrenheit(100) == 212.0
assert celsius_to_fahrenheit(-40) == -40.0
assert abs(celsius_to_fahrenheit(37) - 98.6) < 1e-6
''',
        entry_point="celsius_to_fahrenheit",
        canonical_solution='''
    return celsius * 9 / 5 + 32
''',
    ),
    # ---- Medium: algorithms ----
    CodingProblem(
        task_id="binary_search",
        difficulty="medium",
        category="algorithm",
        prompt='''def binary_search(sorted_list: list[int], target: int) -> int:
    """Search for target in a sorted list using binary search.
    Return the index of target, or -1 if not found.
    >>> binary_search([1, 3, 5, 7, 9], 5)
    2
    >>> binary_search([1, 3, 5, 7, 9], 6)
    -1
    """
''',
        test_code='''
assert binary_search([1, 3, 5, 7, 9], 5) == 2
assert binary_search([1, 3, 5, 7, 9], 6) == -1
assert binary_search([], 1) == -1
assert binary_search([1], 1) == 0
assert binary_search([1], 2) == -1
assert binary_search([1, 2, 3, 4, 5], 1) == 0
assert binary_search([1, 2, 3, 4, 5], 5) == 4
''',
        entry_point="binary_search",
        canonical_solution='''
    lo, hi = 0, len(sorted_list) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_list[mid] == target:
            return mid
        elif sorted_list[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''',
    ),
    CodingProblem(
        task_id="bubble_sort",
        difficulty="medium",
        category="algorithm",
        prompt='''def bubble_sort(lst: list[int]) -> list[int]:
    """Sort a list of integers in ascending order using bubble sort.
    Return a new sorted list (do not modify the original).
    >>> bubble_sort([5, 3, 8, 1, 2])
    [1, 2, 3, 5, 8]
    >>> bubble_sort([1])
    [1]
    """
''',
        test_code='''
assert bubble_sort([5, 3, 8, 1, 2]) == [1, 2, 3, 5, 8]
assert bubble_sort([1]) == [1]
assert bubble_sort([]) == []
assert bubble_sort([3, 1, 2]) == [1, 2, 3]
assert bubble_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]
original = [3, 1, 2]
bubble_sort(original)
assert original == [3, 1, 2]
''',
        entry_point="bubble_sort",
        canonical_solution='''
    lst = lst[:]
    n = len(lst)
    for i in range(n):
        for j in range(0, n - i - 1):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
    return lst
''',
    ),
    CodingProblem(
        task_id="merge_sorted_lists",
        difficulty="medium",
        category="algorithm",
        prompt='''def merge_sorted_lists(l1: list[int], l2: list[int]) -> list[int]:
    """Merge two sorted lists into a single sorted list.
    >>> merge_sorted_lists([1, 3, 5], [2, 4, 6])
    [1, 2, 3, 4, 5, 6]
    >>> merge_sorted_lists([], [1, 2])
    [1, 2]
    """
''',
        test_code='''
assert merge_sorted_lists([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
assert merge_sorted_lists([], [1, 2]) == [1, 2]
assert merge_sorted_lists([1, 2], []) == [1, 2]
assert merge_sorted_lists([], []) == []
assert merge_sorted_lists([1, 1, 2], [1, 3]) == [1, 1, 1, 2, 3]
''',
        entry_point="merge_sorted_lists",
        canonical_solution='''
    result = []
    i = j = 0
    while i < len(l1) and j < len(l2):
        if l1[i] <= l2[j]:
            result.append(l1[i])
            i += 1
        else:
            result.append(l2[j])
            j += 1
    result.extend(l1[i:])
    result.extend(l2[j:])
    return result
''',
    ),
    CodingProblem(
        task_id="two_sum",
        difficulty="medium",
        category="algorithm",
        prompt='''def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:
    """Find two numbers in nums that add up to target.
    Return a tuple of their indices (smaller index first), or None if not found.
    Each input has at most one solution. The same element may not be used twice.
    >>> two_sum([2, 7, 11, 15], 9)
    (0, 1)
    >>> two_sum([1, 2, 3], 10)
    None
    """
''',
        test_code='''
assert two_sum([2, 7, 11, 15], 9) == (0, 1)
assert two_sum([1, 2, 3], 10) is None
assert two_sum([3, 2, 4], 6) == (1, 2)
assert two_sum([3, 3], 6) == (0, 1)
assert two_sum([], 5) is None
''',
        entry_point="two_sum",
        canonical_solution='''
    seen = {}
    for i, n in enumerate(nums):
        complement = target - n
        if complement in seen:
            j = seen[complement]
            return (min(i, j), max(i, j))
        seen[n] = i
    return None
''',
    ),
    CodingProblem(
        task_id="max_subarray_sum",
        difficulty="medium",
        category="algorithm",
        prompt='''def max_subarray_sum(nums: list[int]) -> int:
    """Find the contiguous subarray with the largest sum and return that sum.
    (Kadane's algorithm). Assume the list is non-empty.
    >>> max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4])
    6
    >>> max_subarray_sum([1])
    1
    """
''',
        test_code='''
assert max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
assert max_subarray_sum([1]) == 1
assert max_subarray_sum([-1, -2, -3]) == -1
assert max_subarray_sum([5, 4, -1, 7, 8]) == 23
assert max_subarray_sum([1, 2, 3, 4, 5]) == 15
''',
        entry_point="max_subarray_sum",
        canonical_solution='''
    max_sum = current_sum = nums[0]
    for n in nums[1:]:
        current_sum = max(n, current_sum + n)
        max_sum = max(max_sum, current_sum)
    return max_sum
''',
    ),
    CodingProblem(
        task_id="valid_parentheses",
        difficulty="medium",
        category="algorithm",
        prompt='''def valid_parentheses(s: str) -> bool:
    """Determine if the input string containing '(', ')', '{', '}', '[' and ']'
    is valid. An input string is valid if open brackets are closed by the same
    type of brackets and in the correct order.
    >>> valid_parentheses("()")
    True
    >>> valid_parentheses("()[]{}")
    True
    >>> valid_parentheses("(]")
    False
    """
''',
        test_code='''
assert valid_parentheses("()") == True
assert valid_parentheses("()[]{}") == True
assert valid_parentheses("(]") == False
assert valid_parentheses("([)]") == False
assert valid_parentheses("{[]}") == True
assert valid_parentheses("") == True
assert valid_parentheses("[") == False
''',
        entry_point="valid_parentheses",
        canonical_solution='''
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in '([{':
            stack.append(ch)
        elif ch in ')]}':
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return len(stack) == 0
''',
    ),
    CodingProblem(
        task_id="count_islands",
        difficulty="hard",
        category="algorithm",
        prompt='''def count_islands(grid: list[list[str]]) -> int:
    """Count the number of islands in a 2D grid of '1' (land) and '0' (water).
    An island is a group of adjacent (up, down, left, right) land cells.
    >>> count_islands([["1","1","0"],["0","1","0"],["0","0","1"]])
    2
    >>> count_islands([["1","1","1"],["1","1","1"],["1","1","1"]])
    1
    """
''',
        test_code='''
assert count_islands([["1","1","0"],["0","1","0"],["0","0","1"]]) == 2
assert count_islands([["1","1","1"],["1","1","1"],["1","1","1"]]) == 1
assert count_islands([["0","0","0"],["0","0","0"]]) == 0
assert count_islands([["1","0","1"],["0","0","0"],["1","0","1"]]) == 4
assert count_islands([]) == 0
''',
        entry_point="count_islands",
        canonical_solution='''
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    grid = [row[:] for row in grid]
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return
        grid[r][c] = '0'
        dfs(r + 1, c); dfs(r - 1, c)
        dfs(r, c + 1); dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                dfs(r, c)
                count += 1
    return count
''',
    ),
    # ---- Medium: data structures ----
    CodingProblem(
        task_id="implement_stack",
        difficulty="medium",
        category="data_structure",
        prompt='''def simulate_stack(operations: list[tuple]) -> list:
    """Simulate a stack with a list of operations and return the result list.

    Operations are tuples:
      ("push", value) — push value onto stack
      ("pop",)        — pop from stack; if empty, append None to results
      ("peek",)       — peek at top; if empty, append None to results
      ("size",)       — append current size to results

    For pop/peek/size operations, append the result to an output list.
    Return the output list.

    >>> simulate_stack([("push", 1), ("push", 2), ("peek",), ("pop",), ("size",)])
    [2, 2, 1]
    """
''',
        test_code='''
assert simulate_stack([("push", 1), ("push", 2), ("peek",), ("pop",), ("size",)]) == [2, 2, 1]
assert simulate_stack([("pop",)]) == [None]
assert simulate_stack([("size",)]) == [0]
assert simulate_stack([("push", 5), ("push", 10), ("pop",), ("pop",), ("pop",)]) == [10, 5, None]
''',
        entry_point="simulate_stack",
        canonical_solution='''
    stack = []
    results = []
    for op in operations:
        if op[0] == "push":
            stack.append(op[1])
        elif op[0] == "pop":
            results.append(stack.pop() if stack else None)
        elif op[0] == "peek":
            results.append(stack[-1] if stack else None)
        elif op[0] == "size":
            results.append(len(stack))
    return results
''',
    ),
    CodingProblem(
        task_id="lru_cache_simple",
        difficulty="hard",
        category="data_structure",
        prompt='''def simulate_lru_cache(capacity: int, operations: list[tuple]) -> list:
    """Simulate an LRU (Least Recently Used) cache.

    Operations are tuples:
      ("get", key)        — return the value for key or -1 if not found
      ("put", key, value) — insert or update key; evict least recently used
                            item if capacity is exceeded. Returns None.

    Return a list of results for all "get" operations.

    >>> simulate_lru_cache(2, [("put", 1, 1), ("put", 2, 2), ("get", 1), ("put", 3, 3), ("get", 2), ("get", 1)])
    [1, -1, 1]
    """
''',
        test_code='''
assert simulate_lru_cache(2, [("put", 1, 1), ("put", 2, 2), ("get", 1), ("put", 3, 3), ("get", 2), ("get", 1)]) == [1, -1, 1]
assert simulate_lru_cache(1, [("put", 2, 1), ("get", 2), ("put", 3, 2), ("get", 2), ("get", 3)]) == [1, -1, 2]
assert simulate_lru_cache(2, [("get", 1)]) == [-1]
''',
        entry_point="simulate_lru_cache",
        canonical_solution='''
    from collections import OrderedDict
    cache = OrderedDict()
    results = []
    for op in operations:
        if op[0] == "get":
            key = op[1]
            if key in cache:
                cache.move_to_end(key)
                results.append(cache[key])
            else:
                results.append(-1)
        elif op[0] == "put":
            key, value = op[1], op[2]
            if key in cache:
                cache.move_to_end(key)
            cache[key] = value
            if len(cache) > capacity:
                cache.popitem(last=False)
    return results
''',
    ),
    CodingProblem(
        task_id="group_by_key",
        difficulty="medium",
        category="data_structure",
        prompt='''def group_by_key(items: list[dict], key: str) -> dict[str, list]:
    """Group a list of dicts by the value of a given key.
    Items missing the key are placed under the key None.
    Preserve insertion order within each group.
    >>> group_by_key([{"type": "a", "v": 1}, {"type": "b", "v": 2}, {"type": "a", "v": 3}], "type")
    {'a': [{'type': 'a', 'v': 1}, {'type': 'a', 'v': 3}], 'b': [{'type': 'b', 'v': 2}]}
    """
''',
        test_code='''
result = group_by_key([{"type": "a", "v": 1}, {"type": "b", "v": 2}, {"type": "a", "v": 3}], "type")
assert result == {"a": [{"type": "a", "v": 1}, {"type": "a", "v": 3}], "b": [{"type": "b", "v": 2}]}
assert group_by_key([], "type") == {}
result2 = group_by_key([{"x": 1}, {"type": "a"}], "type")
assert result2[None] == [{"x": 1}]
assert result2["a"] == [{"type": "a"}]
''',
        entry_point="group_by_key",
        canonical_solution='''
    groups = {}
    for item in items:
        k = item.get(key)
        groups.setdefault(k, []).append(item)
    return groups
''',
    ),
    # ---- Medium: string problems ----
    CodingProblem(
        task_id="longest_common_prefix",
        difficulty="medium",
        category="string",
        prompt='''def longest_common_prefix(strs: list[str]) -> str:
    """Find the longest common prefix string amongst a list of strings.
    Return an empty string if there is no common prefix.
    >>> longest_common_prefix(["flower", "flow", "flight"])
    'fl'
    >>> longest_common_prefix(["dog", "racecar", "car"])
    ''
    """
''',
        test_code='''
assert longest_common_prefix(["flower", "flow", "flight"]) == "fl"
assert longest_common_prefix(["dog", "racecar", "car"]) == ""
assert longest_common_prefix([""]) == ""
assert longest_common_prefix(["abc"]) == "abc"
assert longest_common_prefix(["abc", "abc"]) == "abc"
assert longest_common_prefix(["abc", "ab", "a"]) == "a"
''',
        entry_point="longest_common_prefix",
        canonical_solution='''
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix
''',
    ),
    CodingProblem(
        task_id="run_length_encoding",
        difficulty="medium",
        category="string",
        prompt='''def run_length_encode(s: str) -> str:
    """Encode a string using run-length encoding.
    Consecutive identical characters are replaced by the character
    followed by its count. Single characters are not followed by 1.
    >>> run_length_encode("aabbbccdddd")
    'a2b3c2d4'
    >>> run_length_encode("abcd")
    'abcd'
    """
''',
        test_code='''
assert run_length_encode("aabbbccdddd") == "a2b3c2d4"
assert run_length_encode("abcd") == "abcd"
assert run_length_encode("") == ""
assert run_length_encode("aaaa") == "a4"
assert run_length_encode("aabbcc") == "a2b2c2"
assert run_length_encode("a") == "a"
''',
        entry_point="run_length_encode",
        canonical_solution='''
    if not s:
        return ""
    result = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            result.append(s[i - 1] + (str(count) if count > 1 else ''))
            count = 1
    result.append(s[-1] + (str(count) if count > 1 else ''))
    return ''.join(result)
''',
    ),
    CodingProblem(
        task_id="word_frequency",
        difficulty="medium",
        category="string",
        prompt='''def word_frequency(text: str) -> dict[str, int]:
    """Count the frequency of each word in a string (case insensitive).
    Words are separated by spaces; strip punctuation from ends of words.
    >>> word_frequency("the cat sat on the mat")
    {'the': 2, 'cat': 1, 'sat': 1, 'on': 1, 'mat': 1}
    """
''',
        test_code='''
assert word_frequency("the cat sat on the mat") == {"the": 2, "cat": 1, "sat": 1, "on": 1, "mat": 1}
assert word_frequency("") == {}
result = word_frequency("Hello hello HELLO")
assert result == {"hello": 3}
result2 = word_frequency("one, two. one!")
assert result2["one"] == 2
assert result2["two"] == 1
''',
        entry_point="word_frequency",
        canonical_solution='''
    import string
    freq = {}
    for word in text.split():
        word = word.strip(string.punctuation).lower()
        if word:
            freq[word] = freq.get(word, 0) + 1
    return freq
''',
    ),
    # ---- Medium: math ----
    CodingProblem(
        task_id="count_primes",
        difficulty="medium",
        category="math",
        prompt='''def count_primes(n: int) -> int:
    """Count the number of prime numbers strictly less than n (Sieve of Eratosthenes).
    >>> count_primes(10)
    4
    >>> count_primes(0)
    0
    """
''',
        test_code='''
assert count_primes(10) == 4
assert count_primes(0) == 0
assert count_primes(1) == 0
assert count_primes(2) == 0
assert count_primes(3) == 1
assert count_primes(100) == 25
''',
        entry_point="count_primes",
        canonical_solution='''
    if n < 2:
        return 0
    sieve = [True] * n
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n, i):
                sieve[j] = False
    return sum(sieve)
''',
    ),
    CodingProblem(
        task_id="power_of_two",
        difficulty="easy",
        category="math",
        prompt='''def is_power_of_two(n: int) -> bool:
    """Return True if n is a power of two.
    >>> is_power_of_two(1)
    True
    >>> is_power_of_two(16)
    True
    >>> is_power_of_two(6)
    False
    """
''',
        test_code='''
assert is_power_of_two(1) == True
assert is_power_of_two(2) == True
assert is_power_of_two(16) == True
assert is_power_of_two(6) == False
assert is_power_of_two(0) == False
assert is_power_of_two(-4) == False
assert is_power_of_two(1024) == True
''',
        entry_point="is_power_of_two",
        canonical_solution='''
    if n <= 0:
        return False
    return (n & (n - 1)) == 0
''',
    ),
    CodingProblem(
        task_id="digit_sum",
        difficulty="easy",
        category="math",
        prompt='''def digit_sum(n: int) -> int:
    """Return the sum of the digits of n. For negative numbers, ignore the sign.
    >>> digit_sum(123)
    6
    >>> digit_sum(-456)
    15
    """
''',
        test_code='''
assert digit_sum(123) == 6
assert digit_sum(-456) == 15
assert digit_sum(0) == 0
assert digit_sum(9999) == 36
assert digit_sum(1001) == 2
''',
        entry_point="digit_sum",
        canonical_solution='''
    return sum(int(d) for d in str(abs(n)))
''',
    ),
    # ---- Medium: practical utilities ----
    CodingProblem(
        task_id="validate_email",
        difficulty="medium",
        category="practical",
        prompt='''def validate_email(email: str) -> bool:
    """Check if an email address has a valid format.
    A valid email has exactly one '@', at least one '.' in the domain part,
    no spaces, and non-empty local and domain parts.
    >>> validate_email("user@example.com")
    True
    >>> validate_email("invalid-email")
    False
    >>> validate_email("@no-local.com")
    False
    """
''',
        test_code='''
assert validate_email("user@example.com") == True
assert validate_email("invalid-email") == False
assert validate_email("@no-local.com") == False
assert validate_email("no-domain@") == False
assert validate_email("two@@signs.com") == False
assert validate_email("user@sub.domain.com") == True
assert validate_email("user @example.com") == False
assert validate_email("user@nodot") == False
''',
        entry_point="validate_email",
        canonical_solution='''
    if ' ' in email or email.count('@') != 1:
        return False
    local, domain = email.split('@')
    if not local or not domain:
        return False
    if '.' not in domain:
        return False
    if domain.startswith('.') or domain.endswith('.'):
        return False
    return True
''',
    ),
    CodingProblem(
        task_id="parse_csv_line",
        difficulty="medium",
        category="practical",
        prompt='''def parse_csv_line(line: str) -> list[str]:
    """Parse a single CSV line into a list of fields.
    Fields may be quoted with double quotes; quoted fields may contain commas.
    Leading/trailing whitespace outside quotes is stripped from each field.
    >>> parse_csv_line('a,b,c')
    ['a', 'b', 'c']
    >>> parse_csv_line('"hello, world",foo,bar')
    ['hello, world', 'foo', 'bar']
    """
''',
        test_code='''
assert parse_csv_line('a,b,c') == ['a', 'b', 'c']
assert parse_csv_line('"hello, world",foo,bar') == ['hello, world', 'foo', 'bar']
assert parse_csv_line('one') == ['one']
assert parse_csv_line(' a , b , c ') == ['a', 'b', 'c']
assert parse_csv_line('"quoted","also, quoted",plain') == ['quoted', 'also, quoted', 'plain']
''',
        entry_point="parse_csv_line",
        canonical_solution='''
    import csv
    reader = csv.reader([line])
    return [field.strip() for field in next(reader)]
''',
    ),
    CodingProblem(
        task_id="flatten_dict",
        difficulty="medium",
        category="practical",
        prompt='''def flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dictionary using dot notation for keys.
    >>> flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})
    {'a.b': 1, 'a.c': 2, 'd': 3}
    >>> flatten_dict({"x": 1})
    {'x': 1}
    """
''',
        test_code='''
assert flatten_dict({"a": {"b": 1, "c": 2}, "d": 3}) == {"a.b": 1, "a.c": 2, "d": 3}
assert flatten_dict({"x": 1}) == {"x": 1}
assert flatten_dict({}) == {}
assert flatten_dict({"a": {"b": {"c": 42}}}) == {"a.b.c": 42}
assert flatten_dict({"a": 1, "b": {"c": 2, "d": {"e": 3}}}) == {"a": 1, "b.c": 2, "b.d.e": 3}
''',
        entry_point="flatten_dict",
        canonical_solution='''
    items = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, full_key))
        else:
            items[full_key] = v
    return items
''',
    ),
    # ---- Hard: algorithms ----
    CodingProblem(
        task_id="longest_increasing_subsequence",
        difficulty="hard",
        category="algorithm",
        prompt='''def longest_increasing_subsequence(nums: list[int]) -> int:
    """Return the length of the longest strictly increasing subsequence.
    (A subsequence is derived by deleting some elements without changing order.)
    >>> longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18])
    4
    >>> longest_increasing_subsequence([0, 1, 0, 3, 2, 3])
    4
    """
''',
        test_code='''
assert longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18]) == 4
assert longest_increasing_subsequence([0, 1, 0, 3, 2, 3]) == 4
assert longest_increasing_subsequence([7, 7, 7, 7]) == 1
assert longest_increasing_subsequence([1, 3, 6, 7, 9, 4, 10, 5, 6]) == 6
assert longest_increasing_subsequence([1]) == 1
''',
        entry_point="longest_increasing_subsequence",
        canonical_solution='''
    import bisect
    tails = []
    for n in nums:
        pos = bisect.bisect_left(tails, n)
        if pos == len(tails):
            tails.append(n)
        else:
            tails[pos] = n
    return len(tails)
''',
    ),
    CodingProblem(
        task_id="coin_change",
        difficulty="hard",
        category="algorithm",
        prompt='''def coin_change(coins: list[int], amount: int) -> int:
    """Return the fewest number of coins needed to make up the amount.
    Return -1 if the amount cannot be made up by any combination of coins.
    >>> coin_change([1, 5, 10, 25], 36)
    3
    >>> coin_change([2], 3)
    -1
    """
''',
        test_code='''
assert coin_change([1, 5, 10, 25], 36) == 3
assert coin_change([2], 3) == -1
assert coin_change([1], 0) == 0
assert coin_change([1, 2, 5], 11) == 3
assert coin_change([186, 419, 83, 408], 6249) == 20
''',
        entry_point="coin_change",
        canonical_solution='''
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
''',
    ),
    CodingProblem(
        task_id="word_break",
        difficulty="hard",
        category="algorithm",
        prompt='''def word_break(s: str, word_dict: list[str]) -> bool:
    """Return True if s can be segmented into a space-separated sequence of
    one or more words from word_dict.
    >>> word_break("leetcode", ["leet", "code"])
    True
    >>> word_break("applepenapple", ["apple", "pen"])
    True
    >>> word_break("catsandog", ["cats", "dog", "sand", "and", "cat"])
    False
    """
''',
        test_code='''
assert word_break("leetcode", ["leet", "code"]) == True
assert word_break("applepenapple", ["apple", "pen"]) == True
assert word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]) == False
assert word_break("", ["a"]) == True
assert word_break("a", ["a"]) == True
assert word_break("bb", ["a", "b", "bbb", "bbbb"]) == True
''',
        entry_point="word_break",
        canonical_solution='''
    word_set = set(word_dict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[n]
''',
    ),
    CodingProblem(
        task_id="trapping_rain_water",
        difficulty="hard",
        category="algorithm",
        prompt='''def trap_rain_water(height: list[int]) -> int:
    """Calculate how much rain water can be trapped between bars.
    Each element represents the height of a bar (width = 1).
    >>> trap_rain_water([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])
    6
    >>> trap_rain_water([4, 2, 0, 3, 2, 5])
    9
    """
''',
        test_code='''
assert trap_rain_water([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
assert trap_rain_water([4, 2, 0, 3, 2, 5]) == 9
assert trap_rain_water([]) == 0
assert trap_rain_water([3, 0, 2, 0, 4]) == 7
assert trap_rain_water([1, 1, 1]) == 0
''',
        entry_point="trap_rain_water",
        canonical_solution='''
    if not height:
        return 0
    left, right = 0, len(height) - 1
    left_max = right_max = water = 0
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    return water
''',
    ),
    CodingProblem(
        task_id="matrix_spiral_order",
        difficulty="hard",
        category="algorithm",
        prompt='''def spiral_order(matrix: list[list[int]]) -> list[int]:
    """Return all elements of an m×n matrix in spiral order (clockwise).
    >>> spiral_order([[1,2,3],[4,5,6],[7,8,9]])
    [1, 2, 3, 6, 9, 8, 7, 4, 5]
    >>> spiral_order([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
    """
''',
        test_code='''
assert spiral_order([[1,2,3],[4,5,6],[7,8,9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]
assert spiral_order([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) == [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
assert spiral_order([[1]]) == [1]
assert spiral_order([]) == []
assert spiral_order([[1, 2], [3, 4]]) == [1, 2, 4, 3]
''',
        entry_point="spiral_order",
        canonical_solution='''
    if not matrix:
        return []
    result = []
    top, bottom, left, right = 0, len(matrix) - 1, 0, len(matrix[0]) - 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            result.append(matrix[top][c])
        top += 1
        for r in range(top, bottom + 1):
            result.append(matrix[r][right])
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                result.append(matrix[bottom][c])
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                result.append(matrix[r][left])
            left += 1
    return result
''',
    ),
    # ---- Medium: more algorithms ----
    CodingProblem(
        task_id="rotate_array",
        difficulty="medium",
        category="algorithm",
        prompt='''def rotate_array(nums: list[int], k: int) -> list[int]:
    """Rotate the array to the right by k steps. Return a new list.
    >>> rotate_array([1, 2, 3, 4, 5, 6, 7], 3)
    [5, 6, 7, 1, 2, 3, 4]
    >>> rotate_array([1, 2], 3)
    [2, 1]
    """
''',
        test_code='''
assert rotate_array([1, 2, 3, 4, 5, 6, 7], 3) == [5, 6, 7, 1, 2, 3, 4]
assert rotate_array([1, 2], 3) == [2, 1]
assert rotate_array([1], 0) == [1]
assert rotate_array([1, 2, 3], 0) == [1, 2, 3]
assert rotate_array([1, 2, 3, 4], 4) == [1, 2, 3, 4]
''',
        entry_point="rotate_array",
        canonical_solution='''
    n = len(nums)
    if n == 0:
        return nums[:]
    k = k % n
    return nums[-k:] + nums[:-k] if k else nums[:]
''',
    ),
    CodingProblem(
        task_id="product_except_self",
        difficulty="medium",
        category="algorithm",
        prompt='''def product_except_self(nums: list[int]) -> list[int]:
    """Return an array where each element is the product of all elements
    in nums except itself. Do not use division.
    >>> product_except_self([1, 2, 3, 4])
    [24, 12, 8, 6]
    >>> product_except_self([-1, 1, 0, -3, 3])
    [0, 0, 9, 0, 0]
    """
''',
        test_code='''
assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
assert product_except_self([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]
assert product_except_self([2, 3]) == [3, 2]
assert product_except_self([1, 1, 1, 1]) == [1, 1, 1, 1]
''',
        entry_point="product_except_self",
        canonical_solution='''
    n = len(nums)
    result = [1] * n
    prefix = 1
    for i in range(n):
        result[i] = prefix
        prefix *= nums[i]
    suffix = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix
        suffix *= nums[i]
    return result
''',
    ),
    CodingProblem(
        task_id="anagram_groups",
        difficulty="medium",
        category="string",
        prompt='''def group_anagrams(strs: list[str]) -> list[list[str]]:
    """Group strings that are anagrams of each other.
    Each group should be sorted internally. The outer list should be sorted
    by the first element of each group.
    >>> group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    [['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]
    """
''',
        test_code='''
assert group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]) == [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]
assert group_anagrams([""]) == [[""]]
assert group_anagrams(["a"]) == [["a"]]
assert group_anagrams(["abc", "bca", "xyz"]) == [["abc", "bca"], ["xyz"]]
''',
        entry_point="group_anagrams",
        canonical_solution='''
    from collections import defaultdict
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    result = [sorted(g) for g in groups.values()]
    return sorted(result, key=lambda g: g[0])
''',
    ),
    CodingProblem(
        task_id="top_k_frequent",
        difficulty="medium",
        category="algorithm",
        prompt='''def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """Return the k most frequent elements in any order.
    If there is a tie, the smaller element should appear first.
    >>> sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2))
    [1, 2]
    >>> top_k_frequent([1], 1)
    [1]
    """
''',
        test_code='''
assert sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2)) == [1, 2]
assert top_k_frequent([1], 1) == [1]
assert sorted(top_k_frequent([1, 2], 2)) == [1, 2]
result = top_k_frequent([4, 1, 2, 4, 2, 4], 2)
assert sorted(result) == [2, 4]
''',
        entry_point="top_k_frequent",
        canonical_solution='''
    from collections import Counter
    counts = Counter(nums)
    return [x for x, _ in counts.most_common(k)]
''',
    ),
    CodingProblem(
        task_id="climbing_stairs",
        difficulty="easy",
        category="algorithm",
        prompt='''def climbing_stairs(n: int) -> int:
    """Count the number of distinct ways to climb n stairs,
    taking either 1 or 2 steps at a time.
    >>> climbing_stairs(2)
    2
    >>> climbing_stairs(3)
    3
    >>> climbing_stairs(10)
    89
    """
''',
        test_code='''
assert climbing_stairs(1) == 1
assert climbing_stairs(2) == 2
assert climbing_stairs(3) == 3
assert climbing_stairs(5) == 8
assert climbing_stairs(10) == 89
''',
        entry_point="climbing_stairs",
        canonical_solution='''
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b
''',
    ),
    CodingProblem(
        task_id="missing_number",
        difficulty="easy",
        category="math",
        prompt='''def missing_number(nums: list[int]) -> int:
    """Given a list containing n distinct numbers taken from 0, 1, 2, ..., n,
    find the one number that is missing from the list.
    >>> missing_number([3, 0, 1])
    2
    >>> missing_number([9,6,4,2,3,5,7,0,1])
    8
    """
''',
        test_code='''
assert missing_number([3, 0, 1]) == 2
assert missing_number([9, 6, 4, 2, 3, 5, 7, 0, 1]) == 8
assert missing_number([0, 1]) == 2
assert missing_number([1]) == 0
assert missing_number([0]) == 1
''',
        entry_point="missing_number",
        canonical_solution='''
    n = len(nums)
    return n * (n + 1) // 2 - sum(nums)
''',
    ),
    CodingProblem(
        task_id="single_number",
        difficulty="easy",
        category="math",
        prompt='''def single_number(nums: list[int]) -> int:
    """Every element in nums appears exactly twice except for one element
    which appears exactly once. Find that single element.
    Use O(1) extra space (hint: XOR).
    >>> single_number([2, 2, 1])
    1
    >>> single_number([4, 1, 2, 1, 2])
    4
    """
''',
        test_code='''
assert single_number([2, 2, 1]) == 1
assert single_number([4, 1, 2, 1, 2]) == 4
assert single_number([1]) == 1
assert single_number([0, 1, 0]) == 1
assert single_number([7, 3, 7]) == 3
''',
        entry_point="single_number",
        canonical_solution='''
    result = 0
    for n in nums:
        result ^= n
    return result
''',
    ),
    CodingProblem(
        task_id="matrix_transpose",
        difficulty="easy",
        category="algorithm",
        prompt='''def transpose_matrix(matrix: list[list[int]]) -> list[list[int]]:
    """Return the transpose of an m×n matrix.
    The transpose swaps rows and columns.
    >>> transpose_matrix([[1, 2, 3], [4, 5, 6]])
    [[1, 4], [2, 5], [3, 6]]
    """
''',
        test_code='''
assert transpose_matrix([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
assert transpose_matrix([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
assert transpose_matrix([[5]]) == [[5]]
assert transpose_matrix([[1, 2, 3]]) == [[1], [2], [3]]
''',
        entry_point="transpose_matrix",
        canonical_solution='''
    if not matrix or not matrix[0]:
        return []
    return [[matrix[r][c] for r in range(len(matrix))] for c in range(len(matrix[0]))]
''',
    ),
    CodingProblem(
        task_id="is_balanced_bst",
        difficulty="hard",
        category="data_structure",
        prompt='''def is_balanced_heights(heights: list[int | None]) -> bool:
    """Check if a binary tree encoded as a level-order list is height-balanced.
    None values represent missing nodes. A height-balanced tree has the heights
    of the two subtrees of every node differ by at most one.
    Index relationships: left child of i is 2i+1, right child is 2i+2.
    >>> is_balanced_heights([3, 9, 20, None, None, 15, 7])
    True
    >>> is_balanced_heights([1, 2, 2, 3, 3, None, None, 4, 4])
    False
    """
''',
        test_code='''
assert is_balanced_heights([3, 9, 20, None, None, 15, 7]) == True
assert is_balanced_heights([1, 2, 2, 3, 3, None, None, 4, 4]) == False
assert is_balanced_heights([]) == True
assert is_balanced_heights([1]) == True
assert is_balanced_heights([1, 2, None, 3]) == False
''',
        entry_point="is_balanced_heights",
        canonical_solution='''
    def height(i):
        if i >= len(heights) or heights[i] is None:
            return 0
        left = height(2 * i + 1)
        right = height(2 * i + 2)
        if left == -1 or right == -1 or abs(left - right) > 1:
            return -1
        return 1 + max(left, right)
    return height(0) != -1
''',
    ),
    CodingProblem(
        task_id="decode_ways",
        difficulty="hard",
        category="algorithm",
        prompt='''def decode_ways(s: str) -> int:
    """A string of digits can be decoded using the mapping '1' -> 'A', ..., '26' -> 'Z'.
    Return the number of ways to decode the string.
    >>> decode_ways("12")
    2
    >>> decode_ways("226")
    3
    >>> decode_ways("06")
    0
    """
''',
        test_code='''
assert decode_ways("12") == 2
assert decode_ways("226") == 3
assert decode_ways("06") == 0
assert decode_ways("0") == 0
assert decode_ways("1") == 1
assert decode_ways("11106") == 2
''',
        entry_point="decode_ways",
        canonical_solution='''
    if not s or s[0] == '0':
        return 0
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
    for i in range(2, n + 1):
        if s[i - 1] != '0':
            dp[i] += dp[i - 1]
        two_digit = int(s[i - 2:i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i - 2]
    return dp[n]
''',
    ),
    CodingProblem(
        task_id="min_path_sum",
        difficulty="hard",
        category="algorithm",
        prompt='''def min_path_sum(grid: list[list[int]]) -> int:
    """Given an m x n grid of non-negative integers, find the path from
    the top-left to the bottom-right that minimizes the sum of all numbers.
    You can only move right or down.
    >>> min_path_sum([[1, 3, 1], [1, 5, 1], [4, 2, 1]])
    7
    >>> min_path_sum([[1, 2, 3], [4, 5, 6]])
    12
    """
''',
        test_code='''
assert min_path_sum([[1, 3, 1], [1, 5, 1], [4, 2, 1]]) == 7
assert min_path_sum([[1, 2, 3], [4, 5, 6]]) == 12
assert min_path_sum([[1]]) == 1
assert min_path_sum([[1, 2], [1, 1]]) == 3
''',
        entry_point="min_path_sum",
        canonical_solution='''
    rows, cols = len(grid), len(grid[0])
    dp = [row[:] for row in grid]
    for c in range(1, cols):
        dp[0][c] += dp[0][c - 1]
    for r in range(1, rows):
        dp[r][0] += dp[r - 1][0]
    for r in range(1, rows):
        for c in range(1, cols):
            dp[r][c] += min(dp[r - 1][c], dp[r][c - 1])
    return dp[rows - 1][cols - 1]
''',
    ),
]

# All problems combined (original 20 + extended 42 = 62 total)
ALL_PROBLEMS: list[CodingProblem] = PROBLEMS + EXTENDED_PROBLEMS


def get_all_problems() -> list[CodingProblem]:
    """Return all coding problems (original + extended). Backward compatible."""
    return PROBLEMS


def get_extended_problems() -> list[CodingProblem]:
    """Return only the extended (non-original) problems."""
    return EXTENDED_PROBLEMS


def get_all_problems_including_extended() -> list[CodingProblem]:
    """Return the full set: original 20 + extended 42 = 62 problems."""
    return ALL_PROBLEMS


def get_problem_by_id(task_id: str) -> CodingProblem | None:
    """Get a specific problem by its ID."""
    for p in ALL_PROBLEMS:
        if p.task_id == task_id:
            return p
    return None


def get_problems_by_difficulty(difficulty: str) -> list[CodingProblem]:
    """Return all problems with the given difficulty ('easy', 'medium', 'hard')."""
    return [p for p in ALL_PROBLEMS if p.difficulty == difficulty]


def get_problems_by_category(category: str) -> list[CodingProblem]:
    """Return all problems in the given category."""
    return [p for p in ALL_PROBLEMS if p.category == category]
