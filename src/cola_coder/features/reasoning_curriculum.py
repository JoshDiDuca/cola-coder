"""Reasoning Curriculum: curriculum learning for progressively harder reasoning tasks.

Trains the model through five ordered reasoning stages — from simple variable tracing
up to full algorithm design. A curriculum scheduler advances to the next stage when
accuracy meets the stage's minimum threshold.

For a TS dev: think of this like leveling up in a game — you have to pass a test at
the current level before you can unlock the next one. Each stage has its own pool of
training examples and a pass-rate gate.

CLI flag: --curriculum
"""

import random
from dataclasses import dataclass, field

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# CurriculumStage dataclass
# ---------------------------------------------------------------------------

@dataclass
class CurriculumStage:
    """One level in the reasoning curriculum.

    Attributes:
        name: Short identifier for the stage (e.g. "variable_tracing").
        difficulty: Integer 1-5; 1 = easiest, 5 = hardest.
        description: Human-readable description of what this stage covers.
        min_accuracy: Pass-rate threshold required to advance past this stage.
        examples: List of training example dicts for this stage.
    """
    name: str
    difficulty: int
    description: str
    min_accuracy: float
    examples: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Default stage definitions
# ---------------------------------------------------------------------------

def _default_stages() -> list[CurriculumStage]:
    """Return the five default reasoning curriculum stages."""

    variable_tracing_examples = [
        {
            "prompt": "Trace the value of x after: x = 5; x = x + 3; x = x * 2",
            "answer": "16",
            "reasoning": "x starts at 5, then 5+3=8, then 8*2=16",
            "difficulty": 1,
        },
        {
            "prompt": "What is the value of y after: y = 10; y -= 4; y //= 2",
            "answer": "3",
            "reasoning": "y=10, then 10-4=6, then 6//2=3",
            "difficulty": 1,
        },
        {
            "prompt": "After a, b = 1, 2; a, b = b, a+b — what are a and b?",
            "answer": "a=2, b=3",
            "reasoning": "swap: a gets old b=2, b gets old a+b=1+2=3",
            "difficulty": 1,
        },
        {
            "prompt": "Track: lst = [1,2,3]; lst.append(4); lst[0] = 9 — what is lst?",
            "answer": "[9, 2, 3, 4]",
            "reasoning": "append adds 4 at end, then index 0 set to 9",
            "difficulty": 1,
        },
        {
            "prompt": "Trace: s = 'hello'; s = s.upper(); s = s[:3] — what is s?",
            "answer": "'HEL'",
            "reasoning": "'hello'.upper()='HELLO', then slice [:3]='HEL'",
            "difficulty": 1,
        },
        {
            "prompt": "What is result after: result = 0; for i in range(4): result += i",
            "answer": "6",
            "reasoning": "0+0+1+2+3=6",
            "difficulty": 1,
        },
        {
            "prompt": "Trace: d = {}; d['a'] = 1; d['b'] = 2; d['a'] += 5 — what is d['a']?",
            "answer": "6",
            "reasoning": "d['a'] is set to 1 then incremented by 5 giving 6",
            "difficulty": 1,
        },
        {
            "prompt": "After n = 7; n = n ** 2; n = n - 1 — what is n?",
            "answer": "48",
            "reasoning": "7**2=49, then 49-1=48",
            "difficulty": 1,
        },
        {
            "prompt": "Trace: x, y = 3, 4; z = x * x + y * y — what is z?",
            "answer": "25",
            "reasoning": "3*3=9, 4*4=16, 9+16=25",
            "difficulty": 1,
        },
        {
            "prompt": "After: flag = True; flag = not flag; flag = not flag — what is flag?",
            "answer": "True",
            "reasoning": "True -> False -> True (two negations cancel)",
            "difficulty": 1,
        },
    ]

    conditional_logic_examples = [
        {
            "prompt": "What does this return? def f(x): return 'pos' if x > 0 else ('neg' if x < 0 else 'zero') — call f(-3)",
            "answer": "'neg'",
            "reasoning": "-3 < 0 so second branch returns 'neg'",
            "difficulty": 2,
        },
        {
            "prompt": "Evaluate: x=5; y=10; result = x if x > y else y — what is result?",
            "answer": "10",
            "reasoning": "5 > 10 is False so result = y = 10",
            "difficulty": 2,
        },
        {
            "prompt": "What is output? if 3 > 2 and 4 < 5: print('yes') else: print('no')",
            "answer": "'yes'",
            "reasoning": "3>2 is True AND 4<5 is True so condition is True",
            "difficulty": 2,
        },
        {
            "prompt": "Find output: x=4; y=0; z = x/y if y != 0 else -1",
            "answer": "-1",
            "reasoning": "y==0 so the else branch gives z=-1",
            "difficulty": 2,
        },
        {
            "prompt": "What does check(8) return? def check(n): return 'big' if n >= 10 else 'small' if n >= 5 else 'tiny'",
            "answer": "'small'",
            "reasoning": "8 < 10 so not 'big'; 8 >= 5 so 'small'",
            "difficulty": 2,
        },
        {
            "prompt": "Evaluate: a=True; b=False; c = a or b; d = a and b — values of c and d?",
            "answer": "c=True, d=False",
            "reasoning": "True or False = True; True and False = False",
            "difficulty": 2,
        },
        {
            "prompt": "Output of: score=72; grade = 'A' if score>=90 else 'B' if score>=80 else 'C' if score>=70 else 'F'",
            "answer": "'C'",
            "reasoning": "72 < 80 and 72 < 90 but 72 >= 70 so grade='C'",
            "difficulty": 2,
        },
        {
            "prompt": "What is x after: x=1; x = x+1 if x % 2 == 0 else x*2",
            "answer": "2",
            "reasoning": "x=1 is odd so x*2=2",
            "difficulty": 2,
        },
        {
            "prompt": "Result of: not (True and False) or (False and True)?",
            "answer": "True",
            "reasoning": "True and False = False; not False = True; True or False = True",
            "difficulty": 2,
        },
        {
            "prompt": "Given: vals=[3,7,2,9]; result = [v for v in vals if v > 4] — what is result?",
            "answer": "[7, 9]",
            "reasoning": "Only 7 and 9 satisfy v > 4",
            "difficulty": 2,
        },
    ]

    loop_reasoning_examples = [
        {
            "prompt": "What does this produce? total=0; i=1; while i<=5: total+=i; i+=1",
            "answer": "15",
            "reasoning": "1+2+3+4+5=15",
            "difficulty": 3,
        },
        {
            "prompt": "Count iterations: for i in range(2, 10, 3): pass",
            "answer": "3",
            "reasoning": "i takes values 2, 5, 8 — three iterations",
            "difficulty": 3,
        },
        {
            "prompt": "Output: res=[]; for i in range(5):\n  if i%2==0: res.append(i)",
            "answer": "[0, 2, 4]",
            "reasoning": "Even values in range(5) are 0, 2, 4",
            "difficulty": 3,
        },
        {
            "prompt": "What is n after: n=64; while n > 1: n //= 2",
            "answer": "1",
            "reasoning": "64->32->16->8->4->2->1, loop stops at 1",
            "difficulty": 3,
        },
        {
            "prompt": "Output: for i in range(3):\n  for j in range(3):\n    if i==j: print(i) — how many lines printed?",
            "answer": "3",
            "reasoning": "Diagonal (0,0),(1,1),(2,2) — three matches",
            "difficulty": 3,
        },
        {
            "prompt": "What is result? result=1; for _ in range(10): result *= 2",
            "answer": "1024",
            "reasoning": "2^10 = 1024",
            "difficulty": 3,
        },
        {
            "prompt": "After: s=''; for c in 'hello': s = c + s — what is s?",
            "answer": "'olleh'",
            "reasoning": "Each char is prepended, reversing 'hello'",
            "difficulty": 3,
        },
        {
            "prompt": "Result: nums=[1,2,3,4,5]; nums = [x**2 for x in nums if x%2!=0]",
            "answer": "[1, 9, 25]",
            "reasoning": "Odd numbers 1,3,5 squared give 1,9,25",
            "difficulty": 3,
        },
        {
            "prompt": "Find first duplicate in [2,4,6,4,8]: loop with a seen set",
            "answer": "4",
            "reasoning": "4 appears at index 1 and again at index 3 — first duplicate",
            "difficulty": 3,
        },
        {
            "prompt": "How many times does the inner loop body run? for i in range(4): for j in range(i): pass",
            "answer": "6",
            "reasoning": "i=0:0, i=1:1, i=2:2, i=3:3 — total 0+1+2+3=6",
            "difficulty": 3,
        },
    ]

    recursion_examples = [
        {
            "prompt": "What does fib(5) return? def fib(n): return n if n<=1 else fib(n-1)+fib(n-2)",
            "answer": "5",
            "reasoning": "fib sequence: 0,1,1,2,3,5 — fib(5)=5",
            "difficulty": 4,
        },
        {
            "prompt": "Result of fact(4)? def fact(n): return 1 if n==0 else n*fact(n-1)",
            "answer": "24",
            "reasoning": "4*3*2*1=24",
            "difficulty": 4,
        },
        {
            "prompt": "What does flatten([1,[2,[3,4]],5]) return for recursive flatten?",
            "answer": "[1, 2, 3, 4, 5]",
            "reasoning": "Recurse into nested lists, collecting all leaves",
            "difficulty": 4,
        },
        {
            "prompt": "How many recursive calls does sum_list([1,2,3]) make? def sum_list(l): return 0 if not l else l[0]+sum_list(l[1:])",
            "answer": "4",
            "reasoning": "One call per element plus the base case: [1,2,3], [2,3], [3], [] = 4 calls",
            "difficulty": 4,
        },
        {
            "prompt": "What is the maximum recursion depth for merge_sort on 8 elements?",
            "answer": "3",
            "reasoning": "log2(8)=3 — the tree has depth 3",
            "difficulty": 4,
        },
        {
            "prompt": "Result: power(2,10)? def power(b,e): return 1 if e==0 else b*power(b,e-1)",
            "answer": "1024",
            "reasoning": "2^10 = 1024",
            "difficulty": 4,
        },
        {
            "prompt": "Trace: count_digits(1234)? def count_digits(n): return 0 if n==0 else 1+count_digits(n//10)",
            "answer": "4",
            "reasoning": "1234->123->12->1->0, four recursive steps plus base",
            "difficulty": 4,
        },
        {
            "prompt": "What is gcd(48,18) via Euclid recursion? def gcd(a,b): return a if b==0 else gcd(b,a%b)",
            "answer": "6",
            "reasoning": "gcd(48,18)->gcd(18,12)->gcd(12,6)->gcd(6,0)=6",
            "difficulty": 4,
        },
        {
            "prompt": "Identify the base case missing from: def countdown(n): print(n); countdown(n-1)",
            "answer": "if n <= 0: return",
            "reasoning": "Without a stopping condition it recurses infinitely",
            "difficulty": 4,
        },
        {
            "prompt": "What does tree_height return for a single-node tree? def tree_height(node): return 0 if not node else 1+max(tree_height(node.left),tree_height(node.right))",
            "answer": "1",
            "reasoning": "Single node has no children, max(0,0)+1=1",
            "difficulty": 4,
        },
    ]

    algorithm_design_examples = [
        {
            "prompt": "Design an O(n log n) algorithm to find all pairs in an array that sum to a target k.",
            "answer": "Sort the array, then use two pointers from both ends meeting in the middle.",
            "reasoning": "Sort is O(n log n); two-pointer scan is O(n). Space O(1) extra.",
            "difficulty": 5,
        },
        {
            "prompt": "What dynamic programming recurrence solves the 0/1 knapsack problem?",
            "answer": "dp[i][w] = max(dp[i-1][w], dp[i-1][w-weight[i]] + value[i]) if weight[i]<=w else dp[i-1][w]",
            "reasoning": "Either skip item i (carry forward) or include it if it fits.",
            "difficulty": 5,
        },
        {
            "prompt": "Describe BFS vs DFS trade-offs for finding the shortest path in an unweighted graph.",
            "answer": "BFS guarantees shortest path in unweighted graphs; DFS does not but uses O(depth) memory vs O(width) for BFS.",
            "reasoning": "BFS explores level-by-level so first path found is shortest; DFS may go deep before finding a path.",
            "difficulty": 5,
        },
        {
            "prompt": "Given a sorted array of n integers with possible duplicates, find the count of a target in O(log n).",
            "answer": "Binary search for leftmost occurrence and rightmost occurrence, then count = right - left + 1.",
            "reasoning": "Two binary searches each O(log n); total O(log n).",
            "difficulty": 5,
        },
        {
            "prompt": "Design an LRU cache with O(1) get and put using built-in data structures.",
            "answer": "Use an ordered dict (Python OrderedDict or a dict + doubly-linked list). Move accessed keys to the end; evict from the front.",
            "reasoning": "Dict gives O(1) lookup; linked list gives O(1) eviction and promotion.",
            "difficulty": 5,
        },
        {
            "prompt": "What greedy strategy solves the activity selection problem (maximize non-overlapping intervals)?",
            "answer": "Sort by end time, then greedily pick each interval whose start >= last picked end.",
            "reasoning": "Choosing the interval that finishes earliest leaves maximum room for future intervals.",
            "difficulty": 5,
        },
        {
            "prompt": "How do you detect a cycle in a directed graph?",
            "answer": "DFS with three states: unvisited, in-stack, done. A back edge (to an in-stack node) means a cycle.",
            "reasoning": "Coloring nodes white/gray/black distinguishes tree edges from back edges.",
            "difficulty": 5,
        },
        {
            "prompt": "Express the time complexity of building a heap from n elements.",
            "answer": "O(n)",
            "reasoning": "Bottom-up heapify sums to O(n) because most nodes are near the leaves with small subtrees.",
            "difficulty": 5,
        },
        {
            "prompt": "What algorithm finds strongly connected components in a directed graph in linear time?",
            "answer": "Kosaraju's or Tarjan's algorithm, both O(V+E).",
            "reasoning": "Two DFS passes (Kosaraju) or one DFS with a stack tracking discovery/low times (Tarjan).",
            "difficulty": 5,
        },
        {
            "prompt": "Design a sliding-window algorithm to find the maximum sum subarray of length k.",
            "answer": "Compute sum of first k elements; slide the window by adding the next element and removing the leftmost; track the running maximum.",
            "reasoning": "Each element enters and leaves the window exactly once — O(n) time, O(1) space.",
            "difficulty": 5,
        },
    ]

    return [
        CurriculumStage(
            name="variable_tracing",
            difficulty=1,
            description="Trace variable values through simple sequential assignments and basic expressions.",
            min_accuracy=0.90,
            examples=variable_tracing_examples,
        ),
        CurriculumStage(
            name="conditional_logic",
            difficulty=2,
            description="Reason about if/else branches, boolean expressions, and ternary logic.",
            min_accuracy=0.80,
            examples=conditional_logic_examples,
        ),
        CurriculumStage(
            name="loop_reasoning",
            difficulty=3,
            description="Trace loop execution, count iterations, and evaluate accumulator patterns.",
            min_accuracy=0.70,
            examples=loop_reasoning_examples,
        ),
        CurriculumStage(
            name="recursion",
            difficulty=4,
            description="Understand recursive call stacks, base cases, and recursive decomposition.",
            min_accuracy=0.60,
            examples=recursion_examples,
        ),
        CurriculumStage(
            name="algorithm_design",
            difficulty=5,
            description="Design and analyze algorithms: time/space complexity, DP, graphs, greedy strategies.",
            min_accuracy=0.50,
            examples=algorithm_design_examples,
        ),
    ]


# ---------------------------------------------------------------------------
# ReasoningCurriculum
# ---------------------------------------------------------------------------

class ReasoningCurriculum:
    """Manages curriculum learning over progressively harder reasoning stages.

    Stages advance when the measured accuracy meets or exceeds a stage's
    min_accuracy threshold. Progress is tracked and can be reset at any time.

    Usage::

        curriculum = ReasoningCurriculum()
        stage = curriculum.current_stage()
        examples = curriculum.get_training_examples(n=8)
        # ... train, evaluate, get accuracy ...
        advanced = curriculum.advance(accuracy=0.92)
    """

    def __init__(self, stages: list[CurriculumStage] | None = None) -> None:
        """Initialise with default or custom stages.

        Args:
            stages: Optional list of CurriculumStage objects. If None,
                    the five default stages are used.
        """
        self._stages: list[CurriculumStage] = stages if stages is not None else _default_stages()
        if not self._stages:
            raise ValueError("stages must be a non-empty list")
        self._stage_index: int = 0
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Core navigation API
    # ------------------------------------------------------------------

    def current_stage(self) -> CurriculumStage:
        """Return the current CurriculumStage."""
        return self._stages[self._stage_index]

    def should_advance(self, accuracy: float) -> bool:
        """Check whether accuracy meets the threshold to advance, without advancing.

        Args:
            accuracy: Measured accuracy/pass-rate in [0.0, 1.0].

        Returns:
            True if accuracy >= current stage's min_accuracy AND there is a
            next stage to advance to; False otherwise.
        """
        if self._stage_index >= len(self._stages) - 1:
            return False
        return accuracy >= self._stages[self._stage_index].min_accuracy

    def advance(self, accuracy: float) -> bool:
        """Advance to the next stage if accuracy meets the threshold.

        Records the advancement in the internal history log.

        Args:
            accuracy: Measured accuracy/pass-rate in [0.0, 1.0].

        Returns:
            True if the curriculum advanced to a new stage; False otherwise.
        """
        if not self.should_advance(accuracy):
            return False
        old_stage = self._stages[self._stage_index]
        self._stage_index += 1
        new_stage = self._stages[self._stage_index]
        self._history.append(
            {
                "from_stage": old_stage.name,
                "to_stage": new_stage.name,
                "accuracy": accuracy,
            }
        )
        return True

    # ------------------------------------------------------------------
    # Training examples
    # ------------------------------------------------------------------

    def get_training_examples(self, n: int = 10) -> list[dict]:
        """Return up to n training examples from the current stage.

        If the stage has fewer than n examples they are sampled with
        replacement so the caller always receives exactly n items.

        Args:
            n: Number of examples to return.

        Returns:
            List of example dicts appropriate for the current difficulty level.
        """
        pool = self._stages[self._stage_index].examples
        if not pool:
            return []
        if len(pool) >= n:
            return random.sample(pool, n)
        # Sample with replacement to fill n
        return random.choices(pool, k=n)

    # ------------------------------------------------------------------
    # Progress & state
    # ------------------------------------------------------------------

    def progress(self) -> dict:
        """Return a progress snapshot for the current curriculum state.

        Returns:
            Dict with keys:
              - current: zero-based current stage index
              - stage: same as current (alias for test compatibility)
              - total: total number of stages
              - stage_name: name of the current stage
              - completion: percentage of stages completed (0-100)
              - history: list of advancement records
        """
        total = len(self._stages)
        completion = round(self._stage_index / total * 100, 1)
        return {
            "current": self._stage_index,
            "stage": self._stage_index,
            "total": total,
            "stage_name": self._stages[self._stage_index].name,
            "completion": completion,
            "history": list(self._history),
        }

    def reset(self) -> None:
        """Reset the curriculum back to stage 0, clearing advancement history."""
        self._stage_index = 0
        self._history = []

    def summary(self) -> dict:
        """Return a high-level summary of the curriculum configuration and state.

        Returns:
            Dict with keys:
              - num_stages: total number of stages
              - current_stage_index: current position
              - current_stage_name: name of the current stage
              - current_difficulty: difficulty level (1-5)
              - stages: list of stage summary dicts
              - advancements: number of times the curriculum has advanced
        """
        return {
            "num_stages": len(self._stages),
            "current_stage_index": self._stage_index,
            "current_stage_name": self._stages[self._stage_index].name,
            "current_difficulty": self._stages[self._stage_index].difficulty,
            "stages": [
                {
                    "name": s.name,
                    "difficulty": s.difficulty,
                    "description": s.description,
                    "min_accuracy": s.min_accuracy,
                    "num_examples": len(s.examples),
                }
                for s in self._stages
            ],
            "advancements": len(self._history),
        }
