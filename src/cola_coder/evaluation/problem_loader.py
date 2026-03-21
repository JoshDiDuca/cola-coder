"""Problem loader for GRPO training — unified problem set with difficulty support.

Provides a ProblemSet class that can aggregate problems from multiple sources,
filter by difficulty, and return them in curriculum order (easy → medium → hard).

For a TS dev: think of ProblemSet like an Array utility class — it wraps a list
of problems and gives you fluent filter/sort/sample methods.

Usage:
    from cola_coder.evaluation.problem_loader import ProblemSet

    # Load all built-in problems
    ps = ProblemSet().add_builtin()

    # Curriculum order (easy first)
    for problem in ps.curriculum():
        ...

    # Only easy problems
    easy_ps = ps.filter_by_difficulty("easy")

    # Random batch for training
    batch = ps.get_batch(n=8)

Feature flag:
    FEATURE_ENABLED = True
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from .humaneval import (
    CodingProblem,
    get_all_problems_including_extended,
)

# Feature flag — always enabled; kept here for consistency with other feature modules
FEATURE_ENABLED = True

_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def is_enabled() -> bool:
    """Return True if the expanded problem set feature is enabled."""
    return FEATURE_ENABLED


class ProblemSet:
    """Unified collection of coding problems from one or more sources.

    Supports:
    - Loading built-in problems (add_builtin)
    - Loading custom problems from JSONL files (add_from_jsonl)
    - Filtering by difficulty or language
    - Curriculum ordering (easy → medium → hard)
    - Random batches for training

    ProblemSet is immutable-ish — filter/sort methods return NEW ProblemSet
    instances so the original is never modified (similar to TypeScript Array
    methods like .filter() and .sort()).
    """

    def __init__(self, problems: list[CodingProblem] | None = None) -> None:
        self._problems: list[CodingProblem] = list(problems) if problems else []
        self._sources: list[str] = []

    # ------------------------------------------------------------------
    # Loading methods (mutate self and return self for chaining)
    # ------------------------------------------------------------------

    def add_builtin(self, extended: bool = True) -> "ProblemSet":
        """Add built-in problems.

        Args:
            extended: If True (default), load all 62 problems (original 20 +
                      extended 42). If False, load only the original 20 problems.

        Returns:
            self, for method chaining.
        """
        from .humaneval import get_all_problems

        if extended:
            new_problems = get_all_problems_including_extended()
            self._sources.append("builtin_extended")
        else:
            new_problems = get_all_problems()
            self._sources.append("builtin")

        existing_ids = {p.task_id for p in self._problems}
        for p in new_problems:
            if p.task_id not in existing_ids:
                self._problems.append(p)
                existing_ids.add(p.task_id)

        return self

    def add_from_jsonl(self, path: str | Path) -> "ProblemSet":
        """Load problems from a JSONL file.

        Each line must be a JSON object with at minimum:
            task_id, prompt, test_code, entry_point
        Optional fields: difficulty, category, language, canonical_solution

        Args:
            path: Path to the .jsonl file.

        Returns:
            self, for method chaining.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If a line is missing required fields.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Problem file not found: {path}")

        required_fields = {"task_id", "prompt", "test_code", "entry_point"}
        existing_ids = {p.task_id for p in self._problems}
        loaded = 0

        with path.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}") from e

                missing = required_fields - set(data.keys())
                if missing:
                    raise ValueError(
                        f"Line {line_num} of {path} is missing required fields: {missing}"
                    )

                task_id = data["task_id"]
                if task_id in existing_ids:
                    continue  # Skip duplicates

                problem = CodingProblem(
                    task_id=task_id,
                    prompt=data["prompt"],
                    test_code=data["test_code"],
                    entry_point=data["entry_point"],
                    difficulty=data.get("difficulty", "medium"),
                    category=data.get("category", "general"),
                    language=data.get("language", "python"),
                    canonical_solution=data.get("canonical_solution", ""),
                )
                self._problems.append(problem)
                existing_ids.add(task_id)
                loaded += 1

        self._sources.append(f"jsonl:{path.name}({loaded})")
        return self

    # ------------------------------------------------------------------
    # Filtering / sorting — return new ProblemSet instances
    # ------------------------------------------------------------------

    def filter_by_difficulty(self, *levels: str) -> "ProblemSet":
        """Return a new ProblemSet containing only problems at the given difficulty level(s).

        Args:
            *levels: One or more of "easy", "medium", "hard".

        Returns:
            New ProblemSet with the filtered problems.

        Example:
            easy_set = ps.filter_by_difficulty("easy")
            easy_medium = ps.filter_by_difficulty("easy", "medium")
        """
        valid = {"easy", "medium", "hard"}
        for level in levels:
            if level not in valid:
                raise ValueError(f"Invalid difficulty '{level}'. Must be one of {valid}.")

        level_set = set(levels)
        filtered = [p for p in self._problems if p.difficulty in level_set]
        new_ps = ProblemSet(filtered)
        new_ps._sources = self._sources[:]
        return new_ps

    def filter_by_language(self, language: str) -> "ProblemSet":
        """Return a new ProblemSet containing only problems in the given language.

        Args:
            language: e.g. "python" or "typescript".

        Returns:
            New ProblemSet with the filtered problems.
        """
        filtered = [p for p in self._problems if p.language == language]
        new_ps = ProblemSet(filtered)
        new_ps._sources = self._sources[:]
        return new_ps

    def filter_by_category(self, category: str) -> "ProblemSet":
        """Return a new ProblemSet containing only problems in the given category."""
        filtered = [p for p in self._problems if p.category == category]
        new_ps = ProblemSet(filtered)
        new_ps._sources = self._sources[:]
        return new_ps

    def curriculum(self) -> "ProblemSet":
        """Return a new ProblemSet sorted easy → medium → hard (curriculum learning).

        Within the same difficulty level, the original insertion order is preserved
        (stable sort). This mirrors Scaf-GRPO's progressive difficulty strategy.

        Returns:
            New ProblemSet sorted by difficulty.
        """
        sorted_problems = sorted(
            self._problems,
            key=lambda p: _DIFFICULTY_ORDER.get(p.difficulty, 1),
        )
        new_ps = ProblemSet(sorted_problems)
        new_ps._sources = self._sources[:]
        return new_ps

    def shuffle(self, seed: int = 42) -> "ProblemSet":
        """Return a new ProblemSet with problems in a reproducible random order.

        Args:
            seed: Random seed for reproducibility (default: 42).

        Returns:
            New ProblemSet with shuffled problems.
        """
        rng = random.Random(seed)
        shuffled = self._problems[:]
        rng.shuffle(shuffled)
        new_ps = ProblemSet(shuffled)
        new_ps._sources = self._sources[:]
        return new_ps

    # ------------------------------------------------------------------
    # Batch / sampling
    # ------------------------------------------------------------------

    def get_batch(self, n: int, seed: int | None = None) -> list[CodingProblem]:
        """Return a random batch of n problems.

        If n >= len(self), returns all problems shuffled.
        If seed is None, uses non-deterministic randomness.

        Args:
            n: Number of problems to return.
            seed: Optional random seed for reproducibility.

        Returns:
            List of CodingProblem instances.
        """
        if n <= 0:
            return []
        rng = random.Random(seed)
        pool = self._problems[:]
        if n >= len(pool):
            rng.shuffle(pool)
            return pool
        return rng.sample(pool, n)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_jsonl(self, path: str | Path) -> None:
        """Write all problems to a JSONL file.

        Args:
            path: Output file path. Parent directories must exist.
        """
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            for p in self._problems:
                f.write(json.dumps(asdict(p)) + "\n")

    def to_training_dicts(self) -> list[dict]:
        """Convert problems to training-ready dicts for GRPOTrainer.

        Returns:
            List of {'prompt': str, 'test_code': str} dicts.
        """
        return [{"prompt": p.prompt, "test_code": p.test_code} for p in self._problems]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def difficulty_counts(self) -> dict[str, int]:
        """Return a dict of {difficulty: count}."""
        counts: dict[str, int] = {}
        for p in self._problems:
            counts[p.difficulty] = counts.get(p.difficulty, 0) + 1
        return counts

    def category_counts(self) -> dict[str, int]:
        """Return a dict of {category: count}."""
        counts: dict[str, int] = {}
        for p in self._problems:
            counts[p.category] = counts.get(p.category, 0) + 1
        return counts

    def summary(self) -> str:
        """Return a human-readable summary string."""
        d = self.difficulty_counts()
        easy = d.get("easy", 0)
        medium = d.get("medium", 0)
        hard = d.get("hard", 0)
        sources = ", ".join(self._sources) if self._sources else "none"
        return (
            f"ProblemSet({len(self._problems)} problems: "
            f"{easy} easy / {medium} medium / {hard} hard, "
            f"sources=[{sources}])"
        )

    # ------------------------------------------------------------------
    # Python protocols
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._problems)

    def __iter__(self) -> Iterator[CodingProblem]:
        return iter(self._problems)

    def __getitem__(self, index: int) -> CodingProblem:
        return self._problems[index]

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Module-level convenience factory
# ---------------------------------------------------------------------------


def load_problem_set(
    source: str = "builtin",
    jsonl_path: str | None = None,
    difficulty: str | None = None,
    max_problems: int = 0,
    curriculum: bool = False,
    seed: int = 42,
) -> ProblemSet:
    """Convenience function to build a ProblemSet from common configurations.

    Args:
        source: One of:
            "builtin"  — original 20 problems only (backward-compatible)
            "extended" — all 62 built-in problems (original + extended)
            "jsonl"    — load from jsonl_path
            "all"      — extended built-ins (alias for "extended")
        jsonl_path: Path to a JSONL file (required when source="jsonl").
        difficulty: If set, filter to "easy", "medium", or "hard".
        max_problems: If > 0, sample this many problems randomly (after filtering).
        curriculum: If True, return problems sorted easy → medium → hard.
        seed: Random seed for sampling/shuffling.

    Returns:
        Configured ProblemSet.
    """
    ps = ProblemSet()

    if source in ("builtin",):
        ps.add_builtin(extended=False)
    elif source in ("extended", "all"):
        ps.add_builtin(extended=True)
    elif source == "jsonl":
        if not jsonl_path:
            raise ValueError("jsonl_path is required when source='jsonl'")
        ps.add_from_jsonl(jsonl_path)
    else:
        raise ValueError(
            f"Unknown source '{source}'. Choose: 'builtin', 'extended', 'all', 'jsonl'."
        )

    if difficulty:
        ps = ps.filter_by_difficulty(difficulty)

    if curriculum:
        ps = ps.curriculum()

    if max_problems > 0 and len(ps) > max_problems:
        sampled = ps.get_batch(max_problems, seed=seed)
        ps = ProblemSet(sampled)

    return ps
