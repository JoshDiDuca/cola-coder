"""Tests for the expanded problem set and ProblemSet loader.

No GPU required — all tests run on CPU against in-memory data structures.
"""

import json

import pytest

from cola_coder.evaluation.humaneval import (
    ALL_PROBLEMS,
    get_all_problems,
    get_all_problems_including_extended,
    get_extended_problems,
    get_problem_by_id,
    get_problems_by_category,
    get_problems_by_difficulty,
)
from cola_coder.evaluation.problem_loader import (
    FEATURE_ENABLED,
    ProblemSet,
    is_enabled,
    load_problem_set,
)

# ---------------------------------------------------------------------------
# 1. Feature flag
# ---------------------------------------------------------------------------


def test_feature_enabled():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


# ---------------------------------------------------------------------------
# 2. Backward compatibility — original 20 problems
# ---------------------------------------------------------------------------


def test_get_all_problems_returns_original_20():
    """get_all_problems() must still return exactly the original 20 problems."""
    problems = get_all_problems()
    assert len(problems) == 20


def test_original_problems_have_required_fields():
    """Every original problem must have the required fields."""
    for p in get_all_problems():
        assert isinstance(p.task_id, str) and p.task_id
        assert isinstance(p.prompt, str) and p.prompt
        assert isinstance(p.test_code, str) and p.test_code
        assert isinstance(p.entry_point, str) and p.entry_point


def test_original_problems_known_ids():
    """Spot-check a few known task IDs to guard against accidental renames."""
    ids = {p.task_id for p in get_all_problems()}
    assert "fib" in ids
    assert "is_palindrome" in ids
    assert "correct_bracketing" in ids


# ---------------------------------------------------------------------------
# 3. Extended problems (new 42)
# ---------------------------------------------------------------------------


def test_extended_problems_count():
    """There should be at least 40 extended problems."""
    assert len(get_extended_problems()) >= 40


def test_all_problems_count():
    """Total should be original + extended >= 60."""
    all_p = get_all_problems_including_extended()
    assert len(all_p) >= 60


def test_new_problems_have_difficulty_tags():
    """Every extended problem must have a valid difficulty tag."""
    valid = {"easy", "medium", "hard"}
    for p in get_extended_problems():
        assert p.difficulty in valid, f"{p.task_id} has invalid difficulty '{p.difficulty}'"


def test_new_problems_have_category_tags():
    """Every extended problem must have a non-empty category."""
    for p in get_extended_problems():
        assert isinstance(p.category, str) and p.category, (
            f"{p.task_id} is missing a category"
        )


def test_original_problems_have_difficulty_tags():
    """Original 20 problems must now also carry difficulty tags."""
    valid = {"easy", "medium", "hard"}
    for p in get_all_problems():
        assert p.difficulty in valid, f"{p.task_id} missing difficulty"


def test_no_duplicate_task_ids():
    """All task IDs across the full set must be unique."""
    ids = [p.task_id for p in ALL_PROBLEMS]
    assert len(ids) == len(set(ids)), "Duplicate task_id found"


def test_get_problems_by_difficulty():
    easy = get_problems_by_difficulty("easy")
    medium = get_problems_by_difficulty("medium")
    hard = get_problems_by_difficulty("hard")

    assert len(easy) >= 1
    assert len(medium) >= 1
    assert len(hard) >= 1
    assert all(p.difficulty == "easy" for p in easy)
    assert all(p.difficulty == "medium" for p in medium)
    assert all(p.difficulty == "hard" for p in hard)


def test_get_problem_by_id():
    p = get_problem_by_id("fib")
    assert p is not None
    assert p.task_id == "fib"

    p2 = get_problem_by_id("nonexistent_xyz")
    assert p2 is None


def test_get_problems_by_category():
    math_problems = get_problems_by_category("math")
    assert len(math_problems) >= 1
    assert all(p.category == "math" for p in math_problems)


# ---------------------------------------------------------------------------
# 4. ProblemSet — loading
# ---------------------------------------------------------------------------


def test_problem_set_add_builtin_extended():
    ps = ProblemSet().add_builtin(extended=True)
    assert len(ps) >= 60


def test_problem_set_add_builtin_original_only():
    ps = ProblemSet().add_builtin(extended=False)
    assert len(ps) == 20


def test_problem_set_no_duplicates_on_double_add():
    """Adding builtin twice should not create duplicates."""
    ps = ProblemSet().add_builtin(extended=True).add_builtin(extended=True)
    ids = [p.task_id for p in ps]
    assert len(ids) == len(set(ids))


def test_problem_set_add_from_jsonl(tmp_path):
    """Load problems from a JSONL file."""
    custom = [
        {
            "task_id": "custom_add",
            "prompt": "def custom_add(a, b):\n    pass\n",
            "test_code": "assert custom_add(1, 2) == 3\n",
            "entry_point": "custom_add",
            "difficulty": "easy",
            "category": "math",
        },
        {
            "task_id": "custom_mult",
            "prompt": "def custom_mult(a, b):\n    pass\n",
            "test_code": "assert custom_mult(3, 4) == 12\n",
            "entry_point": "custom_mult",
        },
    ]
    jsonl_path = tmp_path / "custom.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(d) for d in custom), encoding="utf-8")

    ps = ProblemSet().add_from_jsonl(str(jsonl_path))
    assert len(ps) == 2
    task_ids = {p.task_id for p in ps}
    assert "custom_add" in task_ids
    assert "custom_mult" in task_ids


def test_problem_set_add_from_jsonl_missing_field(tmp_path):
    """JSONL lines missing required fields should raise ValueError."""
    bad_line = json.dumps({"task_id": "no_prompt"})
    path = tmp_path / "bad.jsonl"
    path.write_text(bad_line, encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        ProblemSet().add_from_jsonl(str(path))


def test_problem_set_add_from_jsonl_file_not_found():
    with pytest.raises(FileNotFoundError):
        ProblemSet().add_from_jsonl("/nonexistent/path/problems.jsonl")


# ---------------------------------------------------------------------------
# 5. ProblemSet — filtering
# ---------------------------------------------------------------------------


def test_filter_by_difficulty_easy():
    ps = ProblemSet().add_builtin(extended=True)
    easy = ps.filter_by_difficulty("easy")
    assert len(easy) >= 1
    assert all(p.difficulty == "easy" for p in easy)


def test_filter_by_difficulty_multiple():
    ps = ProblemSet().add_builtin(extended=True)
    easy_medium = ps.filter_by_difficulty("easy", "medium")
    assert all(p.difficulty in ("easy", "medium") for p in easy_medium)


def test_filter_by_difficulty_invalid_raises():
    ps = ProblemSet().add_builtin(extended=True)
    with pytest.raises(ValueError, match="Invalid difficulty"):
        ps.filter_by_difficulty("very_hard")


def test_filter_by_difficulty_returns_new_instance():
    """Filter must not modify the original ProblemSet."""
    ps = ProblemSet().add_builtin(extended=True)
    original_len = len(ps)
    _ = ps.filter_by_difficulty("easy")
    assert len(ps) == original_len  # original unchanged


def test_filter_by_language():
    ps = ProblemSet().add_builtin(extended=True)
    python_ps = ps.filter_by_language("python")
    assert all(p.language == "python" for p in python_ps)


# ---------------------------------------------------------------------------
# 6. Curriculum ordering
# ---------------------------------------------------------------------------


def test_curriculum_ordering():
    """curriculum() must return problems sorted easy → medium → hard."""
    ps = ProblemSet().add_builtin(extended=True).curriculum()
    difficulties = [p.difficulty for p in ps]

    # Once we transition from easy to medium we must not go back to easy;
    # same rule for medium → hard.
    order = {"easy": 0, "medium": 1, "hard": 2}
    prev = 0
    for d in difficulties:
        curr = order.get(d, 1)
        assert curr >= prev, (
            f"Curriculum ordering violated: '{d}' found after difficulty level {prev}"
        )
        prev = curr


def test_curriculum_returns_new_instance():
    ps = ProblemSet().add_builtin(extended=True)
    curriculum_ps = ps.curriculum()
    # Verify curriculum is a separate object (easy problems first)
    assert curriculum_ps is not ps
    assert curriculum_ps[0].difficulty == "easy"


# ---------------------------------------------------------------------------
# 7. get_batch
# ---------------------------------------------------------------------------


def test_get_batch_correct_count():
    ps = ProblemSet().add_builtin(extended=True)
    batch = ps.get_batch(n=5)
    assert len(batch) == 5


def test_get_batch_reproducible_with_seed():
    ps = ProblemSet().add_builtin(extended=True)
    batch_a = ps.get_batch(n=10, seed=99)
    batch_b = ps.get_batch(n=10, seed=99)
    assert [p.task_id for p in batch_a] == [p.task_id for p in batch_b]


def test_get_batch_all_when_n_exceeds_size():
    ps = ProblemSet().add_builtin(extended=False)  # 20 problems
    batch = ps.get_batch(n=100)
    assert len(batch) == 20


def test_get_batch_zero_returns_empty():
    ps = ProblemSet().add_builtin(extended=True)
    assert ps.get_batch(0) == []


# ---------------------------------------------------------------------------
# 8. to_training_dicts
# ---------------------------------------------------------------------------


def test_to_training_dicts_contains_required_keys():
    ps = ProblemSet().add_builtin(extended=True)
    dicts = ps.to_training_dicts()
    assert len(dicts) == len(ps)
    for d in dicts:
        assert "prompt" in d
        assert "test_code" in d


# ---------------------------------------------------------------------------
# 9. load_problem_set convenience factory
# ---------------------------------------------------------------------------


def test_load_problem_set_builtin():
    ps = load_problem_set(source="builtin")
    assert len(ps) == 20


def test_load_problem_set_extended():
    ps = load_problem_set(source="extended")
    assert len(ps) >= 60


def test_load_problem_set_difficulty_filter():
    ps = load_problem_set(source="extended", difficulty="easy")
    assert all(p.difficulty == "easy" for p in ps)


def test_load_problem_set_max_problems():
    ps = load_problem_set(source="extended", max_problems=10)
    assert len(ps) == 10


def test_load_problem_set_curriculum():
    ps = load_problem_set(source="extended", curriculum=True)
    difficulties = [p.difficulty for p in ps]
    order = {"easy": 0, "medium": 1, "hard": 2}
    prev = 0
    for d in difficulties:
        curr = order.get(d, 1)
        assert curr >= prev
        prev = curr


def test_load_problem_set_unknown_source_raises():
    with pytest.raises(ValueError, match="Unknown source"):
        load_problem_set(source="magic")


def test_load_problem_set_jsonl_requires_path():
    with pytest.raises(ValueError, match="jsonl_path is required"):
        load_problem_set(source="jsonl")
