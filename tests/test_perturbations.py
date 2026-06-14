"""Tests for semantically-preserving docstring perturbations (EVAL-030).

Hermetic — pure-logic only, no GPU, no model, no checkpoint loads. Builds
``CodingProblem`` objects directly from the real dataclass.
"""

from __future__ import annotations

import ast

import pytest

from cola_coder.evaluation.humaneval import CodingProblem
from cola_coder.evaluation.perturbations import (
    ALL_KINDS,
    PerturbedProblem,
    perturb_docstring,
    perturb_problem_set,
)

_PROMPT = '''def add_two(a: int, b: int) -> int:
    """Return the sum of the given two numbers.

    The function should compute a simple integer addition.
    >>> add_two(1, 2)
    3
    >>> add_two(5, 5)
    10
    """
'''

_TEST_CODE = "assert add_two(1, 2) == 3\nassert add_two(5, 5) == 10\n"


def _make_problem(prompt: str = _PROMPT) -> CodingProblem:
    return CodingProblem(
        task_id="add_two",
        prompt=prompt,
        test_code=_TEST_CODE,
        entry_point="add_two",
        canonical_solution="    return a + b\n",
    )


def _sig_line(prompt: str) -> str:
    return prompt.splitlines()[0]


def _doctest_lines(prompt: str) -> list[str]:
    return [ln.strip() for ln in prompt.splitlines() if ln.strip().startswith(">>>")]


def _defines_entry(prompt: str, entry_point: str) -> bool:
    tree = ast.parse(prompt)
    return any(
        isinstance(n, ast.FunctionDef) and n.name == entry_point for n in tree.body
    )


def test_clean_is_first_and_unmodified() -> None:
    problem = _make_problem()
    variants = perturb_docstring(problem)
    assert variants[0].perturbation == "clean"
    assert variants[0].problem.prompt == problem.prompt
    assert all(isinstance(v, PerturbedProblem) for v in variants)


def test_signature_line_byte_identical_across_all_variants() -> None:
    problem = _make_problem()
    sig = _sig_line(problem.prompt)
    for variant in perturb_docstring(problem):
        assert _sig_line(variant.problem.prompt) == sig


def test_entry_point_and_test_code_never_mutated() -> None:
    problem = _make_problem()
    for variant in perturb_docstring(problem):
        assert variant.problem.entry_point == problem.entry_point
        assert variant.problem.test_code == problem.test_code
        assert variant.base_task_id == problem.task_id


def test_every_variant_still_defines_entry_point_via_ast() -> None:
    problem = _make_problem()
    for variant in perturb_docstring(problem):
        assert _defines_entry(variant.problem.prompt, "add_two")


@pytest.mark.parametrize("kind", list(ALL_KINDS))
def test_each_kind_mutates_docstring_but_keeps_signature(kind: str) -> None:
    problem = _make_problem()
    variants = perturb_docstring(problem, kinds=[kind])
    # clean + (possibly) the single kind
    assert variants[0].perturbation == "clean"
    if len(variants) == 2:
        variant = variants[1]
        assert variant.perturbation == kind
        # The prompt changed somewhere...
        assert variant.problem.prompt != problem.prompt
        # ...but the signature line did not.
        assert _sig_line(variant.problem.prompt) == _sig_line(problem.prompt)


def test_reorder_examples_is_a_permutation_of_the_same_doctest_lines() -> None:
    problem = _make_problem()
    variants = perturb_docstring(problem, kinds=["reorder_examples"])
    reordered = [v for v in variants if v.perturbation == "reorder_examples"]
    assert reordered, "reorder_examples should produce a variant for a 2-example docstring"
    original = sorted(_doctest_lines(problem.prompt))
    mutated = sorted(_doctest_lines(reordered[0].problem.prompt))
    assert original == mutated  # same set of >>> calls, just reordered


def test_paraphrase_swaps_known_synonyms_preserving_casing() -> None:
    problem = _make_problem()
    variants = perturb_docstring(problem, kinds=["paraphrase"])
    para = next(v for v in variants if v.perturbation == "paraphrase")
    text = para.problem.prompt
    # "Return" -> "Output" (capital preserved); "given" -> "provided"
    assert "Output" in text
    assert "provided" in text
    assert "Return the sum" not in text


def test_seed_determinism() -> None:
    problem = _make_problem()
    a = perturb_docstring(problem, seed=123)
    b = perturb_docstring(problem, seed=123)
    assert [v.problem.prompt for v in a] == [v.problem.prompt for v in b]


def test_different_seeds_can_differ_for_typo() -> None:
    problem = _make_problem()
    a = perturb_docstring(problem, kinds=["typo"], seed=1)
    b = perturb_docstring(problem, kinds=["typo"], seed=99)
    # Not a hard guarantee for every seed pair, but these two should differ.
    assert a[-1].problem.prompt != b[-1].problem.prompt


def test_no_docstring_returns_clean_only() -> None:
    prompt = "def noop(x):\n    return x\n"
    problem = CodingProblem(
        task_id="noop", prompt=prompt, test_code="assert noop(1) == 1", entry_point="noop"
    )
    variants = perturb_docstring(problem)
    assert len(variants) == 1
    assert variants[0].perturbation == "clean"
    assert variants[0].problem.prompt == prompt


def test_empty_docstring_returns_clean_only() -> None:
    prompt = 'def f(x):\n    """"""\n    return x\n'
    problem = CodingProblem(
        task_id="f", prompt=prompt, test_code="assert f(1) == 1", entry_point="f"
    )
    variants = perturb_docstring(problem)
    assert len(variants) == 1


def test_unknown_kind_raises() -> None:
    problem = _make_problem()
    with pytest.raises(ValueError):
        perturb_docstring(problem, kinds=["not_a_kind"])


def test_perturb_problem_set_keys_by_task_id() -> None:
    p1 = _make_problem()
    p2 = CodingProblem(
        task_id="square",
        prompt='def square(n):\n    """Return n squared."""\n',
        test_code="assert square(3) == 9",
        entry_point="square",
    )
    out = perturb_problem_set([p1, p2])
    assert set(out.keys()) == {"add_two", "square"}
    assert out["add_two"][0].perturbation == "clean"
    assert all(v.base_task_id == "square" for v in out["square"])
