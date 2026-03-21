"""Tests for code_pattern_miner.py (feature 57)."""

import pytest

from cola_coder.features.code_pattern_miner import (
    FEATURE_ENABLED,
    CodePatternMiner,
    Pattern,
    _get_function_signature,
    _is_mutable_default,
    is_enabled,
)
import ast


@pytest.fixture
def miner():
    return CodePatternMiner()


SIMPLE_SAMPLE = """\
def add(a, b):
    return a + b

class Foo:
    pass
"""

IDIOM_SAMPLE = """\
squares = [x**2 for x in range(10)]
mapping = {k: v for k, v in zip("abc", [1, 2, 3])}
gen = (x for x in range(5))
cond = x if x > 0 else -x
fn = lambda x: x + 1
"""

ANTI_PATTERN_SAMPLE = """\
def bad(x, items=[]):
    try:
        return x / 0
    except:
        pass

assert (True, False)
"""

SYNTAX_ERROR_SAMPLE = "def bad(:\n    pass\n"


def test_feature_flag():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_mine_empty_corpus(miner):
    result = miner.mine([])
    assert result.total_files == 0
    assert result.total_nodes == 0


def test_mine_invalid_syntax_skipped(miner):
    result = miner.mine([SYNTAX_ERROR_SAMPLE])
    assert result.total_files == 0


def test_mine_counts_constructs(miner):
    result = miner.mine([SIMPLE_SAMPLE])
    assert result.total_files == 1
    sigs = [p.signature for p in result.top_constructs]
    assert any("def" in s for s in sigs)


def test_mine_counts_class(miner):
    result = miner.mine([SIMPLE_SAMPLE])
    sigs = [p.signature for p in result.top_constructs]
    assert any("class" in s for s in sigs)


def test_mine_idioms_list_comp(miner):
    result = miner.mine([IDIOM_SAMPLE])
    idiom_names = [p.signature for p in result.top_idioms]
    assert "list_comprehension" in idiom_names


def test_mine_idioms_dict_comp(miner):
    result = miner.mine([IDIOM_SAMPLE])
    idiom_names = [p.signature for p in result.top_idioms]
    assert "dict_comprehension" in idiom_names


def test_mine_idioms_lambda(miner):
    result = miner.mine([IDIOM_SAMPLE])
    idiom_names = [p.signature for p in result.top_idioms]
    assert "lambda" in idiom_names


def test_mine_anti_pattern_bare_except(miner):
    result = miner.mine([ANTI_PATTERN_SAMPLE])
    anti_names = [p.signature for p in result.anti_patterns]
    assert "bare_except" in anti_names


def test_mine_anti_pattern_mutable_default(miner):
    result = miner.mine([ANTI_PATTERN_SAMPLE])
    anti_names = [p.signature for p in result.anti_patterns]
    assert "mutable_default_arg" in anti_names


def test_mine_subtrees_non_empty(miner):
    result = miner.mine([SIMPLE_SAMPLE])
    assert len(result.top_subtrees) > 0


def test_pattern_as_dict():
    p = Pattern(signature="list_comprehension", count=5, examples=["[x for x in r]"])
    d = p.as_dict()
    assert d["signature"] == "list_comprehension"
    assert d["count"] == 5
    assert len(d["examples"]) == 1


def test_summary_string(miner):
    result = miner.mine([SIMPLE_SAMPLE, IDIOM_SAMPLE])
    s = result.summary()
    assert "files=" in s
    assert "nodes=" in s


def test_find_pattern_list_comp(miner):
    code = "[x for x in range(10)]\n[y*2 for y in range(5)]\n"
    count = miner.find_pattern(code, "list_comprehension")
    assert count == 2


def test_find_pattern_unknown_returns_zero(miner):
    count = miner.find_pattern("x = 1\n", "nonexistent_pattern")
    assert count == 0


def test_find_pattern_syntax_error_returns_zero(miner):
    count = miner.find_pattern("def bad(:\n", "list_comprehension")
    assert count == 0


def test_compare_corpora(miner):
    a = ["[x for x in range(10)]\n"]
    b = ["[x for x in range(10)]\n[y for y in range(5)]\n"]
    diff = miner.compare_corpora(a, b)
    assert "list_comprehension" in diff
    assert diff["list_comprehension"]["delta"] > 0


def test_get_function_signature():
    code = "def foo(a, b, c): pass\n"
    tree = ast.parse(code)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    sig = _get_function_signature(node)
    assert "def" in sig
    assert "3_args" in sig


def test_is_mutable_default_true():
    code = "def f(x, items=[]): pass\n"
    tree = ast.parse(code)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    assert _is_mutable_default(node) is True


def test_is_mutable_default_false():
    code = "def f(x, y=0): pass\n"
    tree = ast.parse(code)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    assert _is_mutable_default(node) is False
