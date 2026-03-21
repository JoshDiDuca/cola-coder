"""Tests for FunctionExtractor (features/function_extractor.py)."""

from __future__ import annotations


from cola_coder.features.function_extractor import (
    FEATURE_ENABLED,
    CallGraphReport,
    FunctionExtractor,
    FunctionRecord,
    extract_functions,
    is_enabled,
)

# ---------------------------------------------------------------------------
# Snippets
# ---------------------------------------------------------------------------

SIMPLE = """\
def add(a: int, b: int) -> int:
    return a + b


def greet(name: str) -> str:
    \"\"\"Say hello.\"\"\"
    return f"Hello {name}"
"""

COMPLEX = """\
def compute(x: int, y: int) -> int:
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                y += i
    elif y < 0:
        y = abs(y)
    return y
"""

ASYNC_FN = """\
async def fetch(url: str) -> bytes:
    return b""
"""

CLASS_FN = """\
class Foo:
    def __init__(self, x: int) -> None:
        self.x = x

    def bar(self) -> int:
        return self.x
"""

CALL_GRAPH_SOURCE = """\
def helper(x):
    return x + 1

def main(n):
    result = helper(n)
    return result
"""


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


class TestIsEnabled:
    def test_constant(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled(self):
        assert is_enabled() is True


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class TestExtract:
    def test_returns_list(self):
        records = extract_functions(SIMPLE)
        assert isinstance(records, list)

    def test_finds_two_functions(self):
        records = extract_functions(SIMPLE)
        assert len(records) == 2

    def test_record_type(self):
        records = extract_functions(SIMPLE)
        assert all(isinstance(r, FunctionRecord) for r in records)

    def test_function_names(self):
        records = extract_functions(SIMPLE)
        names = {r.name for r in records}
        assert names == {"add", "greet"}

    def test_empty_source(self):
        assert extract_functions("") == []

    def test_syntax_error_returns_empty(self):
        assert extract_functions("def foo(\n    pass\n") == []


# ---------------------------------------------------------------------------
# Signature details
# ---------------------------------------------------------------------------


class TestSignatureDetails:
    def test_params_extracted(self):
        records = extract_functions(SIMPLE)
        add = next(r for r in records if r.name == "add")
        assert "a" in add.signature.params
        assert "b" in add.signature.params

    def test_return_annotation(self):
        records = extract_functions(SIMPLE)
        add = next(r for r in records if r.name == "add")
        assert add.signature.return_annotation == "int"

    def test_async_flagged(self):
        records = extract_functions(ASYNC_FN)
        assert records[0].signature.is_async is True

    def test_method_flagged(self):
        records = extract_functions(CLASS_FN)
        init = next(r for r in records if r.name == "__init__")
        assert init.signature.is_method is True


# ---------------------------------------------------------------------------
# Docstring
# ---------------------------------------------------------------------------


class TestDocstring:
    def test_docstring_extracted(self):
        records = extract_functions(SIMPLE)
        greet = next(r for r in records if r.name == "greet")
        assert greet.docstring == "Say hello."

    def test_no_docstring_none(self):
        records = extract_functions(SIMPLE)
        add = next(r for r in records if r.name == "add")
        assert add.docstring is None


# ---------------------------------------------------------------------------
# Complexity
# ---------------------------------------------------------------------------


class TestComplexity:
    def test_simple_function_low_complexity(self):
        records = extract_functions(SIMPLE)
        add = next(r for r in records if r.name == "add")
        assert add.complexity == 1

    def test_complex_function_higher_complexity(self):
        records = extract_functions(COMPLEX)
        assert records[0].complexity > 1


# ---------------------------------------------------------------------------
# Call graph
# ---------------------------------------------------------------------------


class TestCallGraph:
    def test_build_call_graph_returns_report(self):
        ex = FunctionExtractor()
        report = ex.build_call_graph(CALL_GRAPH_SOURCE)
        assert isinstance(report, CallGraphReport)

    def test_edges_captured(self):
        ex = FunctionExtractor()
        report = ex.build_call_graph(CALL_GRAPH_SOURCE)
        assert "helper" in report.edges.get("main", set())

    def test_callers_of(self):
        ex = FunctionExtractor()
        report = ex.build_call_graph(CALL_GRAPH_SOURCE)
        callers = report.callers_of("helper")
        assert "main" in callers


# ---------------------------------------------------------------------------
# Training samples
# ---------------------------------------------------------------------------


class TestTrainingSamples:
    def test_returns_samples_list(self):
        ex = FunctionExtractor()
        samples = ex.to_training_samples(SIMPLE)
        assert isinstance(samples, list)
        assert len(samples) == 2

    def test_sample_has_keys(self):
        ex = FunctionExtractor()
        samples = ex.to_training_samples(SIMPLE)
        for s in samples:
            assert "prompt" in s
            assert "completion" in s
            assert "metadata" in s

    def test_metadata_has_complexity(self):
        ex = FunctionExtractor()
        samples = ex.to_training_samples(SIMPLE)
        assert all("complexity" in s["metadata"] for s in samples)
