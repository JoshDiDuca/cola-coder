"""Tests for SemanticSearchIndex (features/semantic_search.py)."""

from __future__ import annotations


from cola_coder.features.semantic_search import (
    FEATURE_ENABLED,
    CodeDocument,
    SearchResult,
    SemanticSearchIndex,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Sample code snippets
# ---------------------------------------------------------------------------

MATH_CODE = """\
def add_numbers(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b


def multiply_numbers(a: int, b: int) -> int:
    \"\"\"Multiply two numbers.\"\"\"
    return a * b
"""

FILE_CODE = """\
def read_file(path: str) -> str:
    \"\"\"Read a file and return its contents.\"\"\"
    with open(path) as fh:
        return fh.read()


def write_file(path: str, content: str) -> None:
    \"\"\"Write content to a file.\"\"\"
    with open(path, 'w') as fh:
        fh.write(content)
"""

CLASS_CODE = """\
class Calculator:
    def __init__(self):
        self.history = []

    def add(self, x, y):
        result = x + y
        self.history.append(result)
        return result
"""


class TestIsEnabled:
    def test_feature_enabled(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled_returns_true(self):
        assert is_enabled() is True


class TestAddAndSize:
    def test_add_document(self):
        index = SemanticSearchIndex()
        doc = CodeDocument(doc_id="test.func", source="def add(a, b): return a + b")
        index.add(doc)
        assert index.size() == 1

    def test_add_source_parses_functions(self):
        index = SemanticSearchIndex()
        docs = index.add_source(MATH_CODE, doc_id="math")
        assert len(docs) == 2
        assert index.size() == 2

    def test_add_source_parses_classes(self):
        index = SemanticSearchIndex()
        docs = index.add_source(CLASS_CODE, doc_id="calc")
        assert len(docs) == 1
        assert docs[0].kind == "class"

    def test_add_source_syntax_error_returns_empty(self):
        index = SemanticSearchIndex()
        docs = index.add_source("def broken(:\n    pass")
        assert docs == []
        assert index.size() == 0


class TestSearch:
    def test_search_returns_results(self):
        index = SemanticSearchIndex()
        index.add_source(MATH_CODE, "math")
        results = index.search("add numbers", top_k=5)
        assert len(results) > 0
        assert isinstance(results[0], SearchResult)

    def test_relevant_result_scores_higher(self):
        index = SemanticSearchIndex()
        index.add_source(MATH_CODE, "math")
        index.add_source(FILE_CODE, "files")
        # "add numbers" should rank math functions above file functions
        results = index.search("add numbers")
        assert len(results) > 0
        assert "add" in results[0].doc_id.lower() or "math" in results[0].doc_id.lower()

    def test_file_query_finds_file_functions(self):
        index = SemanticSearchIndex()
        index.add_source(MATH_CODE, "math")
        index.add_source(FILE_CODE, "files")
        results = index.search("read file path")
        assert len(results) > 0
        assert "read" in results[0].doc_id.lower() or "file" in results[0].doc_id.lower()

    def test_empty_index_returns_empty(self):
        index = SemanticSearchIndex()
        results = index.search("anything")
        assert results == []

    def test_top_k_limits_results(self):
        index = SemanticSearchIndex()
        index.add_source(MATH_CODE, "math")
        index.add_source(FILE_CODE, "files")
        results = index.search("return", top_k=1)
        assert len(results) <= 1

    def test_scores_are_positive(self):
        index = SemanticSearchIndex()
        index.add_source(MATH_CODE, "math")
        results = index.search("multiply")
        for r in results:
            assert r.score > 0.0

    def test_snippet_length_capped(self):
        index = SemanticSearchIndex()
        index.add_source(FILE_CODE, "files")
        results = index.search("file")
        for r in results:
            assert len(r.snippet) <= 120


class TestCodeDocument:
    def test_auto_tokenises_source(self):
        doc = CodeDocument(doc_id="x", source="def compute_total(items): pass")
        # tokeniser preserves snake_case identifiers as single tokens
        assert "compute_total" in doc.tokens
        assert "items" in doc.tokens

    def test_manual_tokens_not_overwritten(self):
        doc = CodeDocument(doc_id="x", source="def foo(): pass", tokens=["custom", "tokens"])
        assert doc.tokens == ["custom", "tokens"]
