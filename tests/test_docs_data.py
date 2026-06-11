"""DATA-013: prepare_docs_data.py must verify doc context tokens exist.

The script wraps each doc in "<|doc|>...<|/doc|>" then tokenizer.encode()s it.
But <|doc|>/<|/doc|> are NOT in the base tokenizer's SPECIAL_TOKENS — they're
CONTEXT_TOKENS added only by add_context_tokens(). With a normally-trained
tokenizer, encode() FRAGMENTS those markers into ordinary punctuation tokens,
silently producing degraded docs data (same class as DATA-003). The script must
detect the missing tokens and fail loudly with the remedy.
"""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).parent.parent / "scripts" / "prepare_docs_data.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prep_docs", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeInner:
    """Stands in for tokenizer.tokenizer (the HF Tokenizer)."""

    def __init__(self, table: dict[str, int]):
        self._table = table

    def token_to_id(self, name: str):
        return self._table.get(name)  # None when absent — like the real API


class _FakeTok:
    def __init__(self, table: dict[str, int]):
        self.tokenizer = _FakeInner(table)


class TestMissingDocTokens:
    def test_both_missing(self):
        mod = _load_module()
        tok = _FakeTok({"<|eos|>": 2})  # no doc tokens
        assert mod._missing_doc_tokens(tok) == ["<|doc|>", "<|/doc|>"]

    def test_present(self):
        mod = _load_module()
        tok = _FakeTok({"<|doc|>": 100, "<|/doc|>": 101})
        assert mod._missing_doc_tokens(tok) == []

    def test_partial_missing(self):
        mod = _load_module()
        tok = _FakeTok({"<|doc|>": 100})  # closing tag missing
        assert mod._missing_doc_tokens(tok) == ["<|/doc|>"]


class TestBuildDocText:
    def test_wraps_with_doc_markers_and_eos(self):
        mod = _load_module()
        text = mod._build_doc_text("react@18.2.0", "useState", "body text")
        assert text.startswith("<|doc|>react@18.2.0 - useState<|/doc|>")
        assert text.rstrip().endswith("<|eos|>")
        assert "body text" in text


class TestParseDocHeader:
    def test_extracts_framework_label(self):
        mod = _load_module()
        label, body = mod._parse_doc_header("// Framework: react@18.2.0\nrest\n")
        assert label == "react@18.2.0"
        assert body == "rest\n"

    def test_no_header_defaults(self):
        mod = _load_module()
        label, body = mod._parse_doc_header("no header here\n")
        assert label == "unknown@0.0.0"
        assert body == "no header here\n"
