"""Tests for context special tokens (tokenizer/special_tokens.py).

Covers:
- CONTEXT_TOKENS list length and contents
- is_enabled() feature flag
- wrap_doc(), wrap_repo(), wrap_file() formatting
"""

from __future__ import annotations

import cola_coder.tokenizer.special_tokens as st


class TestContextTokensList:
    def test_has_six_tokens(self):
        assert len(st.CONTEXT_TOKENS) == 6

    def test_contains_doc_open(self):
        assert "<|doc|>" in st.CONTEXT_TOKENS

    def test_contains_doc_close(self):
        assert "<|/doc|>" in st.CONTEXT_TOKENS

    def test_contains_repo_open(self):
        assert "<|repo|>" in st.CONTEXT_TOKENS

    def test_contains_repo_close(self):
        assert "<|/repo|>" in st.CONTEXT_TOKENS

    def test_contains_file_open(self):
        assert "<|file|>" in st.CONTEXT_TOKENS

    def test_contains_file_close(self):
        assert "<|/file|>" in st.CONTEXT_TOKENS

    def test_all_tokens_are_strings(self):
        for token in st.CONTEXT_TOKENS:
            assert isinstance(token, str)

    def test_open_close_paired(self):
        """Every opening token has a matching closing token."""
        opens = [t for t in st.CONTEXT_TOKENS if not t.startswith("<|/")]
        closes = [t for t in st.CONTEXT_TOKENS if t.startswith("<|/")]
        assert len(opens) == len(closes) == 3


class TestIsEnabled:
    def test_returns_true_by_default(self):
        assert st.is_enabled() is True

    def test_reflects_feature_flag(self, monkeypatch):
        monkeypatch.setattr(st, "FEATURE_ENABLED", False)
        assert st.is_enabled() is False
        monkeypatch.setattr(st, "FEATURE_ENABLED", True)
        assert st.is_enabled() is True


class TestWrapDoc:
    def test_basic_wrapping(self):
        result = st.wrap_doc("some docs")
        assert result == "<|doc|>some docs<|/doc|>"

    def test_opens_with_doc_token(self):
        result = st.wrap_doc("content")
        assert result.startswith("<|doc|>")

    def test_closes_with_doc_token(self):
        result = st.wrap_doc("content")
        assert result.endswith("<|/doc|>")

    def test_content_preserved_verbatim(self):
        content = "# useState\nReturns [state, setState].\n"
        result = st.wrap_doc(content)
        assert content in result

    def test_empty_content(self):
        result = st.wrap_doc("")
        assert result == "<|doc|><|/doc|>"


class TestWrapRepo:
    def test_basic_wrapping(self):
        result = st.wrap_repo("repo summary")
        assert result == "<|repo|>repo summary<|/repo|>"

    def test_opens_with_repo_token(self):
        assert st.wrap_repo("x").startswith("<|repo|>")

    def test_closes_with_repo_token(self):
        assert st.wrap_repo("x").endswith("<|/repo|>")

    def test_content_preserved(self):
        content = "next@14.2.0, react@18.2.0"
        assert content in st.wrap_repo(content)

    def test_empty_content(self):
        assert st.wrap_repo("") == "<|repo|><|/repo|>"


class TestWrapFile:
    def test_basic_wrapping(self):
        result = st.wrap_file("src/utils.ts", "export const x = 1;")
        assert result == "<|file|>src/utils.ts\nexport const x = 1;<|/file|>"

    def test_opens_with_file_token(self):
        assert st.wrap_file("a.ts", "").startswith("<|file|>")

    def test_closes_with_file_token(self):
        assert st.wrap_file("a.ts", "code").endswith("<|/file|>")

    def test_path_on_first_line(self):
        result = st.wrap_file("src/hooks/useUser.ts", "import React from 'react'")
        first_line = result.split("\n")[0]
        assert "src/hooks/useUser.ts" in first_line

    def test_content_on_subsequent_line(self):
        content = "export interface User { id: string; }"
        result = st.wrap_file("types.ts", content)
        lines = result.split("\n")
        # Content starts on second line (after the <|file|>path line)
        assert content in "\n".join(lines[1:])

    def test_nested_path(self):
        result = st.wrap_file("src/types/user.ts", "export type ID = string;")
        assert "src/types/user.ts" in result
