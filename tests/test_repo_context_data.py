"""Tests for prepare_repo_context_data.py context-token resolution.

DATA-003: the script used to fall back to eos_id for missing <|repo|>/<|file|>
tokens, silently producing poison training data (the whole repo/file structure
collapsed into eos). The strict resolver must instead report what's missing so
main() can fail loudly.
"""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).parent.parent / "scripts" / "prepare_repo_context_data.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prep_repo_ctx", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeInner:
    """Stands in for tokenizer.tokenizer (the HF Tokenizer)."""

    def __init__(self, table: dict[str, int]):
        self._table = table

    def token_to_id(self, name: str):
        return self._table.get(name)  # None when absent — like the real API


class _FakeTokenizer:
    def __init__(self, table: dict[str, int]):
        self.tokenizer = _FakeInner(table)


_ALL_PRESENT = {
    "<|repo|>": 100, "<|/repo|>": 101, "<|file|>": 102, "<|/file|>": 103,
}


class TestContextTokenResolver:
    def test_all_present_returns_ids_no_missing(self):
        mod = _load_module()
        ids, missing = mod._resolve_context_token_ids(_FakeTokenizer(_ALL_PRESENT))
        assert missing == []
        assert ids == _ALL_PRESENT

    def test_missing_tokens_reported_not_substituted(self):
        mod = _load_module()
        # Only <|repo|> present; the other three absent
        table = {"<|repo|>": 100}
        ids, missing = mod._resolve_context_token_ids(_FakeTokenizer(table))
        assert set(missing) == {"<|/repo|>", "<|file|>", "<|/file|>"}
        # Critically: present token kept, absent ones NOT mapped to any fallback
        assert ids == {"<|repo|>": 100}

    def test_all_missing(self):
        mod = _load_module()
        ids, missing = mod._resolve_context_token_ids(_FakeTokenizer({}))
        assert set(missing) == set(mod._CONTEXT_TOKEN_NAMES)
        assert ids == {}

    def test_no_silent_eos_fallback_helper_removed(self):
        # The old fallback-to-eos helper must be gone so it can't be reused.
        text = _SCRIPT.read_text(encoding="utf-8")
        assert "_get_special_token_id" not in text
        assert "_resolve_context_token_ids" in text
