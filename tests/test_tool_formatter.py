"""Parsing tests for the agent tool-call formatter.

Model output is UNTRUSTED. ``parse_tool_call`` must never raise on whatever the
model emits inside a ``<tool_call>`` block — it should silently skip anything
that isn't a well-formed tool-call object and surface only valid calls.

Regression guard (TOOL-018): a ``<tool_call>`` block holding a bare JSON scalar
(``123``, ``true``, ``null``, ``12.5``) parses fine with ``json.loads`` but is
not iterable, so the old ``"name" in parsed`` membership test raised an UNCAUGHT
``TypeError`` (only ``json.JSONDecodeError`` was caught) and crashed the whole
``AgentLoop.run`` loop instead of ignoring the junk block.
"""

import pytest

from cola_coder.tools.formatter import (
    format_tool_call,
    has_tool_call,
    parse_tool_call,
    strip_tool_calls,
)


class TestParseValidCalls:
    def test_single_call(self):
        text = '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>'
        calls = parse_tool_call(text)
        assert calls == [{"name": "read_file", "arguments": {"path": "a.py"}}]

    def test_missing_arguments_defaults_to_empty_dict(self):
        calls = parse_tool_call('<tool_call>\n{"name": "git_log"}\n</tool_call>')
        assert calls == [{"name": "git_log", "arguments": {}}]

    def test_multiple_calls_in_one_text(self):
        text = (
            '<tool_call>\n{"name": "lint", "arguments": {}}\n</tool_call>\n'
            'some prose\n'
            '<tool_call>\n{"name": "run_tests", "arguments": {}}\n</tool_call>'
        )
        names = [c["name"] for c in parse_tool_call(text)]
        assert names == ["lint", "run_tests"]

    def test_roundtrip_with_formatter(self):
        rendered = format_tool_call("search_code", {"query": "TODO"})
        calls = parse_tool_call(rendered)
        assert calls == [{"name": "search_code", "arguments": {"query": "TODO"}}]


class TestUntrustedJsonNeverRaises:
    """The headline regression: non-object JSON inside a tool_call block must be
    skipped, never raise."""

    @pytest.mark.parametrize("payload", ["123", "true", "false", "null", "12.5", "-7"])
    def test_bare_scalar_is_skipped_not_crashed(self, payload):
        text = f"<tool_call>\n{payload}\n</tool_call>"
        # Must not raise (the bug raised TypeError on non-iterable scalars).
        assert parse_tool_call(text) == []

    def test_bare_string_is_skipped(self):
        assert parse_tool_call('<tool_call>\n"hello"\n</tool_call>') == []

    def test_json_array_is_skipped(self):
        assert parse_tool_call("<tool_call>\n[1, 2, 3]\n</tool_call>") == []

    def test_object_without_name_is_skipped(self):
        assert parse_tool_call('<tool_call>\n{"arguments": {"x": 1}}\n</tool_call>') == []

    def test_malformed_json_is_skipped(self):
        assert parse_tool_call("<tool_call>\n{not valid json}\n</tool_call>") == []

    def test_scalar_block_does_not_drop_following_valid_call(self):
        # A junk scalar block before a real call must not abort parsing.
        text = (
            "<tool_call>\n42\n</tool_call>\n"
            '<tool_call>\n{"name": "lint", "arguments": {}}\n</tool_call>'
        )
        assert parse_tool_call(text) == [{"name": "lint", "arguments": {}}]


class TestHelpers:
    def test_has_tool_call(self):
        assert has_tool_call("<tool_call>\n{}\n</tool_call>") is True
        assert has_tool_call("no tools here") is False

    def test_strip_tool_calls(self):
        text = 'before <tool_call>\n{"name": "x"}\n</tool_call> after'
        stripped = strip_tool_calls(text)
        assert "<tool_call>" not in stripped
        assert "before" in stripped and "after" in stripped
