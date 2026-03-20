"""Tests for the data filter plugin system.

Each test creates simple objects with .content and .metadata attributes
(duck typing) to test the filters without depending on a DataRecord class.
"""

import pytest


class FakeRecord:
    """Simple duck-typed record for testing filters."""

    def __init__(self, content: str = "", metadata: dict | None = None):
        self.content = content
        self.metadata = metadata or {}


# ---------------------------------------------------------------------------
# Helpers for conditional test skipping (must be before decorators)
# ---------------------------------------------------------------------------

def _try_import_datasketch() -> bool:
    try:
        import datasketch  # noqa: F401
        return True
    except ImportError:
        return False


def _try_import_treesitter() -> bool:
    try:
        import tree_sitter  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# LicenseFilter tests
# ---------------------------------------------------------------------------

class TestLicenseFilter:
    def setup_method(self):
        from cola_coder.data.filters.license_filter import LicenseFilter
        self.f = LicenseFilter()

    def test_allows_mit(self):
        record = FakeRecord(content="print('hello')", metadata={"license": "MIT"})
        keep, reason = self.f.check(record)
        assert keep is True

    def test_allows_apache(self):
        record = FakeRecord(content="x = 1", metadata={"license": "Apache-2.0"})
        keep, reason = self.f.check(record)
        assert keep is True

    def test_allows_bsd3(self):
        record = FakeRecord(content="x = 1", metadata={"license": "BSD-3-Clause"})
        keep, _ = self.f.check(record)
        assert keep is True

    def test_rejects_gpl(self):
        record = FakeRecord(content="x = 1", metadata={"license": "GPL-3.0"})
        keep, reason = self.f.check(record)
        assert keep is False
        assert "non_permissive" in reason

    def test_rejects_unknown(self):
        record = FakeRecord(content="x = 1", metadata={"license": "SomethingWeird"})
        keep, reason = self.f.check(record)
        assert keep is False

    def test_rejects_no_license(self):
        record = FakeRecord(content="x = 1", metadata={})
        keep, reason = self.f.check(record)
        assert keep is False
        assert "no_license" in reason

    def test_allow_unknown_mode(self):
        from cola_coder.data.filters.license_filter import LicenseFilter
        f = LicenseFilter(allow_unknown=True)
        record = FakeRecord(content="x = 1", metadata={})
        keep, _ = f.check(record)
        assert keep is True

    def test_normalizes_mit_lowercase(self):
        record = FakeRecord(content="x = 1", metadata={"license": "mit"})
        keep, _ = self.f.check(record)
        assert keep is True

    def test_name(self):
        assert self.f.name() == "license"


# ---------------------------------------------------------------------------
# PIIFilter tests
# ---------------------------------------------------------------------------

class TestPIIFilter:
    def setup_method(self):
        from cola_coder.data.filters.pii import PIIFilter
        self.f = PIIFilter()

    def test_catches_email(self):
        code = 'user_email = "john.doe@company.com"'
        record = FakeRecord(content=code)
        keep, reason = self.f.check(record)
        assert keep is False
        assert "email" in reason

    def test_catches_api_key(self):
        code = 'api_key = "sk1234567890abcdefghijklmnop"'
        record = FakeRecord(content=code)
        keep, reason = self.f.check(record)
        assert keep is False
        assert "api_key" in reason

    def test_catches_aws_key(self):
        code = 'AWS_KEY = "AKIAIOSFODNN7YRQMPXA"'
        record = FakeRecord(content=code)
        keep, reason = self.f.check(record)
        assert keep is False
        assert "aws" in reason

    def test_catches_private_key(self):
        code = '-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQ...'
        record = FakeRecord(content=code)
        keep, reason = self.f.check(record)
        assert keep is False
        assert "private_key" in reason

    def test_catches_password(self):
        code = 'password = "supersecretpassword123"'
        record = FakeRecord(content=code)
        keep, reason = self.f.check(record)
        assert keep is False
        assert "password" in reason

    def test_catches_github_token(self):
        code = 'token = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"'
        record = FakeRecord(content=code)
        keep, reason = self.f.check(record)
        assert keep is False

    def test_passes_clean_code(self):
        code = '''
def hello():
    print("Hello, world!")
    return 42

class MyApp:
    def run(self):
        self.start()
'''
        record = FakeRecord(content=code)
        keep, _ = self.f.check(record)
        assert keep is True

    def test_false_positive_example_values(self):
        code = 'api_key = "your_example_api_key_here"'
        record = FakeRecord(content=code)
        keep, _ = self.f.check(record)
        assert keep is True

    def test_name(self):
        assert self.f.name() == "pii"


# ---------------------------------------------------------------------------
# ContentFilter tests
# ---------------------------------------------------------------------------

class TestContentFilter:
    def setup_method(self):
        from cola_coder.data.filters.content import ContentFilter
        self.f = ContentFilter()

    def test_catches_minified_code(self):
        # Simulate minified JS: very long single line
        code = "var a=" + "b+c;" * 200
        record = FakeRecord(content=code)
        keep, reason = self.f.check(record)
        assert keep is False
        assert "minified" in reason

    def test_catches_autogenerated(self):
        code = (
            "// Code generated by protoc. DO NOT EDIT.\npackage main\n"
            + "x = 1\n" * 20
        )
        record = FakeRecord(content=code)
        keep, reason = self.f.check(record)
        assert keep is False
        assert "autogenerated" in reason

    def test_catches_json_data_dump(self):
        # Lines that start with braces/brackets to trigger JSON detection
        lines = ['{\n'] + ['{"key": "value"},\n'] * 50 + ['}\n']
        code = "".join(lines)
        record = FakeRecord(content=code)
        keep, reason = self.f.check(record)
        assert keep is False
        assert "data_dump" in reason

    def test_catches_lock_file(self):
        code = '# this file is autogenerated\nsome_dep==1.0.0\nother_dep==2.0.0\n'
        record = FakeRecord(content=code)
        keep, reason = self.f.check(record)
        assert keep is False
        # Could match either autogenerated or lock_file check
        assert "lock_file" in reason or "autogenerated" in reason

    def test_passes_normal_code(self):
        code = '''
import os
import sys

def main():
    """Entry point for the application."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        data = f.read()

    process(data)

def process(data):
    lines = data.split("\\n")
    for line in lines:
        if line.strip():
            print(line)

if __name__ == "__main__":
    main()
'''
        record = FakeRecord(content=code)
        keep, _ = self.f.check(record)
        assert keep is True

    def test_name(self):
        assert self.f.name() == "content"


# ---------------------------------------------------------------------------
# DeduplicationFilter tests
# ---------------------------------------------------------------------------

class TestDeduplicationFilter:
    def setup_method(self):
        from cola_coder.data.filters.dedup import DeduplicationFilter
        self.f = DeduplicationFilter(threshold=0.8, num_perm=128)

    @pytest.mark.skipif(
        not _try_import_datasketch(),
        reason="datasketch not installed",
    )
    def test_detects_near_duplicates(self):
        original = "def hello():\n    print('hello world')\n    return 42\n" * 10
        record1 = FakeRecord(content=original, metadata={})
        keep1, _ = self.f.check(record1)
        assert keep1 is True  # First file always passes

        # Near-duplicate: same content with minor change
        duplicate = original.replace("42", "43")
        record2 = FakeRecord(content=duplicate, metadata={})
        keep2, reason = self.f.check(record2)
        assert keep2 is False
        assert "near_duplicate" in reason

    @pytest.mark.skipif(
        not _try_import_datasketch(),
        reason="datasketch not installed",
    )
    def test_allows_different_files(self):
        code1 = "def foo():\n    return 'bar'\n" * 10
        code2 = "class Widget:\n    def render(self):\n        pass\n" * 10
        record1 = FakeRecord(content=code1)
        record2 = FakeRecord(content=code2)
        keep1, _ = self.f.check(record1)
        keep2, _ = self.f.check(record2)
        assert keep1 is True
        assert keep2 is True

    def test_graceful_without_datasketch(self):
        """If datasketch is not installed, filter passes everything."""
        from cola_coder.data.filters.dedup import _HAS_DATASKETCH
        if _HAS_DATASKETCH:
            pytest.skip("datasketch is installed, can't test fallback")
        record = FakeRecord(content="anything")
        keep, _ = self.f.check(record)
        assert keep is True

    def test_name(self):
        assert self.f.name() == "deduplication"

    @pytest.mark.skipif(
        not _try_import_datasketch(),
        reason="datasketch not installed",
    )
    def test_reset(self):
        code = "def hello():\n    print('hello world')\n    return 42\n" * 10
        record = FakeRecord(content=code)
        self.f.check(record)
        self.f.reset()
        # After reset, same content should pass again
        keep, _ = self.f.check(record)
        assert keep is True


# ---------------------------------------------------------------------------
# SyntaxFilter tests
# ---------------------------------------------------------------------------

class TestSyntaxFilter:
    def test_graceful_without_treesitter(self):
        """If tree-sitter is not installed, filter passes everything."""
        from cola_coder.data.filters.syntax import _HAS_TREESITTER
        if _HAS_TREESITTER:
            pytest.skip("tree-sitter is installed, can't test fallback")
        from cola_coder.data.filters.syntax import SyntaxFilter
        f = SyntaxFilter()
        record = FakeRecord(content="broken code {{{{", metadata={"language": "python"})
        keep, _ = f.check(record)
        assert keep is True

    @pytest.mark.skipif(
        not _try_import_treesitter(),
        reason="tree-sitter not installed",
    )
    def test_passes_valid_python(self):
        from cola_coder.data.filters.syntax import SyntaxFilter
        f = SyntaxFilter(languages=["python"])
        code = "def hello():\n    print('hello')\n    return 42\n"
        record = FakeRecord(content=code, metadata={"language": "python"})
        keep, _ = f.check(record)
        assert keep is True

    @pytest.mark.skipif(
        not _try_import_treesitter(),
        reason="tree-sitter not installed",
    )
    def test_rejects_broken_python(self):
        from cola_coder.data.filters.syntax import SyntaxFilter
        f = SyntaxFilter(languages=["python"], max_error_ratio=0.0)
        code = "def hello(\n    print('hello'\n    return\n"
        record = FakeRecord(content=code, metadata={"language": "python"})
        keep, reason = f.check(record)
        assert keep is False
        assert "syntax_errors" in reason

    def test_name(self):
        from cola_coder.data.filters.syntax import SyntaxFilter
        f = SyntaxFilter()
        assert f.name() == "syntax"


# ---------------------------------------------------------------------------
# Integration: __init__ imports
# ---------------------------------------------------------------------------

class TestFilterImports:
    def test_all_filters_importable(self):
        from cola_coder.data.filters import (
            ContentFilter,
            DeduplicationFilter,
            LicenseFilter,
            PIIFilter,
            SyntaxFilter,
        )
        # All should be classes
        assert callable(ContentFilter)
        assert callable(DeduplicationFilter)
        assert callable(LicenseFilter)
        assert callable(PIIFilter)
        assert callable(SyntaxFilter)

    def test_filter_interface(self):
        """All filters should have name(), check(), and optionally setup()."""
        from cola_coder.data.filters import (
            ContentFilter,
            DeduplicationFilter,
            LicenseFilter,
            PIIFilter,
            SyntaxFilter,
        )
        filters = [
            ContentFilter(),
            DeduplicationFilter(),
            LicenseFilter(),
            PIIFilter(),
            SyntaxFilter(),
        ]
        for f in filters:
            assert hasattr(f, "name")
            assert hasattr(f, "check")
            assert hasattr(f, "setup")
            assert isinstance(f.name(), str)
