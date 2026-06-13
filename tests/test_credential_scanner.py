"""Tests for CredentialScanner."""

from __future__ import annotations

import pytest

from cola_coder.data.scorers.credential_scanner import (
    CredentialScanner,
)


class TestScanDetection:
    """Verify credential detection patterns."""

    def test_detects_aws_access_key(self) -> None:
        code = 'const key = "AKIAIOSFODNN7EXAMPLE1";'
        result = CredentialScanner(mode="warn").scan(code)
        assert result.has_credentials
        assert any("AWS" in f.pattern_name for f in result.findings)

    def test_detects_github_token(self) -> None:
        code = 'const token = "ghp_ABCDEFghijklmnop1234567890abcdefghij";'
        result = CredentialScanner(mode="warn").scan(code)
        assert result.has_credentials
        assert any("GitHub" in f.pattern_name for f in result.findings)

    def test_detects_openai_key(self) -> None:
        code = 'export const API_KEY = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMN";'
        result = CredentialScanner(mode="warn").scan(code)
        assert result.has_credentials
        assert any("OpenAI" in f.pattern_name for f in result.findings)

    def test_detects_anthropic_key(self) -> None:
        code = 'const key = "sk-ant-abcdefghijklmnopqrstuvwxyz01234567890123456";'
        result = CredentialScanner(mode="warn").scan(code)
        assert result.has_credentials
        assert any("Anthropic" in f.pattern_name for f in result.findings)

    def test_detects_mongodb_connection(self) -> None:
        code = 'const uri = "mongodb+srv://user:pass@cluster.mongodb.net/db";'
        result = CredentialScanner(mode="warn").scan(code)
        assert result.has_credentials
        assert any("MongoDB" in f.pattern_name for f in result.findings)

    def test_detects_postgres_connection(self) -> None:
        code = 'DATABASE_URL = "postgresql://user:password@localhost:5432/mydb"'
        result = CredentialScanner(mode="warn").scan(code)
        assert result.has_credentials
        assert any("PostgreSQL" in f.pattern_name for f in result.findings)

    def test_detects_private_key(self) -> None:
        code = '-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIB...'
        result = CredentialScanner(mode="warn").scan(code)
        assert result.has_credentials
        assert any("Private Key" in f.pattern_name for f in result.findings)

    def test_detects_jwt_token(self) -> None:
        code = 'const token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U";'
        result = CredentialScanner(mode="warn").scan(code)
        assert result.has_credentials
        assert any("JWT" in f.pattern_name for f in result.findings)

    def test_detects_stripe_key(self) -> None:
        # Constructed dynamically to avoid triggering GitHub push protection
        prefix = "sk_" + "live" + "_"
        code = f'const stripe = "{prefix}abcdefghijklmnopqrstuvwx";'
        result = CredentialScanner(mode="warn").scan(code)
        assert result.has_credentials
        assert any("Stripe" in f.pattern_name for f in result.findings)

    def test_detects_hardcoded_password(self) -> None:
        code = """config = {"password": "supersecretpassword123"}"""
        result = CredentialScanner(mode="warn").scan(code)
        assert result.has_credentials

    def test_detects_slack_token(self) -> None:
        # Constructed dynamically to avoid triggering GitHub push protection
        prefix = "xox" + "b-"
        code = f'const token = "{prefix}123456789-abcdefghijklmnop";'
        result = CredentialScanner(mode="warn").scan(code)
        assert result.has_credentials


class TestNormalCodePasses:
    """Verify no false positives on normal code."""

    def test_normal_typescript(self) -> None:
        code = '''
function add(a: number, b: number): number {
    return a + b;
}

interface User {
    name: string;
    email: string;
}

const users: User[] = [];
'''
        result = CredentialScanner(mode="reject").scan(code)
        assert not result.has_credentials

    def test_variable_named_key(self) -> None:
        code = 'const key = "hello";'
        result = CredentialScanner(mode="reject").scan(code)
        assert not result.has_credentials

    def test_short_token_like_string(self) -> None:
        code = 'const sk = "abc123";'
        result = CredentialScanner(mode="reject").scan(code)
        assert not result.has_credentials

    def test_import_statement(self) -> None:
        code = 'import { createClient } from "@supabase/supabase-js";'
        result = CredentialScanner(mode="reject").scan(code)
        assert not result.has_credentials


class TestModes:
    """Verify mode behavior."""

    def test_mode_off_passes_everything(self) -> None:
        code = 'const key = "AKIAIOSFODNN7EXAMPLE1";'
        processed = CredentialScanner(mode="off").process(code)
        assert processed == code

    def test_mode_off_scan_returns_no_findings(self) -> None:
        code = 'const key = "AKIAIOSFODNN7EXAMPLE1";'
        result = CredentialScanner(mode="off").scan(code)
        assert not result.has_credentials

    def test_mode_warn_passes_through(self) -> None:
        code = 'const key = "AKIAIOSFODNN7EXAMPLE1";'
        processed = CredentialScanner(mode="warn").process(code)
        assert processed == code
        assert "AKIA" in processed

    def test_mode_strip_redacts(self) -> None:
        code = 'const key = "AKIAIOSFODNN7EXAMPLE1";'
        processed = CredentialScanner(mode="strip").process(code)
        assert processed is not None
        assert "AKIA" not in processed
        assert "[REDACTED]" in processed

    def test_mode_reject_returns_none(self) -> None:
        code = 'const key = "AKIAIOSFODNN7EXAMPLE1";'
        processed = CredentialScanner(mode="reject").process(code)
        assert processed is None

    def test_clean_code_passes_all_modes(self) -> None:
        code = "const x: number = 42;"
        for mode in ("off", "warn", "strip", "reject"):
            processed = CredentialScanner(mode=mode).process(code)
            assert processed == code

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid credential scan mode"):
            CredentialScanner(mode="invalid")


class TestScanResult:
    """Verify ScanResult structure."""

    def test_findings_have_line_numbers(self) -> None:
        code = "line1\nconst key = \"AKIAIOSFODNN7EXAMPLE1\";\nline3"
        result = CredentialScanner(mode="warn").scan(code)
        assert result.findings[0].line_number == 2

    def test_findings_have_masked_match(self) -> None:
        code = 'const key = "AKIAIOSFODNN7EXAMPLE1";'
        result = CredentialScanner(mode="warn").scan(code)
        masked = result.findings[0].masked_match
        assert "****" in masked
        assert len(masked) < len("AKIAIOSFODNN7EXAMPLE1")

    def test_multiple_findings_in_one_file(self) -> None:
        code = '''
const aws = "AKIAIOSFODNN7EXAMPLE1";
const gh = "ghp_ABCDEFghijklmnop1234567890abcdefghij";
'''
        result = CredentialScanner(mode="warn").scan(code)
        assert len(result.findings) >= 2


class TestCustomPatterns:
    """Verify extra_patterns support."""

    def test_extra_pattern_detection(self) -> None:
        scanner = CredentialScanner(
            mode="warn",
            extra_patterns=[(r"CUSTOM_[A-Z]{10}", "Custom Token")],
        )
        result = scanner.scan('const t = "CUSTOM_ABCDEFGHIJ";')
        assert result.has_credentials
        assert any("Custom" in f.pattern_name for f in result.findings)


class TestGithubFineGrainedPat:
    """DATA-048(b): the fine-grained PAT pattern is the strict, correct shape
    ("github_pat_" + 22-char prefix + "_" + 59-char body) and is harmonized with
    data/quality_filter.py."""

    def test_detects_real_shape(self) -> None:
        pat = "github_pat_" + "A" * 22 + "_" + "B" * 59
        result = CredentialScanner(mode="warn").scan(f'const t = "{pat}";')
        assert result.has_credentials
        assert any("Fine-Grained" in f.pattern_name for f in result.findings)


class TestScanRedactionParity:
    """DATA-048(a): scan() and strip-mode process() both run over the WHOLE
    text now, so detection coverage and redaction coverage cannot diverge."""

    def test_scan_detects_what_strip_redacts(self) -> None:
        code = (
            'line1\n'
            'const aws = "AKIAIOSFODNN7EXAMPLE1";\n'
            'const gh = "ghp_ABCDEFghijklmnop1234567890abcdefghij";\n'
        )
        scanner = CredentialScanner(mode="strip")
        result = scanner.scan(code)
        processed = scanner.process(code)
        # Every credential scan flagged is also redacted by strip.
        assert result.has_credentials
        assert "AKIA" not in processed
        assert "ghp_" not in processed
        assert processed.count("[REDACTED]") == len(result.findings)

    def test_line_numbers_preserved_after_whole_text_scan(self) -> None:
        code = 'line1\nconst key = "AKIAIOSFODNN7EXAMPLE1";\nline3'
        result = CredentialScanner(mode="warn").scan(code)
        assert result.findings[0].line_number == 2
