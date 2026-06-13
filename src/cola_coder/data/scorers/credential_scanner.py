"""Credential scanner -- detect and handle secrets in code before LLM submission.

Scans for API keys, passwords, connection strings, tokens, and other secrets
that should not be sent to external LLM APIs (Claude, Ollama).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class CredentialFinding:
    """A single credential detection."""
    pattern_name: str       # e.g. "AWS Access Key"
    line_number: int = 0
    masked_match: str = ""  # e.g. "AKIA****XXXX"


@dataclass
class ScanResult:
    """Result of scanning code for credentials."""
    has_credentials: bool = False
    findings: list[CredentialFinding] = field(default_factory=list)


class CredentialScanner:
    """Scan code for credentials before sending to external APIs.

    Modes:
        off:    No scanning, pass through unchanged.
        warn:   Detect and return findings, but pass code through.
        strip:  Replace detected secrets with [REDACTED] (default).
        reject: Return None if any credential is found.
    """

    # Compiled regex patterns: (pattern, name)
    PATTERNS: list[tuple[str, str]] = [
        # Cloud provider keys
        (r"AKIA[0-9A-Z]{16}", "AWS Access Key"),
        (r"(?:aws_secret_access_key|AWS_SECRET_ACCESS_KEY)\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{40}['\"]?", "AWS Secret Key"),

        # API tokens
        (r"ghp_[A-Za-z0-9_]{36}", "GitHub Personal Access Token"),
        (r"gho_[A-Za-z0-9_]{36}", "GitHub OAuth Token"),
        (r"github_pat_[A-Za-z0-9_]{22}_[A-Za-z0-9]{59}", "GitHub Fine-Grained PAT"),
        (r"sk-[A-Za-z0-9]{48}", "OpenAI API Key"),
        (r"sk-ant-[A-Za-z0-9\-]{40,}", "Anthropic API Key"),
        (r"xoxb-[0-9]+-[A-Za-z0-9]+", "Slack Bot Token"),
        (r"xoxp-[0-9]+-[A-Za-z0-9]+", "Slack User Token"),

        # OAuth / JWT
        (r"ya29\.[A-Za-z0-9_-]+", "Google OAuth Token"),
        (r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}", "JWT Token"),
        (r"Bearer\s+[A-Za-z0-9\-._~+/]{20,}=*", "Bearer Token"),

        # Database connection strings
        (r"mongodb(?:\+srv)?://[^\s\"'`]+", "MongoDB Connection String"),
        (r"postgres(?:ql)?://[^\s\"'`]+", "PostgreSQL Connection String"),
        (r"mysql://[^\s\"'`]+", "MySQL Connection String"),
        (r"redis://[^\s\"'`]+", "Redis Connection String"),
        (r"mssql://[^\s\"'`]+", "MSSQL Connection String"),

        # Cryptographic material. Match the WHOLE PEM block (BEGIN..END), not
        # just the header line, so strip mode redacts the secret key body too —
        # otherwise process("strip") left every byte of the private key in the
        # output and only blanked the header. `[\s\S]*?` spans newlines without
        # needing a re.DOTALL flag; the END marker is optional so a truncated key
        # (header only, no END) still matches and is redacted.
        (r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
         r"(?:[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----)?", "Private Key"),
        (r"-----BEGIN CERTIFICATE-----(?:[\s\S]*?-----END CERTIFICATE-----)?", "Certificate"),

        # Generic secrets (high-confidence patterns only)
        (r"""['"](?:password|passwd|pwd|secret|api_key|apikey|api_secret|access_token|auth_token)\s*['"]?\s*[:=]\s*['"][^'"]{8,}['"]""", "Hardcoded Secret"),

        # Payment
        (r"sk_live_[A-Za-z0-9]{24,}", "Stripe Secret Key"),
        (r"rk_live_[A-Za-z0-9]{24,}", "Stripe Restricted Key"),
    ]

    def __init__(
        self,
        mode: str = "strip",
        extra_patterns: list[tuple[str, str]] | None = None,
    ) -> None:
        if mode not in ("off", "warn", "strip", "reject"):
            raise ValueError(f"Invalid credential scan mode: {mode!r}. Use: off, warn, strip, reject")
        self.mode = mode
        all_patterns = self.PATTERNS + (extra_patterns or [])
        self._compiled = [(re.compile(p), name) for p, name in all_patterns]

    def scan(self, code: str) -> ScanResult:
        """Scan code for credentials.

        Returns:
            ScanResult with findings list.
        """
        if self.mode == "off":
            return ScanResult(has_credentials=False)

        findings: list[CredentialFinding] = []

        # Scan over the WHOLE text (not line-by-line) so detection coverage
        # exactly matches the redaction in process()'s "strip" mode, which also
        # runs pattern.sub over the whole text. A line-by-line scan would miss a
        # future multiline pattern (e.g. a PEM PRIVATE KEY block) that strip
        # would still redact — keeping both on the full text means detection and
        # redaction can never diverge. Line numbers are derived from the match
        # offset.
        for pattern, name in self._compiled:
            for match in pattern.finditer(code):
                matched_text = match.group(0)
                line_num = code.count("\n", 0, match.start()) + 1
                # Mask the match for safe logging
                if len(matched_text) > 8:
                    masked = matched_text[:4] + "****" + matched_text[-4:]
                else:
                    masked = "****"
                findings.append(CredentialFinding(
                    pattern_name=name,
                    line_number=line_num,
                    masked_match=masked,
                ))

        return ScanResult(
            has_credentials=len(findings) > 0,
            findings=findings,
        )

    def process(self, code: str) -> str | None:
        """Apply the configured mode to code.

        Returns:
            Processed code string, or None if mode is "reject" and credentials found.
        """
        if self.mode == "off":
            return code

        result = self.scan(code)

        if not result.has_credentials:
            return code

        if self.mode == "warn":
            # Findings logged externally, pass code through unchanged
            return code

        if self.mode == "reject":
            return None

        if self.mode == "strip":
            # Replace all matches with [REDACTED]
            processed = code
            for pattern, _name in self._compiled:
                processed = pattern.sub("[REDACTED]", processed)
            return processed

        return code  # Fallback: pass through
