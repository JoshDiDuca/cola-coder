"""PII (Personally Identifiable Information) detection filter.

Detects and rejects files containing PII such as email addresses,
API keys, AWS credentials, private keys, and hardcoded passwords.

Training on PII is both a privacy risk (the model might memorize and
reproduce real credentials) and a quality issue (code with hardcoded
secrets is bad practice).
"""

import logging
import re

from cola_coder.data.registry import register_filter

logger = logging.getLogger(__name__)


@register_filter("pii")
class PIIFilter:
    """Detect and reject files containing PII.

    Uses regex patterns to detect common forms of PII and secrets in
    code files. Patterns are tuned for code (not prose) — they look for
    assignment patterns like `password = "..."` rather than just any
    string that looks like a password.
    """

    # Compiled patterns with descriptive names
    PATTERNS: list[tuple[str, re.Pattern]] = [
        # Email addresses (but not in comments about email validation)
        ("email_address", re.compile(
            r'["\'][\w.+-]+@[\w-]+\.[\w.]+["\']',
        )),

        # API keys in assignments (generic long alphanumeric strings)
        ("api_key", re.compile(
            r'(?:api[_-]?key|apikey|api_secret)\s*[:=]\s*["\'][a-zA-Z0-9]{20,}["\']',
            re.IGNORECASE,
        )),

        # AWS access keys (always start with AKIA)
        ("aws_access_key", re.compile(
            r'AKIA[0-9A-Z]{16}',
        )),

        # AWS secret keys (40 char base64)
        ("aws_secret_key", re.compile(
            r'(?:aws_secret_access_key|aws_secret)\s*[:=]\s*["\'][A-Za-z0-9/+=]{40}["\']',
            re.IGNORECASE,
        )),

        # Private keys (PEM format)
        ("private_key", re.compile(
            r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----',
        )),

        # OpenAI-style API keys
        ("openai_key", re.compile(
            r'sk-[a-zA-Z0-9]{32,}',
        )),

        # GitHub personal access tokens
        ("github_token", re.compile(
            r'ghp_[a-zA-Z0-9]{36}',
        )),

        # GitHub OAuth tokens
        ("github_oauth", re.compile(
            r'gho_[a-zA-Z0-9]{36}',
        )),

        # Slack tokens
        ("slack_token", re.compile(
            r'xox[bpors]-[a-zA-Z0-9-]{10,}',
        )),

        # Passwords in assignments. The bare `pass` alternative carries a
        # negative lookbehind for a letter so it matches a standalone `pass`
        # (or snake_case `db_pass`) but NOT the suffix of an unrelated
        # identifier like `bypass`/`compass`/`surpass` — which previously
        # false-flagged valid code as PII and dropped the whole file. The full
        # words (password/passwd/pwd) keep matching after `_` or word starts
        # (e.g. `my_password`), so snake_case secrets are still caught.
        ("password_assignment", re.compile(
            r'(?:password|passwd|pwd|(?<![A-Za-z])pass)\s*[:=]\s*["\'][^"\']{8,}["\']',
            re.IGNORECASE,
        )),

        # Generic secret/token assignments (long values)
        ("secret_assignment", re.compile(
            r'(?:secret|token|auth_token|access_token)\s*[:=]\s*["\'][a-zA-Z0-9]{20,}["\']',
            re.IGNORECASE,
        )),

        # Connection strings with credentials
        ("connection_string", re.compile(
            r'(?:mysql|postgres|postgresql|mongodb|redis)://\w+:[^@\s]{8,}@',
            re.IGNORECASE,
        )),
    ]

    # Patterns that indicate this is test/example code, not real PII
    FALSE_POSITIVE_INDICATORS = [
        "example",
        "test",
        "dummy",
        "fake",
        "placeholder",
        "your_",
        "xxx",
        "changeme",
        "replace_",
        "<your",
        "TODO",
    ]

    def __init__(self, max_detections: int = 1, check_false_positives: bool = True):
        """
        Args:
            max_detections: How many PII matches to tolerate before rejecting.
                Default 1 means reject on first real match.
            check_false_positives: If True, try to filter out test/example values.
        """
        self.max_detections = max_detections
        self.check_false_positives = check_false_positives

    def name(self) -> str:
        return "pii"

    def _is_false_positive(self, match_text: str) -> bool:
        """Check if a match looks like a test/example value, not real PII."""
        lower = match_text.lower()
        return any(indicator in lower for indicator in self.FALSE_POSITIVE_INDICATORS)

    def check(self, record) -> tuple[bool, str]:
        """Check if the file contains PII or secrets.

        Args:
            record: Object with .content (str) and .metadata (dict) attributes.

        Returns:
            (keep, reason) tuple.
        """
        content = record.content
        if not content:
            return True, ""

        detections: list[str] = []

        for pattern_name, pattern in self.PATTERNS:
            matches = pattern.findall(content)
            for match in matches:
                if self.check_false_positives and self._is_false_positive(match):
                    continue
                detections.append(pattern_name)
                if len(detections) >= self.max_detections:
                    types = ", ".join(sorted(set(detections)))
                    return False, f"pii_detected ({types})"

        return True, ""

    def setup(self, config: dict) -> None:
        """Optional setup from config dict."""
        if "max_detections" in config:
            self.max_detections = config["max_detections"]
        if "check_false_positives" in config:
            self.check_false_positives = config["check_false_positives"]
