"""License filter for code training data.

Only keeps files from repositories with permissive open-source licenses.
This is important for both legal compliance and ethical training data
curation.
"""

import logging

from cola_coder.data.registry import register_filter

logger = logging.getLogger(__name__)


@register_filter("license")
class LicenseFilter:
    """Only keep files from repos with permissive licenses.

    Checks record.metadata for a "license" field and rejects files from
    repos with non-permissive or unknown licenses. This is the same
    approach used by StarCoder and The Stack.
    """

    # SPDX identifiers for permissive licenses
    PERMISSIVE = {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "Unlicense",
        "0BSD",
        "CC0-1.0",
        "MIT-0",
        "BSL-1.0",       # Boost Software License
        "PostgreSQL",
        "Zlib",
        "X11",
    }

    # Common aliases / alternate spellings to normalize
    _ALIASES = {
        "mit": "MIT",
        "apache-2": "Apache-2.0",
        "apache 2.0": "Apache-2.0",
        "apache2": "Apache-2.0",
        "bsd-2": "BSD-2-Clause",
        "bsd-3": "BSD-3-Clause",
        "bsd 2-clause": "BSD-2-Clause",
        "bsd 3-clause": "BSD-3-Clause",
        "isc": "ISC",
        "unlicense": "Unlicense",
        "cc0": "CC0-1.0",
        "cc0-1.0": "CC0-1.0",
    }

    def __init__(self, allow_unknown: bool = False):
        """
        Args:
            allow_unknown: If True, pass files with no license info.
                If False (default), reject files with missing license.
        """
        self.allow_unknown = allow_unknown
        # SPDX license identifiers are case-insensitive (SPDX spec §10.1).
        # Match on lowercase so a valid permissive license arriving in
        # non-canonical case — e.g. "apache-2.0", "zlib", "bsd-3-clause" — is
        # NOT wrongly rejected (which silently discarded valid training data;
        # the small alias map only covered a handful of lowercase forms).
        self._permissive_lower = {s.lower() for s in self.PERMISSIVE}

    def name(self) -> str:
        return "license"

    def _normalize_license(self, license_str: str) -> str:
        """Normalize license string to SPDX identifier."""
        stripped = license_str.strip()
        # Check aliases (case-insensitive)
        lower = stripped.lower()
        if lower in self._ALIASES:
            return self._ALIASES[lower]
        # Return as-is for direct SPDX match
        return stripped

    def check(self, record) -> tuple[bool, str]:
        """Check if the file's repository has a permissive license.

        Args:
            record: Object with .content (str) and .metadata (dict) attributes.

        Returns:
            (keep, reason) tuple.
        """
        license_raw = record.metadata.get("license", "")

        if not license_raw:
            if self.allow_unknown:
                return True, ""
            return False, "no_license"

        normalized = self._normalize_license(license_raw)

        # Case-insensitive SPDX match (the alias map still handles alternate
        # SPELLINGS like "apache2"; this handles CASING).
        if normalized.lower() in self._permissive_lower:
            return True, ""

        return False, f"non_permissive_license ({normalized})"

    def setup(self, config: dict) -> None:
        """Optional setup from config dict."""
        if "allow_unknown" in config:
            self.allow_unknown = config["allow_unknown"]
