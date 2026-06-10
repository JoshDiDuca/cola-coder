"""YARA-based code threat scanner for detecting malicious patterns in source code."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from cola_coder.security.scanner import MalwareScanResult, ThreatFinding

logger = logging.getLogger(__name__)


# Code extensions to scan
_CODE_EXTENSIONS = {
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".py", ".sh", ".bash", ".bat", ".cmd", ".ps1",
    ".json", ".yaml", ".yml", ".toml",
}

# Embedded YARA rules (no external files needed)
_YARA_RULES_SOURCE = r"""
rule CryptoMiner {
    meta:
        description = "Cryptocurrency mining code"
        severity = "high"
    strings:
        $stratum = "stratum+tcp://" nocase
        $stratum_ssl = "stratum+ssl://" nocase
        $mining_pool1 = /pool\.(minergate|nanopool|f2pool|antpool)/ nocase
        $coinhive = "CoinHive" nocase
        $cryptonight = "cryptonight" nocase
        $xmrig = "xmrig" nocase
        $monero_addr = /4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}/
    condition:
        any of them
}

rule ReverseShell {
    meta:
        description = "Reverse shell or backdoor code"
        severity = "high"
    strings:
        $bash_reverse = /bash\s+-i\s+>&\s*\/dev\/tcp/
        $python_socket = /socket\.connect\(\s*\(\s*['"][^'"]+['"]\s*,\s*\d+\s*\)\s*\)/
        $nc_reverse = /\bnc\b.*-e\s+\/bin\/(sh|bash)/
        $powershell_reverse = /New-Object\s+System\.Net\.Sockets\.TCPClient/ nocase
        $perl_reverse = /use\s+Socket.*open\(STDIN/
        $ruby_reverse = /TCPSocket\.open/
    condition:
        any of them
}

rule ObfuscatedCode {
    meta:
        description = "Heavily obfuscated code (possible malware)"
        severity = "medium"
    strings:
        $long_hex = /\\x[0-9a-fA-F]{2}(\\x[0-9a-fA-F]{2}){50,}/
        $eval_atob = /eval\s*\(\s*atob\s*\(/ nocase
        $eval_decode = /eval\s*\(\s*(Buffer\.from|decodeURIComponent)\s*\(/
        $eval_fromcharcode = /eval\s*\(\s*String\.fromCharCode/
        $exec_base64 = /exec\s*\(\s*base64/
        $char_array_join = /\[[\d,\s]{100,}\]\.map\(.*String\.fromCharCode/
    condition:
        any of them
}

rule DangerousNodeImports {
    meta:
        description = "Dangerous Node.js imports that could execute system commands"
        severity = "medium"
    strings:
        $child_process = /require\s*\(\s*['"]child_process['"]\s*\)/
        $exec_sync = /execSync\s*\(/
        $spawn_shell = /spawn\s*\(\s*['"](?:cmd|bash|sh|powershell)['"]/
        $fs_write_root = /writeFileSync\s*\(\s*['"]\/(?:etc|usr|var|tmp)\//
        $process_env_token = /process\.env\.[A-Z_]*(?:TOKEN|KEY|SECRET|PASSWORD)/
    condition:
        2 of them
}

rule DataExfiltration {
    meta:
        description = "Code that sends data to external servers"
        severity = "medium"
    strings:
        $webhook = /https?:\/\/(?:hooks\.slack\.com|discord\.com\/api\/webhooks|webhook\.site)/
        $ngrok = "ngrok.io" nocase
        $pastebin = /pastebin\.com\/api/
        $telegram_bot = /api\.telegram\.org\/bot/
    condition:
        any of them
}

rule PostInstallScript {
    meta:
        description = "Suspicious npm postinstall script patterns"
        severity = "high"
    strings:
        $postinstall_exec = /"(?:pre|post)?install"\s*:\s*"[^"]*(?:curl|wget|powershell|cmd|bash)/
        $postinstall_node = /"(?:pre|post)?install"\s*:\s*"node\s+[^"]*\.js"/
        $rundll32 = "rundll32" nocase
    condition:
        any of them
}
"""


class YaraScanner:
    """Scan code files for malicious patterns using YARA rules."""

    name: str = "yara"

    def __init__(self, rules_dir: str | Path | None = None) -> None:
        self._rules = None
        self._rules_dir = Path(rules_dir) if rules_dir else None
        self._init_error: str | None = None
        try:
            self._compile_rules()
        except Exception as e:
            self._init_error = str(e)

    def _compile_rules(self) -> None:
        """Compile YARA rules from embedded source and optional external files."""
        try:
            import yara
        except ImportError:
            # Fall back to regex-based scanning if yara-python not installed
            self._rules = None
            return

        sources = {"embedded": _YARA_RULES_SOURCE}

        # Load external rule files if directory provided
        if self._rules_dir and self._rules_dir.is_dir():
            for rule_file in self._rules_dir.glob("*.yar"):
                sources[rule_file.stem] = rule_file.read_text(encoding="utf-8")

        self._rules = yara.compile(sources=sources)

    def scan_file(self, path: Path) -> MalwareScanResult:
        """Scan a single file for threats."""
        start = time.perf_counter()
        path = Path(path)

        if not path.exists() or not path.is_file():
            return MalwareScanResult(is_clean=True, files_scanned=0)

        if path.suffix.lower() not in _CODE_EXTENSIONS:
            return MalwareScanResult(is_clean=True, files_scanned=1)

        threats = self._scan_file_content(path)
        duration = (time.perf_counter() - start) * 1000

        return MalwareScanResult(
            is_clean=len(threats) == 0,
            threats=threats,
            files_scanned=1,
            scan_duration_ms=duration,
        )

    def scan_directory(self, path: Path) -> MalwareScanResult:
        """Scan all code files in a directory recursively."""
        start = time.perf_counter()
        path = Path(path)
        all_threats: list[ThreatFinding] = []
        files_scanned = 0

        if not path.exists() or not path.is_dir():
            return MalwareScanResult()

        for file_path in path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in _CODE_EXTENSIONS:
                continue
            files_scanned += 1
            threats = self._scan_file_content(file_path)
            all_threats.extend(threats)

        duration = (time.perf_counter() - start) * 1000
        return MalwareScanResult(
            is_clean=len(all_threats) == 0,
            threats=all_threats,
            files_scanned=files_scanned,
            scan_duration_ms=duration,
        )

    def scan_text(
        self,
        content: str,
        identifier: str = "<stream>",
    ) -> list[ThreatFinding]:
        """Scan in-memory text for threats (no file needed).

        This is the right entry point for streamed data (HuggingFace
        records) BEFORE tokenization — scanning tokenized .npy output is
        useless because token IDs carry none of the textual patterns the
        rules match on.

        Args:
            content: The raw text/code to scan.
            identifier: Label used in findings instead of a file path
                (e.g. "code#1042").

        Returns:
            List of threats found (empty = clean).
        """
        pseudo_path = Path(identifier)
        if self._rules is not None:
            return self._scan_with_yara(pseudo_path, content)
        return self._scan_with_regex(pseudo_path, content)

    def _scan_file_content(self, path: Path) -> list[ThreatFinding]:
        """Scan a single file's content."""
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            return []

        if self._rules is not None:
            return self._scan_with_yara(path, content)
        else:
            return self._scan_with_regex(path, content)

    def _scan_with_yara(self, path: Path, content: str) -> list[ThreatFinding]:
        """Use compiled YARA rules."""
        threats: list[ThreatFinding] = []
        try:
            matches = self._rules.match(data=content.encode("utf-8", errors="ignore"))
            for match in matches:
                severity = "medium"
                if hasattr(match, "meta") and "severity" in match.meta:
                    severity = match.meta["severity"]
                description = ""
                if hasattr(match, "meta") and "description" in match.meta:
                    description = match.meta["description"]
                threats.append(ThreatFinding(
                    name=match.rule,
                    severity=severity,
                    file_path=str(path),
                    scanner=self.name,
                    details=description,
                ))
            if threats:
                for t in threats:
                    logger.warning(
                        "YARA threat [%s/%s]: %s in %s",
                        t.name, t.severity, t.details, t.file_path,
                    )
        except Exception as e:
            logger.debug("YARA scan error: %s", e)
        return threats

    def _scan_with_regex(self, path: Path, content: str) -> list[ThreatFinding]:
        """Fallback regex scanning when yara-python is not installed."""
        import re
        threats: list[ThreatFinding] = []

        # Simplified regex patterns matching the YARA rules
        patterns = [
            (r"stratum\+(?:tcp|ssl)://", "CryptoMiner", "high", "Mining pool connection"),
            (r"(?:CoinHive|cryptonight|xmrig)", "CryptoMiner", "high", "Crypto miner reference"),
            (r"bash\s+-i\s+>&\s*/dev/tcp", "ReverseShell", "high", "Bash reverse shell"),
            (r"New-Object\s+System\.Net\.Sockets\.TCPClient", "ReverseShell", "high", "PowerShell reverse shell"),
            (r"eval\s*\(\s*atob\s*\(", "ObfuscatedCode", "medium", "Eval with base64 decode"),
            (r"eval\s*\(\s*String\.fromCharCode", "ObfuscatedCode", "medium", "Eval with char codes"),
            (r'"(?:pre|post)?install"\s*:\s*"[^"]*(?:curl|wget|powershell|bash)', "PostInstallScript", "high", "Suspicious install script"),
            (r"rundll32", "PostInstallScript", "high", "DLL execution"),
            (r"(?:hooks\.slack\.com|discord\.com/api/webhooks|webhook\.site)", "DataExfiltration", "medium", "Webhook data exfiltration"),
            (r"api\.telegram\.org/bot", "DataExfiltration", "medium", "Telegram bot API"),
        ]

        for pattern, name, severity, details in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                threats.append(ThreatFinding(
                    name=name, severity=severity,
                    file_path=str(path), scanner=f"{self.name}_regex",
                    details=details,
                ))

        if threats:
            for t in threats:
                logger.warning(
                    "YARA regex threat [%s/%s]: %s in %s",
                    t.name, t.severity, t.details, t.file_path,
                )

        return threats

    def is_available(self) -> bool:
        """Always available -- falls back to regex if yara-python not installed."""
        return True
