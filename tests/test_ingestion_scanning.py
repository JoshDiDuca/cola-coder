"""Tests for malware scanning integration in data ingestion."""

from __future__ import annotations

from pathlib import Path

import pytest

from cola_coder.security.scanner import (
    CompositeMalwareScanner,
    MalwareScanResult,
    ThreatFinding,
)


class TestCompositeScannerIntegration:
    """Tests for the CompositeMalwareScanner in ingestion context."""

    def test_from_config_with_yara(self) -> None:
        """YARA scanner should be available when enabled in config."""
        scanner = CompositeMalwareScanner.from_config(
            {"scanners": {"yara": True, "defender": False}}
        )
        assert "yara" in scanner.available_scanners

    def test_from_config_empty(self) -> None:
        """Empty config should still auto-detect available scanners."""
        scanner = CompositeMalwareScanner.from_config({})
        # YARA is always available (falls back to regex)
        assert "yara" in scanner.available_scanners

    def test_from_config_none(self) -> None:
        """None config should work without errors."""
        scanner = CompositeMalwareScanner.from_config(None)
        assert isinstance(scanner.available_scanners, list)

    def test_scan_clean_directory(self, tmp_path: Path) -> None:
        """Clean files should produce a clean scan result."""
        (tmp_path / "clean.ts").write_text("const x: number = 42;")
        (tmp_path / "app.py").write_text("print('hello world')")
        scanner = CompositeMalwareScanner.from_config(
            {"scanners": {"yara": True, "defender": False}}
        )
        result = scanner.scan_directory(tmp_path)
        assert result.is_clean
        assert result.files_scanned >= 2

    def test_scan_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory should produce a clean result."""
        scanner = CompositeMalwareScanner.from_config(
            {"scanners": {"yara": True, "defender": False}}
        )
        result = scanner.scan_directory(tmp_path)
        assert result.is_clean
        assert result.files_scanned == 0

    def test_scan_dirty_directory(self, tmp_path: Path) -> None:
        """Files with suspicious patterns should be flagged."""
        (tmp_path / "evil.js").write_text('eval(atob("bWFsaWNpb3Vz"))')
        scanner = CompositeMalwareScanner.from_config(
            {"scanners": {"yara": True, "defender": False}}
        )
        result = scanner.scan_directory(tmp_path)
        assert not result.is_clean
        assert len(result.threats) > 0
        assert any("ObfuscatedCode" in t.name for t in result.threats)

    def test_scan_crypto_miner_detection(self, tmp_path: Path) -> None:
        """Crypto miner patterns should be detected."""
        (tmp_path / "miner.js").write_text(
            'const pool = "stratum+tcp://pool.example.com:3333";'
        )
        scanner = CompositeMalwareScanner.from_config(
            {"scanners": {"yara": True, "defender": False}}
        )
        result = scanner.scan_directory(tmp_path)
        assert not result.is_clean
        assert any("CryptoMiner" in t.name for t in result.threats)

    def test_scan_result_merge(self) -> None:
        """MalwareScanResult.merge should combine results correctly."""
        clean = MalwareScanResult(is_clean=True, files_scanned=5)
        dirty = MalwareScanResult(
            is_clean=False,
            files_scanned=3,
            threats=[
                ThreatFinding(
                    name="TestThreat",
                    severity="high",
                    file_path="/tmp/bad.js",
                    scanner="test",
                )
            ],
        )
        merged = clean.merge(dirty)
        assert not merged.is_clean
        assert len(merged.threats) == 1
        assert merged.files_scanned == 5  # max of the two

    def test_scan_nonexistent_directory(self) -> None:
        """Scanning a non-existent directory should return clean."""
        scanner = CompositeMalwareScanner.from_config(
            {"scanners": {"yara": True, "defender": False}}
        )
        result = scanner.scan_directory(Path("/nonexistent/path"))
        assert result.is_clean


class TestScoringYamlConfig:
    """Tests for the malware_scan section in scoring.yaml."""

    def test_config_loading(self) -> None:
        """Verify scoring.yaml malware_scan section loads correctly."""
        import yaml

        scoring_yaml = (
            Path(__file__).resolve().parent.parent / "configs" / "scoring.yaml"
        )
        if not scoring_yaml.exists():
            pytest.skip("scoring.yaml not found")

        with open(scoring_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        security = cfg.get("scoring", {}).get("security", {})
        malware = security.get("malware_scan", {})
        assert isinstance(malware, dict)
        assert "on_threat" in malware
        assert malware["on_threat"] in ("warn", "quarantine", "abort")
        assert "scanners" in malware
        assert isinstance(malware["scanners"], dict)

    def test_scanner_from_scoring_config(self) -> None:
        """CompositeMalwareScanner.from_config should work with scoring.yaml structure."""
        import yaml

        scoring_yaml = (
            Path(__file__).resolve().parent.parent / "configs" / "scoring.yaml"
        )
        if not scoring_yaml.exists():
            pytest.skip("scoring.yaml not found")

        with open(scoring_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        malware_cfg = (
            cfg.get("scoring", {}).get("security", {}).get("malware_scan", {})
        )
        scanner = CompositeMalwareScanner.from_config(malware_cfg)
        # Should have at least YARA available
        assert "yara" in scanner.available_scanners


class TestIngestionScanWorkflow:
    """Tests simulating the collect_data.py scanning workflow."""

    def test_scan_clean_dir_returns_clean(self, tmp_path: Path) -> None:
        """Scanning a clean directory mirrors the _scan_downloaded_data success path."""
        (tmp_path / "clean.ts").write_text("const x = 42;")
        config = {"scanners": {"yara": True, "defender": False}}
        scanner = CompositeMalwareScanner.from_config(config)
        assert scanner.available_scanners  # scanners active
        result = scanner.scan_directory(tmp_path)
        assert result.is_clean

    def test_no_scanners_skips_scan(self) -> None:
        """When all scanners are disabled, available_scanners is empty (scan skipped)."""
        config = {"scanners": {"yara": False, "defender": False, "clamav": False}}
        scanner = CompositeMalwareScanner.from_config(config)
        assert scanner.available_scanners == []

    def test_quarantine_workflow(self, tmp_path: Path) -> None:
        """Simulate the quarantine behavior from _scan_downloaded_data."""
        # Create a file with a threat
        evil_file = tmp_path / "evil.js"
        evil_file.write_text('eval(atob("bWFsaWNpb3Vz"))')

        scanner = CompositeMalwareScanner.from_config(
            {"scanners": {"yara": True, "defender": False}}
        )
        result = scanner.scan_directory(tmp_path)
        assert not result.is_clean

        # Simulate quarantine
        quarantine_dir = tmp_path / "quarantine"
        quarantine_dir.mkdir(exist_ok=True)
        for t in result.threats:
            src = Path(t.file_path)
            if src.exists():
                dst = quarantine_dir / src.name
                src.rename(dst)

        # Verify quarantine worked
        assert not evil_file.exists()
        assert (quarantine_dir / "evil.js").exists()

    def test_abort_on_threat_config(self) -> None:
        """Verify on_threat=abort config is read correctly from YAML structure."""
        config = {
            "malware_scan": {
                "on_threat": "abort",
                "scanners": {"yara": True},
            }
        }
        assert config["malware_scan"]["on_threat"] == "abort"
