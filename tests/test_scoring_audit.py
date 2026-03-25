"""Tests for ScoringAuditLogger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cola_coder.data.scorers.audit import AuditEntry, ScoringAuditLogger


class TestAuditEntry:
    def test_auto_timestamp(self) -> None:
        entry = AuditEntry(scorer="tsc", file_hash="abc123")
        assert entry.timestamp  # Non-empty
        assert "T" in entry.timestamp  # ISO format

    def test_explicit_timestamp(self) -> None:
        entry = AuditEntry(timestamp="2026-01-01T00:00:00Z", scorer="tsc")
        assert entry.timestamp == "2026-01-01T00:00:00Z"

    def test_default_fields(self) -> None:
        entry = AuditEntry()
        assert entry.scorer == ""
        assert entry.exit_code == 0
        assert entry.security_events == []


class TestScoringAuditLogger:
    def test_creates_log_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        logger = ScoringAuditLogger(log_path)
        entry = AuditEntry(scorer="tsc", exit_code=0)
        logger.log(entry)
        assert log_path.exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        log_path = tmp_path / "deep" / "nested" / "audit.jsonl"
        logger = ScoringAuditLogger(log_path)
        entry = AuditEntry(scorer="eslint")
        logger.log(entry)
        assert log_path.exists()

    def test_appends_jsonl(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        logger = ScoringAuditLogger(log_path)
        logger.log(AuditEntry(scorer="tsc"))
        logger.log(AuditEntry(scorer="eslint"))
        logger.log(AuditEntry(scorer="stars"))
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_entries_are_valid_json(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        logger = ScoringAuditLogger(log_path)
        logger.log(AuditEntry(scorer="tsc", file_hash="abc", exit_code=1, duration_ms=42.5))
        line = log_path.read_text().strip()
        data = json.loads(line)
        assert data["scorer"] == "tsc"
        assert data["file_hash"] == "abc"
        assert data["exit_code"] == 1
        assert data["duration_ms"] == 42.5

    def test_required_fields_present(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        logger = ScoringAuditLogger(log_path)
        logger.log(AuditEntry(scorer="tsc"))
        data = json.loads(log_path.read_text().strip())
        required = ["timestamp", "scorer", "file_hash", "security_mode", "command",
                     "exit_code", "duration_ms", "security_events"]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_log_security_event(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        logger = ScoringAuditLogger(log_path)
        logger.log_security_event("credential_stripped:AWS Access Key", scorer="llm_judge", file_hash="xyz")
        data = json.loads(log_path.read_text().strip())
        assert "credential_stripped:AWS Access Key" in data["security_events"]

    def test_path_property(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        logger = ScoringAuditLogger(log_path)
        assert logger.path == log_path
