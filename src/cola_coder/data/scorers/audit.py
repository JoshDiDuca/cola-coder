"""Audit logging for scoring pipeline operations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class AuditEntry:
    """Single audit log entry for a scoring operation."""
    timestamp: str = ""
    scorer: str = ""
    file_hash: str = ""
    security_mode: str = ""
    command: list[str] = field(default_factory=list)
    exit_code: int = 0
    duration_ms: float = 0.0
    security_events: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class ScoringAuditLogger:
    """Append-only JSONL audit log for all scoring operations.

    Thread-safe via file append (atomic on most OSes for <4KB writes).
    """

    def __init__(self, log_path: str | Path = "logs/scoring_audit.jsonl") -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: AuditEntry) -> None:
        """Append a single audit entry."""
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def log_security_event(
        self,
        event: str,
        scorer: str = "",
        file_hash: str = "",
    ) -> None:
        """Log a security event (credential detected, sandbox bypass, etc.)."""
        entry = AuditEntry(
            scorer=scorer,
            file_hash=file_hash,
            security_events=[event],
        )
        self.log(entry)

    @property
    def path(self) -> Path:
        return self._path
