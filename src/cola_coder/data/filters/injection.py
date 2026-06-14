"""Prompt-injection filter plugin (DATA-063).

A from-scratch model trains on SCRAPED code whose comments / docstrings / string
literals can carry prompt-injection payloads ("ignore previous instructions",
hidden exfiltration directives, invisible control characters). Training on them
risks teaching the model to EMIT or OBEY injections — a data-poisoning vector.

This filter reuses the canonical SEC-019 scanner (`security.injection_patterns`)
to drop scraped records carrying injection payloads at data-prep time, closing the
loop between input-time defense (SEC-019, doc fetcher) and training-time hygiene.

Opt-in: like every filter plugin it only runs when added to the pipeline's filter
chain. High-precision patterns keep false positives low, but a record that merely
DISCUSSES prompt injection (e.g. a security tool's source, an injection test
fixture) can trip it — acceptable for scraped pretraining data, where a file full
of exfiltration directives is exactly what we don't want to learn.
"""

from __future__ import annotations

from typing import Any

from cola_coder.data.pipeline import DataRecord, FilterPlugin
from cola_coder.data.registry import register_filter
from cola_coder.security.injection_patterns import scan_injection


@register_filter("injection")
class InjectionFilter(FilterPlugin):
    """Drop records whose content carries prompt-injection payloads.

    Config options (via setup() or YAML):
        min_hits: Minimum number of distinct injection patterns to trigger a drop
            (default 1). Raise to 2+ to require corroborating signals and further
            cut false positives on security-related source.
    """

    def __init__(self, min_hits: int = 1) -> None:
        self._min_hits = max(1, int(min_hits))

    def name(self) -> str:
        return f"injection(min_hits={self._min_hits})"

    def setup(self, config: dict[str, Any]) -> None:
        self._min_hits = max(1, int(config.get("min_hits", self._min_hits)))

    def check(self, record: DataRecord) -> tuple[bool, str]:
        hits = scan_injection(record.content)
        if len(hits) >= self._min_hits:
            return False, f"prompt_injection ({', '.join(hits[:3])})"
        return True, ""
