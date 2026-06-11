"""Metadata enrichment transform.

Computes and attaches metadata to DataRecords as they flow
through the pipeline. Useful for downstream analysis and filtering.
"""

from __future__ import annotations

import hashlib

from cola_coder.data.pipeline import DataRecord, Transform
from cola_coder.data.registry import register_transform


@register_transform("add_metadata")
class AddMetadata(Transform):
    """Enrich DataRecords with computed metadata.

    Adds the following metadata fields:
    - line_count: Number of lines in the content
    - char_count: Number of characters
    - content_hash: SHA-256 hash of the content (for dedup)
    - estimated_language: Best guess at programming language
    """

    def name(self) -> str:
        return "add_metadata"

    def apply(self, record: DataRecord) -> DataRecord:
        content = record.content
        metadata = dict(record.metadata)  # copy to avoid mutation

        # Basic stats
        metadata["line_count"] = content.count("\n") + 1
        metadata["char_count"] = len(content)

        # Content hash for dedup
        metadata["content_hash"] = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()[:16]

        # Language detection (simple heuristic)
        if "estimated_language" not in metadata:
            metadata["estimated_language"] = _guess_language(content)

        return DataRecord(content=content, metadata=metadata)


def _guess_language(content: str) -> str:
    """Simple heuristic language detection from content.

    Not meant to be accurate for all cases — just a best-effort guess
    based on common patterns in the first 2000 characters.
    """
    header = content[:2000]

    # Check for strong signals
    if "#!/usr/bin/env python" in header or "#!/usr/bin/python" in header:
        return "python"
    if "#!/usr/bin/env node" in header or "#!/usr/bin/env ts-node" in header:
        return "javascript"

    # Count language-specific keywords
    scores: dict[str, int] = {
        "python": 0,
        "typescript": 0,
        "javascript": 0,
        "go": 0,
        "rust": 0,
        "java": 0,
    }

    py_signals = ["def ", "self.", "elif ", "except ", "import ", "__init__"]
    ts_signals = ["interface ", ": string", ": number", ": boolean", "readonly "]
    js_signals = ["const ", "let ", "=> ", "require(", "module.exports"]
    go_signals = ["func ", "package ", "fmt.", "go func", ":= "]
    rust_signals = ["fn ", "let mut ", "impl ", "pub fn", "use std::"]
    java_signals = ["public class", "private ", "protected ", "System.out", "@Override"]

    for s in py_signals:
        if s in header:
            scores["python"] += 1
    for s in ts_signals:
        if s in header:
            scores["typescript"] += 1
    for s in js_signals:
        if s in header:
            scores["javascript"] += 1
    for s in go_signals:
        if s in header:
            scores["go"] += 1
    for s in rust_signals:
        if s in header:
            scores["rust"] += 1
    for s in java_signals:
        if s in header:
            scores["java"] += 1

    # Return highest scoring language, or "unknown"
    best = max(scores, key=lambda k: scores[k])
    if scores[best] >= 2:
        return best
    return "unknown"
