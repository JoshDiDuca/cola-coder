"""Canonical prompt-injection static patterns + scanner (single source of truth).

Static (NO execution) detection of INDIRECT prompt-injection / jailbreak directives
and hidden-content vectors in UNTRUSTED RETRIEVED text — fetched framework docs,
scraped repo files, tool output, RAG/repo context — BEFORE it is assembled into a
model prompt. OWASP ranks prompt injection LLM01 (the #1 LLM risk, 3rd year running);
poisoned retrieved content is the dominant INDIRECT vector (hidden markdown in docs,
fake chat markers, invisible control characters that smuggle instructions).

This is a defense-in-depth LAYER, not a guarantee — it flags the common, high-signal
patterns cheaply so callers can warn/annotate/skip. Patterns are high-precision to
avoid flagging docs that merely *discuss* prompt injection; tune toward false negatives
over false positives (a warning, not a hard block, at the fetch sites).

Mirrors security/code_patterns.py (dangerous-code scanner) — different threat class
(injection of INSTRUCTIONS vs execution of dangerous CODE), same shape.
"""

from __future__ import annotations

import re

# (regex, human-readable name). Compiled case-insensitively below.
INJECTION_PATTERNS: list[tuple[str, str]] = [
    # --- Instruction override / jailbreak ---
    (r"ignore\s+(?:all\s+|the\s+|any\s+)?(?:previous|prior|above|earlier|preceding)"
     r"\s+(?:instructions?|prompts?|messages?|context)", "Ignore-previous-instructions override"),
    (r"disregard\s+(?:all\s+|your\s+|the\s+|any\s+)?(?:instructions?|rules?|"
     r"system\s+prompt|guidelines?)", "Disregard-instructions override"),
    (r"(?:new|updated|revised)\s+(?:instructions?|system\s+prompt|rules?)\s*:",
     "Injected new-instructions block"),
    (r"ignore\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|guardrails?)",
     "Ignore-system-prompt override"),
    # --- System-prompt exfiltration ---
    (r"(?:reveal|print|repeat|show|output|disclose)\s+(?:me\s+)?(?:your\s+|the\s+)?"
     r"(?:system\s+prompt|initial\s+prompt|instructions?\s+above)",
     "System-prompt exfiltration request"),
    # --- Secret / credential exfiltration ---
    (r"(?:exfiltrate|leak|upload|send|post|email|transmit|curl|fetch)\s+[^\n]{0,40}?"
     r"(?:ssh\s+key|api[_\s-]?key|secret|password|credential|token|\.env\b|env\s+var)",
     "Secret/credential exfiltration directive"),
    # --- Pipe-to-shell embedded in retrieved content ---
    (r"(?:curl|wget)\s+[^\n|]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh)\b",
     "Pipe-to-shell command in retrieved content"),
    # --- Fake chat/role markers trying to inject a system/assistant turn ---
    (r"<\|im_start\|>\s*system", "Injected ChatML system turn"),
    (r"<\s*/?system\s*>", "Injected <system> tag"),
    (r"\[/?INST\]", "Injected [INST] instruction marker"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), name) for p, name in INJECTION_PATTERNS]

# Invisible / directional control characters used to HIDE injected instructions
# (zero-width spaces, bidi overrides/isolates) — legitimate plain docs don't need them.
_INVISIBLE_CHARS = frozenset(
    "​‌‍﻿"          # zero-width space/non-joiner/joiner, ZWNBSP
    "‪‫‬‭‮"    # bidi embeddings / overrides
    "⁦⁧⁨⁩"         # bidi isolates
)


def scan_injection(text: str) -> list[str]:
    """Return the names of prompt-injection patterns found in ``text`` ([] = clean)."""
    if not text:
        return []
    found = [name for rx, name in _COMPILED if rx.search(text)]
    if any(ch in _INVISIBLE_CHARS for ch in text):
        found.append("Hidden/invisible control characters")
    return found


def has_injection(text: str) -> bool:
    """True if ``text`` contains any prompt-injection pattern."""
    return bool(scan_injection(text))
