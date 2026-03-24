"""Automatic memory updates from interactions.

Extracts patterns, errors, and knowledge from conversations
and appends them to the appropriate memory files.

This is intentionally conservative — only extracts high-confidence
patterns to avoid filling memory with noise.
"""

import re
from dataclasses import dataclass

from cola_coder.memory.manager import MemoryManager


@dataclass
class ExtractedFact:
    """A fact extracted from an interaction."""

    category: str  # "pattern", "error", "decision", "knowledge"
    content: str
    confidence: float  # 0-1, only store if > 0.5


class MemoryUpdater:
    """Extracts and stores knowledge from interactions.

    Heuristic extraction (no model needed):
    - Error patterns: regex for error messages, stack traces
    - Code patterns: repeated code structures
    - Decisions: "we decided", "I chose", "the approach is"
    - Domain facts: API references, library usage
    """

    def __init__(self, manager: MemoryManager):
        self.manager = manager

    def process_interaction(
        self,
        prompt: str,
        response: str,
        domain: str = "",
    ) -> list[ExtractedFact]:
        """Extract facts from an interaction and store them.

        Args:
            prompt: User's prompt
            response: Model's response
            domain: Detected domain

        Returns:
            List of extracted facts (for transparency/debugging)
        """
        facts = []

        # Extract errors
        for error in self._extract_errors(response):
            facts.append(error)
            if error.confidence > 0.5:
                self.manager.add_error(error.content)

        # Extract patterns (from code blocks in response)
        for pattern in self._extract_patterns(response):
            facts.append(pattern)
            if pattern.confidence > 0.6:
                self.manager.add_pattern(pattern.content)

        # Log session
        summary = prompt[:200] if len(prompt) > 200 else prompt
        self.manager.log_session(summary, domain)

        return facts

    def _extract_errors(self, text: str) -> list[ExtractedFact]:
        """Extract error messages from text."""
        facts = []

        # Common error patterns
        patterns = [
            (r"(?:Error|error|ERROR)[\s:]+(.+?)(?:\n|$)", 0.7),
            (r"(?:TypeError|ReferenceError|SyntaxError):\s*(.+?)(?:\n|$)", 0.8),
            (r"TS\d{4}:\s*(.+?)(?:\n|$)", 0.8),  # TypeScript errors
            (r"(?:ENOENT|EACCES|EPERM):\s*(.+?)(?:\n|$)", 0.7),
        ]

        for pattern, confidence in patterns:
            matches = re.findall(pattern, text)
            for match in matches[:3]:  # Cap per pattern
                facts.append(
                    ExtractedFact(
                        category="error",
                        content=match.strip()[:200],
                        confidence=confidence,
                    )
                )

        return facts

    def _extract_patterns(self, text: str) -> list[ExtractedFact]:
        """Extract code patterns from response."""
        facts = []

        # Find code blocks
        code_blocks = re.findall(r"```\w*\n(.+?)```", text, re.DOTALL)

        for block in code_blocks[:3]:
            # Only extract substantial patterns (not tiny snippets)
            if len(block) < 50 or len(block) > 500:
                continue

            # Look for function/component definitions
            if re.search(r"(?:function|const|class|interface|type)\s+\w+", block):
                facts.append(
                    ExtractedFact(
                        category="pattern",
                        content=block.strip()[:300],
                        confidence=0.5,
                    )
                )

        return facts
