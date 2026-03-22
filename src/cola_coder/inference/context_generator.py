"""Context-aware code generation with automatic repository context.

Wraps CodeGenerator to automatically prepend a <|repo|>...</|repo|> context
block before every generation call.  The context block contains:
  - Project name and key framework versions
  - Exported types/signatures from files the target imports
  - Exported types/signatures from similar files (Jaccard similarity)

For a TS dev: this is like having your IDE's language server feed type
information directly into the prompt — the model sees the same context your
editor does, so it generates code that fits the existing codebase.

Usage:
    generator = CodeGenerator(model, tokenizer, device)
    ctx_gen = ContextAwareGenerator(generator, Path("/path/to/myapp"))

    # One-shot
    result = ctx_gen.generate("src/api/users.ts", prompt)

    # Streaming
    for chunk in ctx_gen.generate_stream("src/api/users.ts", prompt):
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

from .generator import CodeGenerator
from .repo_context import RepoScanner

logger = logging.getLogger(__name__)

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


class ContextAwareGenerator:
    """CodeGenerator wrapper that prepends repository context to every prompt.

    The repository scan runs once during __init__ so the per-call overhead is
    only the context assembly (string operations, no disk I/O after scan).

    Args:
        generator: A fully initialised CodeGenerator instance.
        repo_root: Root directory of the TypeScript/JavaScript project.
        eager_scan: If True (default), scan the repo immediately in __init__.
                    Set to False to defer scanning until the first generate call
                    (useful for testing or lazy initialisation).
    """

    def __init__(
        self,
        generator: CodeGenerator,
        repo_root: Path,
        eager_scan: bool = True,
    ) -> None:
        self.generator = generator
        self.scanner = RepoScanner(repo_root)
        self.context = self.scanner.scan() if eager_scan else None

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        file_path: str,
        prompt: str,
        max_context_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """Generate code with automatic repo context prepended.

        The context block is assembled from the target file's imports and
        similar files, trimmed to max_context_tokens before being prepended.

        Args:
            file_path: Path to the file being completed (relative or absolute).
                       Used to resolve imports and find similar files.
            prompt: The generation prompt (code prefix / instruction).
            max_context_tokens: Token budget for the repo context block.
                                 Does not affect the prompt or generation length.
            **kwargs: Forwarded to CodeGenerator.generate() (temperature, top_k,
                      top_p, repetition_penalty, max_new_tokens, stop_tokens).

        Returns:
            Generated text (context + prompt + new tokens, as decoded string).
        """
        self._ensure_scanned()
        context_str = self.scanner.get_context_for_file(file_path, max_tokens=max_context_tokens)
        full_prompt = context_str + prompt
        logger.debug(
            "generate: file=%s context_chars=%d prompt_chars=%d",
            file_path,
            len(context_str),
            len(prompt),
        )
        return self.generator.generate(full_prompt, **kwargs)

    def generate_stream(
        self,
        file_path: str,
        prompt: str,
        max_context_tokens: int = 2048,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Streaming generation with repo context prepended.

        Yields incremental text chunks as tokens are produced, identical to
        CodeGenerator.generate_stream() but with context automatically prepended.

        Args:
            file_path: Path to the file being completed.
            prompt: The generation prompt.
            max_context_tokens: Token budget for the repo context block.
            **kwargs: Forwarded to CodeGenerator.generate_stream().

        Yields:
            Incremental decoded text chunks.
        """
        self._ensure_scanned()
        context_str = self.scanner.get_context_for_file(file_path, max_tokens=max_context_tokens)
        full_prompt = context_str + prompt
        logger.debug(
            "generate_stream: file=%s context_chars=%d",
            file_path,
            len(context_str),
        )
        yield from self.generator.generate_stream(full_prompt, **kwargs)

    def rescan(self) -> None:
        """Re-scan the repository (e.g. after adding new files).

        Useful in long-running sessions where the project changes on disk.
        The new scan replaces the cached context and token sets.
        """
        logger.info("Re-scanning repository: %s", self.scanner.root)
        self.context = self.scanner.scan()

    def get_repo_summary(self) -> str:
        """Return a human-readable repository summary for CLI display.

        Example output::

            Repository: my-app
            Files: 142 total, 89 TS/JS
            Imports parsed: 312
            Frameworks: next@14.2.3, react@18.2.0, zod@3.22.0
            tsconfig.json: found

        Returns:
            Plain-text multi-line summary (no ANSI codes).
        """
        self._ensure_scanned()
        return self.scanner.get_repo_summary()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_scanned(self) -> None:
        """Run scan() if it hasn't been run yet (deferred-scan mode)."""
        if self.context is None:
            logger.info("Deferred scan triggered by first generate() call")
            self.context = self.scanner.scan()
