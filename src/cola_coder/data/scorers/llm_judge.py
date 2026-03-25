"""LLM-as-Judge scorer — use Claude API or Ollama to score code quality."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from cola_coder.data.scorers.credential_scanner import CredentialScanner
from cola_coder.data.scorers.protocol import ScorerResult

_JUDGE_PROMPT = """Rate this {language} code on a scale of 0-5 for training data quality.

0 = Garbage: auto-generated, minified, data dump, or broken
1 = Very poor: no structure, terrible naming, no documentation
2 = Poor: some structure but significant quality issues
3 = Average: functional, reasonable structure, some documentation
4 = Good: well-structured, good naming, clear documentation
5 = Excellent: production-quality, idiomatic, educational value

Consider: correctness, readability, idiomatic patterns, documentation, educational value.

Reply in exactly this format:
Score: <0-5>
Reason: <one sentence>

Code:
```
{code}
```"""


def _parse_judge_response(text: str) -> tuple[int, str]:
    """Parse 'Score: N' and 'Reason: ...' from LLM response.

    Returns (score_0_5, reason). Falls back to (-1, "") on parse failure.
    """
    score = -1
    reason = ""

    score_match = re.search(r"Score:\s*(\d)", text)
    if score_match:
        score = int(score_match.group(1))
        score = max(0, min(5, score))

    reason_match = re.search(r"Reason:\s*(.+)", text)
    if reason_match:
        reason = reason_match.group(1).strip()

    return score, reason


def _code_hash(code: str) -> str:
    """MD5 hash of code for dedup/caching."""
    return hashlib.md5(code.encode("utf-8")).hexdigest()


class OllamaBackend:
    """Score code via Ollama local models."""

    def __init__(
        self,
        model: str = "codellama",
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def score_code(self, code: str, language: str = "TypeScript") -> tuple[int, str]:
        """Returns (score_0_5, explanation). (-1, "") on failure."""
        prompt = _JUDGE_PROMPT.format(language=language, code=code[:4000])
        response = self._call(prompt)
        if response is None:
            return -1, ""
        return _parse_judge_response(response)

    def _call(self, prompt: str) -> str | None:
        """Call Ollama API. Returns response text or None."""
        import urllib.request
        import urllib.error

        url = f"{self.base_url}/api/generate"
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "")
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
            return None

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                return any(self.model in m for m in models)
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return False


class ClaudeBackend:
    """Score code via Anthropic Claude API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        timeout: int = 30,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.timeout = timeout

    def score_code(self, code: str, language: str = "TypeScript") -> tuple[int, str]:
        """Returns (score_0_5, explanation). (-1, "") on failure."""
        try:
            import anthropic
        except ImportError:
            return -1, "anthropic SDK not installed"

        if not self.api_key:
            return -1, "No API key"

        prompt = _JUDGE_PROMPT.format(language=language, code=code[:4000])
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model=self.model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text = message.content[0].text if message.content else ""
            return _parse_judge_response(text)
        except Exception:
            return -1, ""

    def is_available(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(self.api_key)


class LlmJudge:
    """LLM-as-Judge scorer for code quality annotation."""

    name: str = "llm_judge"

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "codellama",
        api_key: str | None = None,
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
        credential_scanner: CredentialScanner | None = None,
    ) -> None:
        self.provider = provider
        self._scanner = credential_scanner
        if provider == "ollama":
            self._backend = OllamaBackend(model, base_url, timeout)
        elif provider == "claude":
            self._backend = ClaudeBackend(model, api_key, timeout)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'ollama' or 'claude'.")

    def score(self, code: str, metadata: dict[str, object] | None = None) -> ScorerResult:
        """Score a single code sample (expensive — use annotate_batch for bulk)."""
        # Scan for credentials before sending to external API
        if self._scanner is not None:
            processed = self._scanner.process(code)
            if processed is None:
                return ScorerResult(
                    score=0.5, scorer_name=self.name,
                    details={"skipped": True, "reason": "credential_detected"},
                )
            code = processed  # Use potentially redacted code

        language = "TypeScript"
        if metadata and "language" in metadata:
            language = str(metadata["language"])

        score_int, reason = self._backend.score_code(code, language)
        if score_int < 0:
            return ScorerResult(score=0.5, scorer_name=self.name, details={"error": True, "reason": reason})

        normalized = score_int / 5.0
        return ScorerResult(
            score=normalized,
            scorer_name=self.name,
            details={"score_raw": score_int, "reason": reason, "provider": self.provider},
        )

    def score_batch(self, items: list[tuple[str, dict[str, object] | None]]) -> list[ScorerResult]:
        """Score multiple samples sequentially (each is an LLM call)."""
        return [self.score(code, meta) for code, meta in items]

    def annotate_batch(
        self,
        codes: list[str],
        languages: list[str] | None = None,
        output_path: str = "data/annotations.jsonl",
    ) -> str:
        """Annotate a batch of code samples, saving to JSONL.

        Resume-capable: reads existing JSONL to find already-annotated hashes.

        Returns:
            Path to the annotations JSONL file.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Load existing annotations for resume
        existing_hashes: set[str] = set()
        if out.exists():
            with open(out, encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if "code_hash" in entry:
                            existing_hashes.add(entry["code_hash"])
                    except json.JSONDecodeError:
                        continue

        new_count = 0
        skipped = 0

        with open(out, "a", encoding="utf-8") as f:
            for i, code in enumerate(codes):
                h = _code_hash(code)
                if h in existing_hashes:
                    skipped += 1
                    continue

                # Scan for credentials before sending to external API
                scan_code = code
                if self._scanner is not None:
                    processed = self._scanner.process(code)
                    if processed is None:
                        continue  # Skip samples with credentials
                    scan_code = processed

                lang = languages[i] if languages and i < len(languages) else "TypeScript"
                score_int, reason = self._backend.score_code(scan_code, lang)

                if score_int < 0:
                    continue  # Skip failed annotations

                entry = {
                    "code_hash": h,
                    "score": score_int,
                    "reason": reason,
                    "provider": self.provider,
                    "language": lang,
                    "code_prefix": code[:200],
                }
                f.write(json.dumps(entry) + "\n")
                existing_hashes.add(h)
                new_count += 1

        return str(out)

    @staticmethod
    def is_available() -> bool:
        """At least one backend must be usable."""
        # Check Ollama
        try:
            backend = OllamaBackend()
            if backend.is_available():
                return True
        except Exception:
            pass
        # Check Claude
        if os.environ.get("ANTHROPIC_API_KEY"):
            return True
        return False
