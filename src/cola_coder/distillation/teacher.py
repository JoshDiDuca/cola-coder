"""Distillation teacher backends.

A ``Teacher`` turns a list of chat messages into a completion string. The default
backend, :class:`OpenAICompatibleTeacher`, speaks the OpenAI ``/chat/completions``
API — which is the lingua franca that BOTH local servers and cloud providers
implement, so one backend covers, e.g.:

* **Local (no extra GPU on the trainer):** Ollama (``http://localhost:11434/v1``,
  e.g. ``ollama run qwen2.5-coder`` or ``deepseek-coder``), llama.cpp ``--server``,
  vLLM, LM Studio, text-generation-webui — run on a spare GPU or CPU.
* **Cloud:** DeepSeek (``https://api.deepseek.com``), OpenAI, OpenRouter, Together,
  Qwen/DashScope (OpenAI-compatible mode).

Pick the teacher in ``configs/distillation.yaml`` and build it with
:func:`build_teacher`. A direct in-process HuggingFace teacher (load Qwen/DeepSeek
weights without a server) is a planned follow-up (backlog MODEL-024) — for now,
serve a local model with Ollama/llama.cpp and point the OpenAI-compatible backend
at it, which keeps the teacher off the training GPU.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

import requests

from ..data.scorers.credential_scanner import CredentialScanner


class TeacherError(RuntimeError):
    """Raised when a teacher backend fails (network, non-200, malformed reply)."""


@runtime_checkable
class Teacher(Protocol):
    """A distillation teacher: chat messages in, completion text out."""

    name: str

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> str:
        ...


def _is_local_url(base_url: str) -> bool:
    """True for localhost / loopback endpoints (no secret-leak risk)."""
    try:
        host = (urlparse(base_url).hostname or "").lower()
    except ValueError:
        return False
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or host.endswith(".local")


class OpenAICompatibleTeacher:
    """Teacher backed by any OpenAI-compatible ``/chat/completions`` endpoint.

    Args:
        model: Teacher model name (e.g. ``"qwen2.5-coder:7b"``, ``"deepseek-chat"``).
        base_url: API base, INCLUDING the version segment, e.g.
            ``"http://localhost:11434/v1"`` (Ollama) or ``"https://api.deepseek.com"``.
        api_key_env: Name of the env var holding the API key. Read lazily; the key
            is never stored on the instance or logged. Omit/None for keyless local
            servers (Ollama/llama.cpp need no key).
        timeout: Per-request timeout (seconds).
        redact_secrets: When True (default) and the endpoint is REMOTE (not
            localhost), prompts are scrubbed with :class:`CredentialScanner` before
            sending, so the user's code can't leak credentials to a cloud API. Local
            endpoints are never redacted (no external exposure).
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key_env: str | None = None,
        timeout: float = 120.0,
        redact_secrets: bool = True,
    ) -> None:
        if not model:
            raise ValueError("teacher 'model' is required")
        self.name = f"openai:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self.timeout = timeout
        self._is_local = _is_local_url(self.base_url)
        # Only redact when talking to a REMOTE endpoint — local servers stay on the
        # host so there's no leak surface (and redaction could corrupt the prompt).
        self._redact = redact_secrets and not self._is_local
        self._scanner = CredentialScanner(mode="strip") if self._redact else None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key_env:
            key = os.environ.get(self._api_key_env)
            if key:
                headers["Authorization"] = f"Bearer {key}"
            elif not self._is_local:
                raise TeacherError(
                    f"API key env var '{self._api_key_env}' is not set; "
                    f"cannot authenticate to remote teacher {self.base_url}"
                )
        return headers

    def _redact_messages(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        if self._scanner is None:
            return messages
        out: list[dict[str, str]] = []
        for m in messages:
            content = m.get("content", "")
            if content and self._scanner.scan(content).has_credentials:
                # process() strip-redacts the secret spans; fall back to the
                # original only if it returns None (never happens in strip mode).
                content = self._scanner.process(content) or content
            out.append({**m, "content": content})
        return out

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> str:
        if not messages:
            raise ValueError("messages must be non-empty")
        payload: dict[str, object] = {
            "model": self.model,
            "messages": self._redact_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        url = f"{self.base_url}/chat/completions"
        try:
            resp = requests.post(
                url, json=payload, headers=self._headers(), timeout=self.timeout
            )
        except requests.RequestException as e:  # network / timeout
            raise TeacherError(f"teacher request to {url} failed: {e}") from e

        if resp.status_code != 200:
            body = resp.text[:500]
            raise TeacherError(f"teacher {url} returned {resp.status_code}: {body}")

        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise TeacherError(f"malformed teacher response from {url}: {e}") from e

        if not isinstance(content, str):
            raise TeacherError(f"teacher returned non-string content: {type(content)}")
        return content


def build_teacher(cfg: dict) -> Teacher:
    """Construct a teacher from a config dict (see configs/distillation.yaml).

    Expected keys: ``backend`` (default ``"openai_compatible"``), plus
    backend-specific keys. Raises ``ValueError`` on an unknown backend so a typo
    fails loudly rather than silently picking a default.
    """
    backend = cfg.get("backend", "openai_compatible")
    if backend == "openai_compatible":
        return OpenAICompatibleTeacher(
            model=cfg["model"],
            base_url=cfg.get("base_url", "http://localhost:11434/v1"),
            api_key_env=cfg.get("api_key_env"),
            timeout=float(cfg.get("timeout", 120.0)),
            redact_secrets=bool(cfg.get("redact_secrets", True)),
        )
    if backend == "hf_local":
        # Planned follow-up (MODEL-024): load Qwen/DeepSeek weights in-process via
        # transformers on a chosen device (e.g. the spare GPU). Until then, serve
        # the model with Ollama/llama.cpp and use the openai_compatible backend —
        # which keeps the teacher off the training GPU and needs no new deps.
        raise NotImplementedError(
            "backend 'hf_local' (in-process transformers teacher) is not implemented "
            "yet — serve the local model with Ollama/llama.cpp/vLLM and use "
            "backend: openai_compatible pointed at it (e.g. http://localhost:11434/v1). "
            "Tracked as backlog MODEL-024."
        )
    raise ValueError(
        f"unknown distillation teacher backend '{backend}' "
        "(expected 'openai_compatible' or 'hf_local')"
    )
