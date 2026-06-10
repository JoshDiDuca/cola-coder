"""Tests for the OpenAI-compatible API endpoints in server.py.

Uses FastAPI's TestClient with a mock generator so no real model
or GPU is needed.
"""

import sys
from pathlib import Path

# Ensure the worktree's src/ is first on sys.path so we import the worktree's
# cola_coder package rather than whatever is installed in the venv.
_WORKTREE_SRC = str(Path(__file__).parent.parent / "src")
if _WORKTREE_SRC not in sys.path:
    sys.path.insert(0, _WORKTREE_SRC)

import pytest
from fastapi.testclient import TestClient

from cola_coder.inference.server import create_app
from cola_coder.model.config import ModelConfig


# ══════════════════════════════════════════════════════════════════════════════
# Mock generator
# ══════════════════════════════════════════════════════════════════════════════


class MockTokenizer:
    """Minimal tokenizer stub."""

    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        # Approximate: one token per whitespace-separated word
        return list(range(len(text.split())))

    def encode_fim(self, prefix: str, suffix: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return "decoded fim prompt"


class MockModel:
    """Minimal model stub that exposes metadata attributes."""

    def __init__(self) -> None:
        self.num_parameters = 50_000_000
        self.config = ModelConfig(vocab_size=32768, max_seq_len=4096)


class MockGenerator:
    """Drop-in replacement for CodeGenerator used by the server."""

    def __init__(self) -> None:
        self.tokenizer = MockTokenizer()
        self.model = MockModel()
        self.device = "cpu"

    def generate(self, prompt: str, **kwargs) -> str:
        # Return prompt + new text so strip logic in the server works
        return prompt + " generated text"

    def generate_stream(self, prompt: str, **kwargs):
        # Yield a few chunks; first chunk echoes the prompt (server strips it)
        yield prompt + " chunk_one"
        yield " chunk_two"
        yield " chunk_three"


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def client() -> TestClient:
    app = create_app(MockGenerator())
    return TestClient(app)


@pytest.fixture()
def cors_client() -> TestClient:
    app = create_app(MockGenerator(), enable_cors=True)
    return TestClient(app)


# ══════════════════════════════════════════════════════════════════════════════
# Health
# ══════════════════════════════════════════════════════════════════════════════


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_includes_uptime(self, client: TestClient) -> None:
        resp = client.get("/health")
        uptime = resp.json()["uptime_seconds"]
        assert uptime is not None
        assert uptime >= 0


# ══════════════════════════════════════════════════════════════════════════════
# Info
# ══════════════════════════════════════════════════════════════════════════════


class TestInfo:
    def test_info_returns_model_info(self, client: TestClient) -> None:
        resp = client.get("/info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_params"] == 50_000_000
        assert body["vocab_size"] == 32768
        assert body["max_seq_len"] == 4096
        assert body["device"] == "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# Legacy /generate
# ══════════════════════════════════════════════════════════════════════════════


class TestLegacyGenerate:
    def test_legacy_generate_works(self, client: TestClient) -> None:
        resp = client.post("/generate", json={"prompt": "def hello"})
        assert resp.status_code == 200
        body = resp.json()
        assert "generated_text" in body
        assert body["prompt"] == "def hello"
        assert body["num_tokens"] >= 0


# ══════════════════════════════════════════════════════════════════════════════
# /v1/models
# ══════════════════════════════════════════════════════════════════════════════


class TestV1Models:
    def test_v1_models_returns_list(self, client: TestClient) -> None:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert isinstance(body["data"], list)
        assert len(body["data"]) >= 1

    def test_v1_models_has_model_name(self, client: TestClient) -> None:
        resp = client.get("/v1/models")
        first = resp.json()["data"][0]
        assert first["id"] == "cola-coder"


# ══════════════════════════════════════════════════════════════════════════════
# /v1/chat/completions
# ══════════════════════════════════════════════════════════════════════════════


class TestChatCompletions:
    _payload = {
        "model": "cola-coder",
        "messages": [{"role": "user", "content": "write hello world"}],
        "stream": False,
    }

    def test_chat_completions_non_streaming(self, client: TestClient) -> None:
        resp = client.post("/v1/chat/completions", json=self._payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert len(body["choices"]) == 1
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(body["choices"][0]["message"]["content"], str)

    def test_chat_completions_has_usage(self, client: TestClient) -> None:
        resp = client.post("/v1/chat/completions", json=self._payload)
        usage = resp.json()["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_chat_completions_streaming(self, client: TestClient) -> None:
        payload = {**self._payload, "stream": True}
        resp = client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Accept": "text/event-stream"},
        )
        assert resp.status_code == 200
        # At least one SSE data line should be present
        lines = [ln for ln in resp.text.splitlines() if ln.startswith("data:")]
        assert len(lines) >= 1

    def test_chat_completions_streaming_ends_with_done(self, client: TestClient) -> None:
        payload = {**self._payload, "stream": True}
        resp = client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Accept": "text/event-stream"},
        )
        lines = [ln.strip() for ln in resp.text.splitlines() if ln.strip()]
        assert lines[-1] == "data: [DONE]"


# ══════════════════════════════════════════════════════════════════════════════
# /v1/completions
# ══════════════════════════════════════════════════════════════════════════════


class TestCompletions:
    _payload = {
        "model": "cola-coder",
        "prompt": "def hello world",
        "stream": False,
    }

    def test_completions_non_streaming(self, client: TestClient) -> None:
        resp = client.post("/v1/completions", json=self._payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "text_completion"
        assert len(body["choices"]) == 1
        assert "text" in body["choices"][0]

    def test_completions_strips_prompt(self, client: TestClient) -> None:
        resp = client.post("/v1/completions", json=self._payload)
        text = resp.json()["choices"][0]["text"]
        # The server strips the prompt from the result, so the response text
        # should not start with the original prompt.
        assert not text.startswith(self._payload["prompt"])


# ══════════════════════════════════════════════════════════════════════════════
# /v1/fim
# ══════════════════════════════════════════════════════════════════════════════


class TestFim:
    _payload = {
        "prefix": "def add(a, b):",
        "suffix": "    return result",
        "max_tokens": 64,
    }

    def test_fim_returns_infill(self, client: TestClient) -> None:
        resp = client.post("/v1/fim", json=self._payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "infill" in body
        assert "id" in body
        assert body["id"].startswith("fim-")

    def test_fim_has_usage(self, client: TestClient) -> None:
        resp = client.post("/v1/fim", json=self._payload)
        usage = resp.json()["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage


# ══════════════════════════════════════════════════════════════════════════════
# /v1/context
# ══════════════════════════════════════════════════════════════════════════════


class TestContext:
    def test_context_returns_404_without_scanner(self, client: TestClient) -> None:
        # MockGenerator has no .scanner attribute, so a 404 is expected.
        resp = client.post("/v1/context", json={"file_path": "src/main.py"})
        assert resp.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
# CORS
# ══════════════════════════════════════════════════════════════════════════════


class TestCors:
    def test_cors_enabled(self, cors_client: TestClient) -> None:
        resp = cors_client.options(
            "/v1/models",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # With CORS enabled the middleware should respond to preflight
        assert resp.status_code in (200, 204)
        assert "access-control-allow-origin" in resp.headers

    def test_cors_disabled_by_default(self, client: TestClient) -> None:
        resp = client.get(
            "/v1/models",
            headers={"Origin": "http://localhost:3000"},
        )
        # CORS header should NOT be present when enable_cors=False
        assert "access-control-allow-origin" not in resp.headers
