"""Tests for the distillation teacher backends (no network — requests mocked)."""

import pytest

from cola_coder.distillation import (
    OpenAICompatibleTeacher,
    Teacher,
    TeacherError,
    build_teacher,
)
from cola_coder.distillation import teacher as teacher_mod


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _ok_payload(content="hello"):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


@pytest.fixture
def captured(monkeypatch):
    """Capture the POST args; return a canned OK response."""
    calls = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["url"] = url
        calls["json"] = json
        calls["headers"] = headers
        calls["timeout"] = timeout
        return _FakeResponse(200, _ok_payload("def add(a, b): return a + b"))

    monkeypatch.setattr(teacher_mod.requests, "post", fake_post)
    return calls


class TestProtocolAndFactory:
    def test_openai_teacher_satisfies_protocol(self):
        t = OpenAICompatibleTeacher(model="qwen2.5-coder:7b")
        assert isinstance(t, Teacher)

    def test_build_teacher_openai_compatible(self):
        t = build_teacher({"backend": "openai_compatible", "model": "deepseek-chat"})
        assert isinstance(t, OpenAICompatibleTeacher)
        assert t.model == "deepseek-chat"

    def test_build_teacher_defaults_backend(self):
        t = build_teacher({"model": "qwen2.5-coder:7b"})
        assert isinstance(t, OpenAICompatibleTeacher)

    def test_build_teacher_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            build_teacher({"backend": "nope", "model": "x"})

    def test_hf_local_not_implemented_points_to_followup(self):
        with pytest.raises(NotImplementedError):
            build_teacher({"backend": "hf_local", "model": "Qwen/Qwen2.5-Coder-7B"})

    def test_model_required(self):
        with pytest.raises(ValueError):
            OpenAICompatibleTeacher(model="")


class TestComplete:
    def test_returns_content_and_posts_payload(self, captured):
        t = OpenAICompatibleTeacher(model="m", base_url="http://localhost:11434/v1")
        out = t.complete([{"role": "user", "content": "add"}], max_tokens=64, temperature=0.2)
        assert out == "def add(a, b): return a + b"
        assert captured["url"] == "http://localhost:11434/v1/chat/completions"
        assert captured["json"]["model"] == "m"
        assert captured["json"]["max_tokens"] == 64
        assert captured["json"]["temperature"] == 0.2
        assert captured["json"]["stream"] is False

    def test_empty_messages_raises(self, captured):
        t = OpenAICompatibleTeacher(model="m")
        with pytest.raises(ValueError):
            t.complete([])

    def test_local_endpoint_sends_no_auth_header(self, captured):
        t = OpenAICompatibleTeacher(model="m", base_url="http://localhost:11434/v1")
        t.complete([{"role": "user", "content": "hi"}])
        assert "Authorization" not in captured["headers"]

    def test_remote_endpoint_sends_bearer_key(self, captured, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret-key-123")
        t = OpenAICompatibleTeacher(
            model="deepseek-chat", base_url="https://api.deepseek.com",
            api_key_env="MY_KEY",
        )
        t.complete([{"role": "user", "content": "hi"}])
        assert captured["headers"]["Authorization"] == "Bearer secret-key-123"

    def test_remote_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        t = OpenAICompatibleTeacher(
            model="x", base_url="https://api.deepseek.com", api_key_env="MISSING_KEY",
        )
        with pytest.raises(TeacherError):
            t.complete([{"role": "user", "content": "hi"}])

    def test_non_200_raises(self, monkeypatch):
        monkeypatch.setattr(
            teacher_mod.requests, "post",
            lambda *a, **k: _FakeResponse(500, None, "server boom"),
        )
        t = OpenAICompatibleTeacher(model="m", base_url="http://localhost:11434/v1")
        with pytest.raises(TeacherError):
            t.complete([{"role": "user", "content": "hi"}])

    def test_network_error_raises(self, monkeypatch):
        def boom(*a, **k):
            raise teacher_mod.requests.RequestException("no route")
        monkeypatch.setattr(teacher_mod.requests, "post", boom)
        t = OpenAICompatibleTeacher(model="m", base_url="http://localhost:11434/v1")
        with pytest.raises(TeacherError):
            t.complete([{"role": "user", "content": "hi"}])


class TestSecretRedaction:
    def test_remote_redacts_secrets_in_prompt(self, captured):
        # An OpenAI-style key in the prompt must NOT reach a remote teacher verbatim.
        t = OpenAICompatibleTeacher(
            model="deepseek-chat", base_url="https://api.deepseek.com",
            api_key_env=None, redact_secrets=True,
        )
        leak = "key = 'sk-" + "a" * 48 + "'"
        t.complete([{"role": "user", "content": leak}])
        sent = captured["json"]["messages"][0]["content"]
        assert "sk-" + "a" * 48 not in sent
        assert "REDACTED" in sent

    def test_local_endpoint_does_not_redact(self, captured):
        # No leak surface locally — prompt passes through unchanged.
        t = OpenAICompatibleTeacher(
            model="m", base_url="http://localhost:11434/v1", redact_secrets=True,
        )
        leak = "key = 'sk-" + "b" * 48 + "'"
        t.complete([{"role": "user", "content": leak}])
        assert captured["json"]["messages"][0]["content"] == leak
