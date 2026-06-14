"""INFER-028 follow-up: top-nσ must be exposed by the server endpoints (so the
VS Code extension / API can use the 2026 sampler), and forwarded at every
generate/generate_stream call site — never accepted as a silent no-op.
"""

import inspect
import re
from pathlib import Path

from cola_coder.inference.generator import CodeGenerator

SERVER = Path(__file__).resolve().parent.parent / "src/cola_coder/inference/server.py"


def test_request_models_expose_top_n_sigma():
    text = SERVER.read_text(encoding="utf-8")
    # GenerateRequest, ChatCompletionRequest, CompletionRequest, FimRequest.
    assert text.count("top_n_sigma: float = 0.0") == 4


def test_generate_sites_forward_top_n_sigma():
    text = SERVER.read_text(encoding="utf-8")
    # A generate/generate_stream call forwards min_p then repetition_penalty. After
    # wiring, top_n_sigma sits between them — so min_p must NEVER be immediately
    # followed by repetition_penalty (that would be a forgotten/no-op site).
    leak = re.search(
        r"min_p=request\.min_p,\s*\n\s*repetition_penalty=request\.repetition_penalty",
        text,
    )
    assert leak is None, "a generate/stream call forwards min_p but not top_n_sigma"
    assert text.count("top_n_sigma=request.top_n_sigma") == 6


def test_generate_and_stream_accept_top_n_sigma():
    for method in ("generate", "generate_stream"):
        params = inspect.signature(getattr(CodeGenerator, method)).parameters
        assert "top_n_sigma" in params, f"CodeGenerator.{method} missing top_n_sigma"
        assert params["top_n_sigma"].default == 0.0
