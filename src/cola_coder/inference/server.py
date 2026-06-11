"""FastAPI inference server with OpenAI-compatible API.

A HTTP API that serves your trained model for code generation.
Provides both the original cola-coder endpoints and OpenAI-compatible
/v1/chat/completions, /v1/completions, /v1/models endpoints, plus
cola-coder-specific /v1/fim and /v1/context endpoints.

For a TS dev: this is like an Express server with both a custom API and
an OpenAI-compatible adapter layer. FastAPI auto-generates OpenAPI/Swagger
docs at /docs.

Usage:
    python scripts/serve.py --checkpoint ./checkpoints/small/latest
    # Then: curl -X POST http://localhost:8000/generate \\
    #   -d '{"prompt": "def hello"}'
    # Or:  curl -X POST http://localhost:8000/v1/chat/completions \\
    #   -d '{"model":"cola-coder","messages":[{"role":"user","content":"def hello"}]}'
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from .text_utils import strip_prompt_prefix

logger = logging.getLogger(__name__)

# Sentinel returned by next(stream_iter, _STREAM_END) when a sync token
# generator is exhausted — lets us pull blocking GPU steps through
# asyncio.to_thread without catching StopIteration across threads.
_STREAM_END = object()

# ── Server start time (for uptime tracking) ──────────────────────────────────

_SERVER_START_TIME: float = time.time()


# ══════════════════════════════════════════════════════════════════════════════
# Original cola-coder models (backward compat)
# ══════════════════════════════════════════════════════════════════════════════


class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""

    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    min_p: float = 0.0
    repetition_penalty: float = 1.1
    stop_tokens: list[str] | None = None


class GenerateResponse(BaseModel):
    """Response body from the /generate endpoint."""

    generated_text: str
    num_tokens: int
    prompt: str


class ModelInfo(BaseModel):
    """Response body for the /info endpoint."""

    model_params: int
    vocab_size: int
    max_seq_len: int
    device: str


# ══════════════════════════════════════════════════════════════════════════════
# OpenAI-compatible models
# ══════════════════════════════════════════════════════════════════════════════


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class UsageStats(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# ── Chat completions ─────────────────────────────────────────────────────────


class StreamOptions(BaseModel):
    """OpenAI stream_options. include_usage=true emits a final usage chunk."""

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    """Request body for /v1/chat/completions (OpenAI format)."""

    model: str = "cola-coder"
    messages: list[ChatMessage]
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float = 0.8
    max_tokens: int = 256
    top_p: float = 0.9
    top_k: int = 50
    min_p: float = 0.0
    repetition_penalty: float = 1.1
    stop: list[str] | None = None
    # Best-of-N with sandboxed verification (non-streaming only):
    # generate N candidates, verify (tsc / Python syntax), return the best.
    best_of: int = 1
    verify_language: str = "auto"


class ChatChoice(BaseModel):
    """A single choice in a chat completion response."""

    index: int
    message: ChatMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    """Response body for /v1/chat/completions (OpenAI format)."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: UsageStats


# ── Text completions ─────────────────────────────────────────────────────────


class CompletionRequest(BaseModel):
    """Request body for /v1/completions (OpenAI format)."""

    model: str = "cola-coder"
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    min_p: float = 0.0
    repetition_penalty: float = 1.1
    stop: list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    # Best-of-N with sandboxed verification (non-streaming only):
    # generate N candidates, verify (tsc / Python syntax), return the best.
    best_of: int = 1
    verify_language: str = "auto"


class CompletionChoice(BaseModel):
    """A single choice in a text completion response."""

    index: int
    text: str
    finish_reason: str | None = "stop"


class CompletionResponse(BaseModel):
    """Response body for /v1/completions (OpenAI format)."""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: UsageStats


# ── Fill-in-the-middle ───────────────────────────────────────────────────────


class FimRequest(BaseModel):
    """Request body for /v1/fim (cola-coder specific)."""

    prefix: str
    suffix: str
    max_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 50
    min_p: float = 0.0
    language: str | None = None  # Metadata only — reserved for language-specific stops
    file_path: str | None = None


class FimResponse(BaseModel):
    """Response body for /v1/fim."""

    id: str
    infill: str
    finish_reason: str = "stop"
    usage: UsageStats


# ── Repo context ─────────────────────────────────────────────────────────────


class ContextRequest(BaseModel):
    """Request body for /v1/context (cola-coder specific)."""

    file_path: str
    max_tokens: int = 2048


class ContextResponse(BaseModel):
    """Response body for /v1/context."""

    context: str
    files_referenced: list[str]
    project_name: str | None = None
    frameworks: dict[str, str] = Field(default_factory=dict)


# ── Health ───────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Detailed health check response."""

    status: str = "ok"
    model: str | None = None
    params: int | None = None
    device: str | None = None
    gpu_name: str | None = None
    vram_used_gb: float | None = None
    vram_total_gb: float | None = None
    uptime_seconds: float | None = None


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _chat_id() -> str:
    """Generate a unique chat completion ID."""
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def _completion_id() -> str:
    """Generate a unique completion ID."""
    return f"cmpl-{uuid.uuid4().hex[:12]}"


def _fim_id() -> str:
    """Generate a unique FIM completion ID."""
    return f"fim-{uuid.uuid4().hex[:12]}"


def _messages_to_prompt(
    messages: list[ChatMessage],
    use_chat_template: bool = False,
) -> str:
    """Concatenate chat messages into a single prompt string.

    When use_chat_template is True, formats messages in ChatML format
    (used by instruction-tuned models). Otherwise, plain concatenation.
    """
    if use_chat_template:
        try:
            from cola_coder.tokenizer.chat_template import format_chat
            msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
            # Append an empty assistant turn to prompt the model to respond
            msg_dicts.append({"role": "assistant", "content": ""})
            prompt = format_chat(msg_dicts)
            # Remove the trailing <|im_end|> so the model generates the response
            if prompt.endswith("<|im_end|>\n"):
                prompt = prompt[: -len("<|im_end|>\n")]
            elif prompt.endswith("<|im_end|>"):
                prompt = prompt[: -len("<|im_end|>")]
            return prompt
        except ImportError:
            pass

    # Fallback: plain concatenation (base model mode)
    parts: list[str] = []
    for msg in messages:
        if msg.role == "system":
            parts.insert(0, msg.content)
        elif msg.role == "user":
            parts.append(msg.content)
        elif msg.role == "assistant":
            parts.append(msg.content)
    return "\n".join(parts)


def _get_base_generator(generator):
    """Unwrap a ContextAwareGenerator to get the underlying CodeGenerator."""
    if hasattr(generator, "generator"):
        return generator.generator
    return generator


def _parse_referenced_files(context_str: str) -> list[str]:
    """Extract file paths from <|file|>path\\n...<|/file|> blocks."""
    return re.findall(r"<\|file\|>([^\n]+)\n", context_str)


# ══════════════════════════════════════════════════════════════════════════════
# App factory
# ══════════════════════════════════════════════════════════════════════════════


def create_app(
    generator,
    config=None,
    model_name: str = "cola-coder",
    enable_thinking: bool = False,
    enable_cors: bool = False,
    enable_instruct: bool = False,
) -> FastAPI:
    """Create the FastAPI application with a loaded model.

    Args:
        generator: A CodeGenerator or ContextAwareGenerator instance.
        config: Optional model Config object for metadata.
        model_name: Name to report in OpenAI-format responses.
        enable_thinking: Whether thinking tokens are enabled.
        enable_cors: If True, add CORSMiddleware allowing all origins.

    Returns:
        FastAPI app ready to serve.
    """
    global _SERVER_START_TIME
    _SERVER_START_TIME = time.time()

    app = FastAPI(
        title="Cola-Coder API",
        description=(
            "Code generation API powered by a custom transformer model. "
            "Provides OpenAI-compatible endpoints alongside native ones."
        ),
        version="0.2.0",
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    if enable_cors:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Serialize generation requests — single GPU can only run one at a time
    _gen_lock = asyncio.Lock()

    # Get the base CodeGenerator for direct calls
    base_gen = _get_base_generator(generator)

    async def _best_of_generate(
        prompt: str,
        *,
        best_of: int,
        verify_language: str,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
    ) -> str:
        """Best-of-N generation + verification, serialized behind the GPU lock.

        Verification (tsc / syntax check) also runs inside the lock — it's
        CPU-only and takes seconds, an acceptable cost for keeping one code
        path. Returns the best candidate's full text (prompt + completion).
        """
        from .best_of_n import generate_best_of_n

        async with _gen_lock:
            result = await asyncio.to_thread(
                generate_best_of_n,
                base_gen,
                prompt,
                num_candidates=best_of,
                language=verify_language,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
            )
        return result.best.text

    # ══════════════════════════════════════════════════════════════════════
    # Original endpoints (backward compat)
    # ══════════════════════════════════════════════════════════════════════

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        """Generate code from a prompt (original cola-coder endpoint)."""
        async with _gen_lock:
            result = await asyncio.to_thread(
                base_gen.generate,
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                min_p=request.min_p,
                repetition_penalty=request.repetition_penalty,
                stop_tokens=request.stop_tokens,
            )

        prompt_tokens = len(
            base_gen.tokenizer.encode(request.prompt, add_bos=False)
        )
        total_tokens = len(
            base_gen.tokenizer.encode(result, add_bos=False)
        )
        new_tokens = total_tokens - prompt_tokens

        return GenerateResponse(
            generated_text=result,
            num_tokens=new_tokens,
            prompt=request.prompt,
        )

    @app.get("/info", response_model=ModelInfo)
    async def info() -> ModelInfo:
        """Get information about the loaded model."""
        model = base_gen.model
        return ModelInfo(
            model_params=model.num_parameters,
            vocab_size=model.config.vocab_size,
            max_seq_len=model.config.max_seq_len,
            device=str(base_gen.device),
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check with optional GPU stats."""
        model = base_gen.model
        device_str = str(base_gen.device)
        uptime = time.time() - _SERVER_START_TIME

        gpu_name = None
        vram_used_gb = None
        vram_total_gb = None

        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_used_gb = round(
                    torch.cuda.memory_allocated(0) / (1024**3), 2
                )
                vram_total_gb = round(
                    torch.cuda.get_device_properties(0).total_memory
                    / (1024**3),
                    2,
                )
        except (AttributeError, RuntimeError) as e:
            # Narrow except: a bare `except Exception: pass` here hid a
            # .total_mem typo for months, leaving vram_total_gb always null.
            logger.debug("GPU stats unavailable for /health: %s", e)

        return HealthResponse(
            status="ok",
            model=model_name,
            params=model.num_parameters,
            device=device_str,
            gpu_name=gpu_name,
            vram_used_gb=vram_used_gb,
            vram_total_gb=vram_total_gb,
            uptime_seconds=round(uptime, 1),
        )

    # ══════════════════════════════════════════════════════════════════════
    # OpenAI-compatible: /v1/chat/completions
    # ══════════════════════════════════════════════════════════════════════

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest, raw_request: Request
    ):
        """Chat completion endpoint (OpenAI-compatible).

        Supports both streaming and non-streaming responses.
        """
        prompt = _messages_to_prompt(request.messages, enable_instruct)
        prompt_token_count = len(
            base_gen.tokenizer.encode(prompt, add_bos=False)
        )

        if request.best_of > 1 and request.stream:
            raise HTTPException(
                status_code=400,
                detail="best_of > 1 requires stream=false "
                "(candidates must be verified before one can be returned)",
            )

        if request.stream:
            return StreamingResponse(
                _stream_chat(
                    prompt=prompt,
                    request=request,
                    prompt_token_count=prompt_token_count,
                    raw_request=raw_request,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming
        if request.best_of > 1:
            result = await _best_of_generate(
                prompt,
                best_of=request.best_of,
                verify_language=request.verify_language,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                min_p=request.min_p,
            )
        else:
            async with _gen_lock:
                result = await asyncio.to_thread(
                    base_gen.generate,
                    prompt=prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    min_p=request.min_p,
                    repetition_penalty=request.repetition_penalty,
                    stop_tokens=request.stop,
                )

        # Strip the prompt echo robustly (BPE decode(encode(prompt)) is not
        # always byte-identical, so a raw startswith can fail and leak the
        # whole prompt — see text_utils.strip_prompt_prefix).
        completion_text = strip_prompt_prefix(result, prompt)
        completion_tokens = len(
            base_gen.tokenizer.encode(completion_text, add_bos=False)
        )

        return ChatCompletionResponse(
            id=_chat_id(),
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant", content=completion_text
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageStats(
                prompt_tokens=prompt_token_count,
                completion_tokens=completion_tokens,
                total_tokens=prompt_token_count + completion_tokens,
            ),
        )

    async def _stream_chat(
        prompt: str,
        request: ChatCompletionRequest,
        prompt_token_count: int,
        raw_request: Request,
    ):
        """SSE generator for streaming chat completions."""
        chat_id = _chat_id()
        completion_text = ""

        async with _gen_lock:
            stream_iter = base_gen.generate_stream(
                prompt=prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                min_p=request.min_p,
                repetition_penalty=request.repetition_penalty,
                stop_tokens=request.stop,
            )

            # Each next() runs a full GPU decode step — pull it through
            # to_thread so the event loop stays responsive (a bare
            # `for chunk in stream_iter` would block /health and every
            # other request for the duration of each token).
            prompt_text = prompt
            first_chunk = True

            while True:
                chunk = await asyncio.to_thread(next, stream_iter, _STREAM_END)
                if chunk is _STREAM_END:
                    break
                # Check for client disconnect
                if await raw_request.is_disconnected():
                    break

                # Skip the prompt echo on the first chunk
                if first_chunk and chunk.startswith(prompt_text):
                    chunk = chunk[len(prompt_text):]
                    first_chunk = False
                    if not chunk:
                        continue
                else:
                    first_chunk = False

                completion_text += chunk
                data = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"

        # Final chunk with finish_reason
        final_data = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final_data)}\n\n"

        # OpenAI stream_options.include_usage: emit a final usage-only chunk
        # (choices: []). Count tokens by re-encoding the accumulated text — per
        # SSE chunk is wrong (chunks != tokens, and empty-decode tokens yield
        # nothing), so this matches the non-streaming usage exactly.
        if request.stream_options and request.stream_options.include_usage:
            completion_tokens = len(
                base_gen.tokenizer.encode(completion_text, add_bos=False)
            )
            usage_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_token_count,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_token_count + completion_tokens,
                },
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    # ══════════════════════════════════════════════════════════════════════
    # OpenAI-compatible: /v1/completions
    # ══════════════════════════════════════════════════════════════════════

    @app.post("/v1/completions")
    async def completions(
        request: CompletionRequest, raw_request: Request
    ):
        """Text completion endpoint (OpenAI-compatible).

        Supports both streaming and non-streaming responses.
        """
        prompt_token_count = len(
            base_gen.tokenizer.encode(request.prompt, add_bos=False)
        )

        if request.best_of > 1 and request.stream:
            raise HTTPException(
                status_code=400,
                detail="best_of > 1 requires stream=false "
                "(candidates must be verified before one can be returned)",
            )

        if request.stream:
            return StreamingResponse(
                _stream_completion(
                    prompt=request.prompt,
                    request=request,
                    prompt_token_count=prompt_token_count,
                    raw_request=raw_request,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming
        if request.best_of > 1:
            result = await _best_of_generate(
                request.prompt,
                best_of=request.best_of,
                verify_language=request.verify_language,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                min_p=request.min_p,
            )
        else:
            async with _gen_lock:
                result = await asyncio.to_thread(
                    base_gen.generate,
                    prompt=request.prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    min_p=request.min_p,
                    repetition_penalty=request.repetition_penalty,
                    stop_tokens=request.stop,
                )

        # Strip prompt echo robustly (see text_utils.strip_prompt_prefix).
        completion_text = strip_prompt_prefix(result, request.prompt)
        completion_tokens = len(
            base_gen.tokenizer.encode(completion_text, add_bos=False)
        )

        return CompletionResponse(
            id=_completion_id(),
            created=int(time.time()),
            model=model_name,
            choices=[
                CompletionChoice(
                    index=0,
                    text=completion_text,
                    finish_reason="stop",
                )
            ],
            usage=UsageStats(
                prompt_tokens=prompt_token_count,
                completion_tokens=completion_tokens,
                total_tokens=prompt_token_count + completion_tokens,
            ),
        )

    async def _stream_completion(
        prompt: str,
        request: CompletionRequest,
        prompt_token_count: int,
        raw_request: Request,
    ):
        """SSE generator for streaming text completions."""
        cmpl_id = _completion_id()
        completion_text = ""

        async with _gen_lock:
            stream_iter = base_gen.generate_stream(
                prompt=prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                min_p=request.min_p,
                repetition_penalty=request.repetition_penalty,
                stop_tokens=request.stop,
            )

            prompt_text = prompt
            first_chunk = True

            while True:
                chunk = await asyncio.to_thread(next, stream_iter, _STREAM_END)
                if chunk is _STREAM_END:
                    break
                if await raw_request.is_disconnected():
                    break

                # Skip prompt echo
                if first_chunk and chunk.startswith(prompt_text):
                    chunk = chunk[len(prompt_text):]
                    first_chunk = False
                    if not chunk:
                        continue
                else:
                    first_chunk = False

                completion_text += chunk
                data = {
                    "id": cmpl_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "text": chunk,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"

        final_data = {
            "id": cmpl_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final_data)}\n\n"

        # OpenAI stream_options.include_usage: accurate usage via re-encoding.
        if request.stream_options and request.stream_options.include_usage:
            completion_tokens = len(
                base_gen.tokenizer.encode(completion_text, add_bos=False)
            )
            usage_chunk = {
                "id": cmpl_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_token_count,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_token_count + completion_tokens,
                },
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    # ══════════════════════════════════════════════════════════════════════
    # Cola-coder specific: /v1/fim (Fill-in-the-Middle)
    # ══════════════════════════════════════════════════════════════════════

    @app.post("/v1/fim", response_model=FimResponse)
    async def fim(request: FimRequest, raw_request: Request):
        """Fill-in-the-middle completion (cola-coder specific).

        Uses the FIM token format to generate code that fits between
        a prefix and suffix.
        """
        tokenizer = base_gen.tokenizer

        # Build FIM-encoded prompt
        fim_ids = tokenizer.encode_fim(request.prefix, request.suffix)
        fim_prompt = tokenizer.decode(fim_ids)

        # Optionally prepend repo context if available
        if request.file_path and hasattr(generator, "scanner"):
            try:
                context = generator.scanner.get_context_for_file(
                    request.file_path, max_tokens=512
                )
                fim_prompt = context + fim_prompt
            except Exception:
                logger.debug(
                    "Could not get repo context for FIM: %s",
                    request.file_path,
                )

        # Add FIM stop tokens
        stop_tokens = ["<|fim_suffix|>", "<|eos|>"]

        prompt_token_count = len(
            tokenizer.encode(fim_prompt, add_bos=False)
        )

        async with _gen_lock:
            # Inline completions are aborted on every keystroke; requests
            # often queue behind the GPU lock and are already dead by the
            # time it's their turn. Skip generation for those instead of
            # burning a full decode on a response nobody will read.
            if await raw_request.is_disconnected():
                return FimResponse(
                    id=_fim_id(),
                    infill="",
                    finish_reason="abort",
                    usage=UsageStats(
                        prompt_tokens=prompt_token_count,
                        completion_tokens=0,
                        total_tokens=prompt_token_count,
                    ),
                )
            result = await asyncio.to_thread(
                base_gen.generate,
                prompt=fim_prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                min_p=request.min_p,
                stop_tokens=stop_tokens,
            )

        # Extract only the infilled text (after <|fim_middle|>)
        infill = result[len(fim_prompt):] if result.startswith(
            fim_prompt
        ) else result
        infill_tokens = len(
            tokenizer.encode(infill, add_bos=False)
        ) if infill else 0

        return FimResponse(
            id=_fim_id(),
            infill=infill,
            finish_reason="stop",
            usage=UsageStats(
                prompt_tokens=prompt_token_count,
                completion_tokens=infill_tokens,
                total_tokens=prompt_token_count + infill_tokens,
            ),
        )

    # ══════════════════════════════════════════════════════════════════════
    # Cola-coder specific: /v1/context (Repo context)
    # ══════════════════════════════════════════════════════════════════════

    @app.post("/v1/context", response_model=ContextResponse)
    async def context(request: ContextRequest):
        """Get repository context for a file (cola-coder specific).

        Only available when the server is running with a
        ContextAwareGenerator (i.e., with a repo root configured).
        """
        if not hasattr(generator, "scanner"):
            raise HTTPException(
                status_code=404,
                detail=(
                    "Context not available. Server must be started "
                    "with a ContextAwareGenerator (--repo-root flag)."
                ),
            )

        try:
            context_str = generator.scanner.get_context_for_file(
                request.file_path, max_tokens=request.max_tokens
            )
        except (FileNotFoundError, ValueError, KeyError) as exc:
            # Bad input from the client (nonexistent/invalid path) is a 4xx,
            # not a server error.
            raise HTTPException(
                status_code=404,
                detail=f"No context for {request.file_path!r}: {exc}",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get context: {exc}",
            )

        files_referenced = _parse_referenced_files(context_str)

        # Extract project info from the ContextAwareGenerator
        project_name = None
        frameworks: dict[str, str] = {}
        if hasattr(generator, "context") and generator.context is not None:
            ctx = generator.context
            project_name = ctx.package_info.get("name", ctx.root.name)
            frameworks = dict(ctx.framework_versions)

        return ContextResponse(
            context=context_str,
            files_referenced=files_referenced,
            project_name=project_name,
            frameworks=frameworks,
        )

    # ══════════════════════════════════════════════════════════════════════
    # OpenAI-compatible: /v1/models
    # ══════════════════════════════════════════════════════════════════════

    @app.get("/v1/models")
    async def list_models():
        """List available models (OpenAI-compatible)."""
        model = base_gen.model
        model_info = {
            "id": model_name,
            "object": "model",
            "created": int(_SERVER_START_TIME),
            "owned_by": "cola-coder",
        }

        # Add extra metadata if config is available
        if config is not None:
            model_info["metadata"] = {
                "params": model.num_parameters,
                "vocab_size": model.config.vocab_size,
                "max_seq_len": model.config.max_seq_len,
            }

        return {"object": "list", "data": [model_info]}

    return app
