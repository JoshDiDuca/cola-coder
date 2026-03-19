"""FastAPI inference server.

A simple HTTP API that serves your trained model for code generation.
You can send POST requests with a prompt and get generated code back.

For a TS dev: this is like an Express server, but using FastAPI (Python's
equivalent). FastAPI auto-generates OpenAPI/Swagger docs at /docs.

Usage:
    python scripts/serve.py --checkpoint ./checkpoints/small/latest
    # Then: curl -X POST http://localhost:8000/generate -d '{"prompt": "def hello"}'
"""

from dataclasses import dataclass

from fastapi import FastAPI
from pydantic import BaseModel


# Request/response schemas (like TypeScript interfaces)
class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
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


def create_app(generator) -> FastAPI:
    """Create the FastAPI application with a loaded model.

    Args:
        generator: A CodeGenerator instance with a loaded model.

    Returns:
        FastAPI app ready to serve.
    """
    app = FastAPI(
        title="Cola-Coder API",
        description="Code generation API powered by your custom transformer model",
        version="0.1.0",
    )

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        """Generate code from a prompt.

        Send a prompt and receive generated code. The model will continue
        writing code from where your prompt ends.
        """
        result = generator.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            stop_tokens=request.stop_tokens,
        )

        # Count generated tokens (approximate)
        prompt_tokens = len(generator.tokenizer.encode(request.prompt, add_bos=False))
        total_tokens = len(generator.tokenizer.encode(result, add_bos=False))
        new_tokens = total_tokens - prompt_tokens

        return GenerateResponse(
            generated_text=result,
            num_tokens=new_tokens,
            prompt=request.prompt,
        )

    @app.get("/info", response_model=ModelInfo)
    async def info() -> ModelInfo:
        """Get information about the loaded model."""
        model = generator.model
        return ModelInfo(
            model_params=model.num_parameters,
            vocab_size=model.config.vocab_size,
            max_seq_len=model.config.max_seq_len,
            device=str(generator.device),
        )

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}

    return app
