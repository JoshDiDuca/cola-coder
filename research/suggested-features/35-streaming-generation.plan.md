# Feature 35: Streaming Token Generation

## Overview

Instead of waiting for the full completion before displaying output, streaming generation
yields each token as it is produced. This dramatically improves perceived latency: users
see the first token in ~100 ms instead of waiting 5–30 seconds for a full completion.

Cola-Coder supports three streaming interfaces:
1. **CLI** — print each token with `flush=True` as it arrives
2. **HTTP (SSE)** — Server-Sent Events endpoint in FastAPI for web clients
3. **WebSocket** — bidirectional streaming for interactive sessions

Status: OPTIONAL — enable via `--feature streaming` or CLI menu toggle. SSE and
WebSocket are always available on the server once enabled; CLI streaming is the default
interactive mode.

---

## Motivation

- Code generation at 20–50 tokens/sec means a 200-token completion takes 4–10 seconds.
  With streaming, users see progress immediately and can cancel early if wrong.
- SSE is the simplest protocol for one-way server→client streaming (used by OpenAI API,
  Anthropic API). Implementing SSE makes Cola-Coder API-compatible with many clients.
- Real-time tokens/sec display helps users understand model performance.

---

## Architecture / Design

### Generator as Python Generator

The core change: `generate()` becomes a generator function using `yield`:

```python
# cola_coder/generator.py  (streaming version)

import time
import torch
from typing import Generator
from .sampling import sample_next_token


class CodeGenerator:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.kv_cache = None

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        stop_tokens: list[str] | None = None,
    ) -> Generator[dict, None, None]:
        """
        Yields dicts: {"token": str, "token_id": int, "tokens_per_sec": float,
                       "is_finished": bool, "finish_reason": str | None}
        """
        stop_ids = set()
        if stop_tokens:
            for s in stop_tokens:
                ids = self.tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    stop_ids.add(ids[0])
        stop_ids.add(self.tokenizer.eos_token_id)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.config.device)
        self._reset_kv_cache()

        start_time = time.perf_counter()
        generated = 0
        past_key_values = None

        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids if step == 0 else next_token.unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)
            past_key_values = outputs.past_key_values

            next_token_id = sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            next_token = torch.tensor([[next_token_id]], device=self.config.device)
            generated += 1

            token_str = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
            elapsed = time.perf_counter() - start_time
            tps = generated / max(elapsed, 1e-9)

            is_stop = next_token_id in stop_ids
            yield {
                "token": token_str,
                "token_id": next_token_id,
                "tokens_per_sec": tps,
                "is_finished": is_stop,
                "finish_reason": "stop_token" if is_stop else None,
                "step": step,
            }

            if is_stop:
                return

        yield {
            "token": "",
            "token_id": -1,
            "tokens_per_sec": generated / (time.perf_counter() - start_time),
            "is_finished": True,
            "finish_reason": "max_tokens",
            "step": max_new_tokens,
        }

    def generate(self, prompt: str, **kwargs) -> str:
        """Non-streaming convenience wrapper."""
        tokens = []
        for event in self.generate_stream(prompt, **kwargs):
            if not event["is_finished"]:
                tokens.append(event["token"])
        return "".join(tokens)
```

### CLI Streaming with Rich

```python
# cola_coder/cli/stream_display.py

import sys
from rich.console import Console
from rich.live import Live
from rich.text import Text

console = Console()


def stream_to_cli(
    generator,       # generator yielding event dicts from generate_stream()
    show_stats: bool = True,
) -> str:
    """Print tokens as they arrive, show live tokens/sec."""
    collected = []
    tps_display = ""

    with Live(console=console, refresh_per_second=10) as live:
        for event in generator:
            if event["is_finished"]:
                break
            token = event["token"]
            collected.append(token)
            tps = event["tokens_per_sec"]
            tps_display = f" [{tps:.1f} tok/s]" if show_stats else ""
            text = Text("".join(collected))
            if show_stats:
                text.append(tps_display, style="dim green")
            live.update(text)

    # Print final newline after live display ends
    console.print()
    if show_stats:
        last_tps = event.get("tokens_per_sec", 0)
        console.print(
            f"[dim]Generated {len(collected)} tokens at {last_tps:.1f} tok/s[/dim]"
        )
    return "".join(collected)
```

### FastAPI SSE Endpoint

```python
# cola_coder/server.py  (SSE addition)

import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

app = FastAPI()


async def token_event_generator(prompt: str, params: dict, generator: "CodeGenerator"):
    """Async generator yielding SSE-formatted events."""
    loop = asyncio.get_event_loop()

    # Run blocking generator in thread pool to avoid blocking event loop
    def run_gen():
        return list(generator.generate_stream(prompt, **params))

    # Stream tokens as they arrive using a queue
    queue: asyncio.Queue = asyncio.Queue()

    def producer():
        for event in generator.generate_stream(prompt, **params):
            loop.call_soon_threadsafe(queue.put_nowait, event)
        loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    import threading
    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    while True:
        event = await queue.get()
        if event is None:
            break
        yield {
            "event": "token",
            "data": json.dumps({
                "token": event["token"],
                "tokens_per_sec": round(event["tokens_per_sec"], 2),
                "is_finished": event["is_finished"],
                "finish_reason": event.get("finish_reason"),
            }),
        }
        if event["is_finished"]:
            break

    yield {"event": "done", "data": json.dumps({"status": "complete"})}


@app.post("/generate/stream")
async def generate_stream(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    params = {
        "max_new_tokens": body.get("max_new_tokens", 256),
        "temperature": body.get("temperature", 0.8),
        "top_k": body.get("top_k", 50),
        "top_p": body.get("top_p", 0.95),
    }
    return EventSourceResponse(
        token_event_generator(prompt, params, app.state.generator)
    )
```

### WebSocket Endpoint

```python
# cola_coder/server.py  (WebSocket addition)

from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import threading


@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "generate":
                prompt = data["prompt"]
                params = data.get("params", {})
                cancel_event = threading.Event()

                queue: asyncio.Queue = asyncio.Queue()

                def producer():
                    for event in app.state.generator.generate_stream(prompt, **params):
                        if cancel_event.is_set():
                            break
                        loop.call_soon_threadsafe(queue.put_nowait, event)
                    loop.call_soon_threadsafe(queue.put_nowait, None)

                t = threading.Thread(target=producer, daemon=True)
                t.start()

                while True:
                    event = await queue.get()
                    if event is None:
                        await websocket.send_json({"type": "done"})
                        break
                    await websocket.send_json({
                        "type": "token",
                        "token": event["token"],
                        "tps": round(event["tokens_per_sec"], 2),
                        "finished": event["is_finished"],
                    })
                    if event["is_finished"]:
                        await websocket.send_json({"type": "done"})
                        break

            elif action == "cancel":
                cancel_event.set()
                await websocket.send_json({"type": "cancelled"})

    except WebSocketDisconnect:
        pass
```

### Frontend EventSource Example

```javascript
// browser client — included as documentation in server.py docstring

const source = new EventSource('/generate/stream');
let output = '';

source.addEventListener('token', (e) => {
    const data = JSON.parse(e.data);
    output += data.token;
    document.getElementById('output').textContent = output;
    document.getElementById('tps').textContent = `${data.tokens_per_sec} tok/s`;
    if (data.is_finished) source.close();
});

source.addEventListener('done', () => source.close());
source.onerror = () => source.close();
```

---

## Implementation Steps

1. **Refactor `generator.py`**: add `generate_stream()` as generator. Keep existing
   `generate()` as a non-streaming wrapper that calls `generate_stream` internally.

2. **Add `stream_display.py`** to `cli/` for Rich live display.

3. **Install `sse-starlette`**:
   ```bash
   pip install sse-starlette
   ```

4. **Add `/generate/stream` SSE endpoint** to `server.py`.

5. **Add `/ws/generate` WebSocket endpoint** to `server.py`.

6. **Wire CLI menu option**: "Generate (streaming)" → calls `stream_to_cli`.

7. **Handle cancellation**: WebSocket `cancel` action and keyboard interrupt (Ctrl-C)
   in CLI should stop generation cleanly.

8. **Tokens/sec tracking**: compute and display in real-time. Also log to a stats file
   for benchmarking: `{"timestamp": ..., "model": ..., "tokens_per_sec": ...}`.

---

## Key Files to Modify

| File | Change |
|---|---|
| `generator.py` | Add `generate_stream()` generator method |
| `server.py` | Add SSE and WebSocket endpoints |
| `cli/menu.py` | Add streaming generation option |
| `cli/stream_display.py` | New file — Rich streaming display |
| `requirements.txt` | Add `sse-starlette` |

---

## Testing Strategy

```python
# tests/test_streaming.py

def test_generate_stream_yields_tokens():
    gen = build_test_generator()
    events = list(gen.generate_stream("def hello(", max_new_tokens=10))
    assert len(events) >= 1
    assert all("token" in e for e in events)
    assert events[-1]["is_finished"] is True

def test_generate_stream_matches_generate():
    """Streaming and non-streaming should produce identical output."""
    gen = build_test_generator()
    prompt = "def add(a, b):"

    full = gen.generate(prompt, max_new_tokens=20)
    stream_tokens = [e["token"] for e in gen.generate_stream(prompt, max_new_tokens=20)
                     if not e["is_finished"]]
    assert full == "".join(stream_tokens)

def test_generate_stream_respects_stop_token():
    gen = build_test_generator()
    events = list(gen.generate_stream("x = ", max_new_tokens=100,
                                       stop_tokens=["\n"]))
    tokens = [e["token"] for e in events if not e["is_finished"]]
    assert "\n" not in "".join(tokens)

def test_sse_endpoint(test_client):
    """Test SSE endpoint returns proper event-stream content type."""
    response = test_client.post(
        "/generate/stream",
        json={"prompt": "def f():", "max_new_tokens": 5},
        stream=True,
    )
    assert response.headers["content-type"].startswith("text/event-stream")
```

---

## Performance Considerations

- **GIL and threading**: PyTorch inference releases the GIL during CUDA ops, so the
  producer thread and asyncio event loop can run concurrently.
- **Queue backpressure**: if the client is slow, the queue grows. Add a maxsize limit
  (e.g., `asyncio.Queue(maxsize=50)`) and let producers block.
- **Flash Attention 2 is unaffected** by streaming — it's purely a generation-loop change.
- **KV-cache state**: streaming does not change KV-cache behavior; each new token still
  benefits from cached past keys/values.
- **First-token latency** is dominated by prompt prefill (encoding the input). For
  long prompts, this is the main latency bottleneck regardless of streaming.
- **Concurrent streams**: if multiple clients stream simultaneously, ensure each
  request gets its own KV-cache state (no sharing). Thread-safe generator instances.

---

## Dependencies

```
sse-starlette>=1.8.0   # SSE support for FastAPI
fastapi>=0.110.0       # base requirement (already present)
rich>=13.0.0           # CLI live display (already present)
```

---

## Estimated Complexity

| Aspect | Estimate |
|---|---|
| Refactor generator to yield | 2 hours |
| CLI streaming display | 2 hours |
| SSE endpoint | 2 hours |
| WebSocket endpoint | 3 hours |
| Cancellation handling | 2 hours |
| Tests | 2 hours |
| **Total** | **~13 hours** |

Complexity rating: **Medium** — generator refactor is straightforward; the trickiest
part is correct threading between PyTorch (blocking) and asyncio (non-blocking).

---

## 2026 Best Practices

- **Structured streaming (JSON mode)**: in 2026, structured outputs (streaming JSON
  that parses incrementally as tokens arrive) is standard. Consider supporting
  `response_format={"type": "json_object"}` alongside plain text streaming.
- **Speculative streaming**: use a small draft model to generate multiple tokens per
  step, then verify with the main model. Reduces latency per token significantly.
- **Chunk-level SSE**: instead of per-token SSE events (high overhead for fast models),
  buffer tokens into ~50 ms chunks. Reduces event count by ~10x.
- **OpenAI-compatible SSE format**: emit `data: {"choices": [{"delta": {"content": "..."}}]}`
  to be compatible with OpenAI API clients (litellm, open-webui, etc.).
- **Back-pressure via HTTP/2**: use HTTP/2 streams which have built-in flow control,
  avoiding manual queue backpressure logic.
