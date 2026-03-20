# Feature 41: Multi-Turn Chat

## Overview

Multi-turn chat maintains a running conversation across multiple exchanges, enabling
the model to reference earlier messages ("fix that function", "add types to the class
above"). A `ChatSession` tracks message history, manages the context window budget,
and formats messages with special tokens. The CLI presents an interactive REPL loop,
while the HTTP API provides session-based endpoints.

Status: OPTIONAL — enable via `--feature multi-turn-chat` or CLI menu toggle.

---

## Motivation

- Single-turn generation is stateless. "Fix the bug in the code above" requires knowing
  what "the code above" is.
- Conversational code generation is the dominant UX pattern in 2026 (Copilot Chat,
  Claude in the IDE, Cursor). Building this capability makes Cola-Coder more realistic.
- Long sessions accumulate context that makes later generations more accurate.

---

## Architecture / Design

### Message Format

Use a simple chat template with special tokens:

```
<|system|>You are Cola-Coder, a code generation assistant.
<|user|>Write a function to reverse a string.
<|assistant|>def reverse_string(s: str) -> str:
    return s[::-1]
<|user|>Now add error handling.
<|assistant|>
```

The model generates the assistant's response starting from the final `<|assistant|>` tag.

### ChatSession Class

```python
# cola_coder/chat/session.py

from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class Message:
    role: str           # "system" | "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    token_count: int = 0


class ChatSession:
    SYSTEM_TOKEN = "<|system|>"
    USER_TOKEN = "<|user|>"
    ASSISTANT_TOKEN = "<|assistant|>"
    EOS_TOKEN = "<|endoftext|>"

    def __init__(
        self,
        session_id: str | None = None,
        system_prompt: str = "You are Cola-Coder, a helpful code generation assistant.",
        max_context_tokens: int = 2048,
        tokenizer=None,
    ):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.messages: list[Message] = []
        self.max_context_tokens = max_context_tokens
        self.tokenizer = tokenizer
        self.created_at = datetime.utcnow().isoformat()

        if system_prompt:
            self.messages.append(Message(role="system", content=system_prompt))

    def add_user_message(self, content: str) -> None:
        msg = Message(role="user", content=content)
        if self.tokenizer:
            msg.token_count = len(self.tokenizer.encode(content))
        self.messages.append(msg)

    def add_assistant_message(self, content: str) -> None:
        msg = Message(role="assistant", content=content)
        if self.tokenizer:
            msg.token_count = len(self.tokenizer.encode(content))
        self.messages.append(msg)

    def format_for_generation(self) -> str:
        """Format all messages as a single prompt string, ready for the model."""
        parts = []
        for msg in self._get_context_messages():
            if msg.role == "system":
                parts.append(f"{self.SYSTEM_TOKEN}{msg.content}\n")
            elif msg.role == "user":
                parts.append(f"{self.USER_TOKEN}{msg.content}\n")
            elif msg.role == "assistant":
                parts.append(f"{self.ASSISTANT_TOKEN}{msg.content}{self.EOS_TOKEN}\n")
        # Add opening tag for next assistant turn
        parts.append(self.ASSISTANT_TOKEN)
        return "".join(parts)

    def _get_context_messages(self) -> list[Message]:
        """
        Implement sliding window: keep system prompt + as many recent messages
        as fit in max_context_tokens. Truncates oldest non-system messages first.
        """
        if not self.tokenizer:
            return self.messages   # no truncation without tokenizer

        # Always keep system prompt
        system_msgs = [m for m in self.messages if m.role == "system"]
        other_msgs = [m for m in self.messages if m.role != "system"]

        system_tokens = sum(m.token_count for m in system_msgs)
        budget = self.max_context_tokens - system_tokens - 100  # 100 token reserve

        # Add messages from newest to oldest until budget exhausted
        selected = []
        used_tokens = 0
        for msg in reversed(other_msgs):
            cost = msg.token_count or len(self.tokenizer.encode(msg.content))
            if used_tokens + cost <= budget:
                selected.insert(0, msg)
                used_tokens += cost
            else:
                break  # drop oldest messages

        return system_msgs + selected

    def get_total_tokens(self) -> int:
        return sum(m.token_count for m in self.messages)

    def clear_history(self, keep_system: bool = True) -> None:
        if keep_system:
            self.messages = [m for m in self.messages if m.role == "system"]
        else:
            self.messages = []

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "messages": [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                for m in self.messages
            ],
        }

    @classmethod
    def from_dict(cls, data: dict, tokenizer=None) -> "ChatSession":
        session = cls(
            session_id=data["session_id"],
            system_prompt="",   # will be loaded from messages
            tokenizer=tokenizer,
        )
        session.messages = []
        for m in data["messages"]:
            session.messages.append(Message(
                role=m["role"],
                content=m["content"],
                timestamp=m.get("timestamp", ""),
            ))
        return session
```

### Context Window Management with Summarization

For very long sessions, summarize old messages instead of dropping them:

```python
# cola_coder/chat/summarizer.py

class ConversationSummarizer:
    """Summarize old conversation turns to free up context window space."""

    def __init__(self, generator):
        self.generator = generator

    def summarize(self, messages: list[Message]) -> str:
        """Compress a list of messages into a summary."""
        history_text = "\n".join(
            f"{m.role.upper()}: {m.content[:200]}" for m in messages
        )
        prompt = (
            f"Summarize this conversation history briefly:\n\n{history_text}\n\n"
            f"Summary:"
        )
        return self.generator.generate(prompt, max_new_tokens=128, temperature=0.3)

    def maybe_summarize(self, session: ChatSession, threshold: float = 0.8) -> None:
        """If session is over threshold fraction of context, summarize oldest turns."""
        used = session.get_total_tokens()
        if used < session.max_context_tokens * threshold:
            return

        # Take oldest half of non-system messages and summarize
        non_sys = [m for m in session.messages if m.role != "system"]
        to_summarize = non_sys[:len(non_sys)//2]
        summary_text = self.summarize(to_summarize)

        # Replace old messages with summary
        summary_msg = Message(
            role="system",
            content=f"[Earlier conversation summary: {summary_text}]",
        )
        # Remove summarized messages
        for msg in to_summarize:
            session.messages.remove(msg)
        # Insert summary after system prompt
        insert_idx = 1 if session.messages and session.messages[0].role == "system" else 0
        session.messages.insert(insert_idx, summary_msg)
```

### CLI Chat REPL

```python
# cola_coder/cli/chat_repl.py

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from ..chat.session import ChatSession

console = Console()


def run_chat_repl(generator, tokenizer, config) -> None:
    """Interactive multi-turn chat loop."""
    session = ChatSession(
        system_prompt=config.chat.system_prompt,
        max_context_tokens=config.chat.max_context_tokens,
        tokenizer=tokenizer,
    )

    console.print(Panel(
        f"[bold green]Cola-Coder Chat[/bold green]\n"
        f"Session: {session.session_id} | Context: {config.chat.max_context_tokens} tokens\n"
        f"Commands: /clear (reset history), /save (save session), /quit",
        expand=False
    ))

    while True:
        user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/q", "exit"):
            break
        if user_input.lower() == "/clear":
            session.clear_history(keep_system=True)
            console.print("[dim]History cleared.[/dim]")
            continue
        if user_input.lower().startswith("/save"):
            path = user_input.split(maxsplit=1)[1] if " " in user_input else f"session_{session.session_id}.json"
            import json
            with open(path, "w") as f:
                json.dump(session.to_dict(), f, indent=2)
            console.print(f"[dim]Saved to {path}[/dim]")
            continue

        session.add_user_message(user_input)
        prompt = session.format_for_generation()

        # Generate response
        response = generator.generate(
            prompt,
            max_new_tokens=config.chat.max_response_tokens,
            temperature=config.chat.temperature,
            stop_tokens=["\n<|user|>", "\n<|system|>"],
        )

        session.add_assistant_message(response)

        tokens_used = session.get_total_tokens()
        console.print(
            Panel(
                Markdown(response),
                title="[bold green]Cola-Coder[/bold green]",
                subtitle=f"[dim]{tokens_used}/{config.chat.max_context_tokens} tokens[/dim]",
            )
        )
```

### HTTP Session-Based API

```python
# cola_coder/server.py  (chat endpoints)

import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

chat_sessions: dict[str, ChatSession] = {}


class ChatMessage(BaseModel):
    message: str
    max_new_tokens: int = 256
    temperature: float = 0.8


class NewSessionRequest(BaseModel):
    system_prompt: str = "You are Cola-Coder, a helpful code generation assistant."
    max_context_tokens: int = 2048


@app.post("/chat/sessions")
def create_session(req: NewSessionRequest):
    session = ChatSession(
        system_prompt=req.system_prompt,
        max_context_tokens=req.max_context_tokens,
        tokenizer=app.state.tokenizer,
    )
    chat_sessions[session.session_id] = session
    return {"session_id": session.session_id}


@app.post("/chat/sessions/{session_id}/messages")
def send_message(session_id: str, req: ChatMessage):
    session = chat_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found")

    session.add_user_message(req.message)
    prompt = session.format_for_generation()

    response = app.state.generator.generate(
        prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        stop_tokens=["\n<|user|>", "\n<|system|>"],
    )
    session.add_assistant_message(response)
    return {
        "response": response,
        "session_id": session_id,
        "tokens_used": session.get_total_tokens(),
    }


@app.get("/chat/sessions/{session_id}")
def get_session(session_id: str):
    session = chat_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found")
    return session.to_dict()


@app.delete("/chat/sessions/{session_id}")
def delete_session(session_id: str):
    chat_sessions.pop(session_id, None)
    return {"status": "deleted"}
```

---

## Implementation Steps

1. **Create `cola_coder/chat/` package**: `__init__.py`, `session.py`, `summarizer.py`.

2. **Add chat special tokens** to tokenizer vocabulary during training:
   `<|system|>`, `<|user|>`, `<|assistant|>`. Or use existing separator tokens.

3. **Add `ChatConfig` to `config.py`**:
   ```python
   @dataclass
   class ChatConfig:
       enabled: bool = False
       system_prompt: str = "You are Cola-Coder, a helpful code generation assistant."
       max_context_tokens: int = 2048
       max_response_tokens: int = 512
       temperature: float = 0.8
       summarize_on_overflow: bool = False
   ```

4. **Implement CLI REPL** in `cli/chat_repl.py`.

5. **Add HTTP session endpoints** to `server.py`.

6. **Session persistence**: save/load sessions as JSON to `~/.cola_coder/sessions/`.

7. **Add "Chat mode" option** to CLI menu.

8. **Handle stop tokens**: generation must stop at `<|user|>` or `<|system|>` to avoid
   the model generating the next user message itself.

---

## Key Files to Modify

| File | Change |
|---|---|
| `server.py` | Add chat session endpoints |
| `cli/menu.py` | Add "Chat mode" option |
| `config.py` | Add `ChatConfig` |
| `generator.py` | Ensure stop_tokens work correctly for chat delimiters |
| `cola_coder/chat/` | New package |
| `cli/chat_repl.py` | New file |

---

## Testing Strategy

```python
# tests/test_chat.py

def test_session_formats_messages_correctly():
    session = ChatSession(system_prompt="You are helpful.")
    session.add_user_message("Hello")
    session.add_assistant_message("Hi there!")
    session.add_user_message("Write a function")
    prompt = session.format_for_generation()
    assert "<|system|>" in prompt
    assert "<|user|>Hello" in prompt
    assert "<|assistant|>Hi there!" in prompt
    assert prompt.endswith("<|assistant|>")

def test_session_sliding_window_drops_oldest():
    tokenizer = build_test_tokenizer()
    session = ChatSession(max_context_tokens=100, tokenizer=tokenizer)
    session.add_user_message("message one")
    session.add_assistant_message("response one")
    session.add_user_message("message two")
    session.add_assistant_message("response two")
    session.add_user_message("message three")
    context = session._get_context_messages()
    contents = [m.content for m in context]
    # Oldest messages should be dropped
    assert "message one" not in contents or len(contents) <= 4

def test_session_to_dict_roundtrip():
    session = ChatSession(session_id="test123", system_prompt="Be helpful.")
    session.add_user_message("Hello")
    d = session.to_dict()
    loaded = ChatSession.from_dict(d)
    assert loaded.session_id == "test123"
    assert len(loaded.messages) == len(session.messages)

def test_session_clear_keeps_system():
    session = ChatSession(system_prompt="Be helpful.")
    session.add_user_message("Hi")
    session.add_assistant_message("Hello!")
    session.clear_history(keep_system=True)
    assert len(session.messages) == 1
    assert session.messages[0].role == "system"
```

---

## Performance Considerations

- **Context length grows per turn**: each assistant response adds tokens to the context.
  Without sliding window, a 10-turn conversation easily hits 2K+ tokens.
- **Reformatting the prompt every turn**: for KV-cache efficiency, avoid re-encoding the
  full history every turn. Instead, cache the KV state after each assistant response
  and only encode the new user message.
- **Session storage**: in-memory sessions are lost on server restart. Use Redis or a
  simple SQLite file for persistence if needed.
- **Concurrent sessions**: each session has independent message history. Server-side
  sessions are just dicts — cheap. Generation is the bottleneck, not session management.

---

## Dependencies

```
rich>=13.0.0       # CLI display (already required)
fastapi>=0.110.0   # HTTP endpoints (already required)
pydantic>=2.0.0    # Request validation (already required)
```

No new dependencies for core functionality.

---

## Estimated Complexity

| Aspect | Estimate |
|---|---|
| ChatSession class | 3 hours |
| Sliding window + summarization | 3 hours |
| CLI REPL | 3 hours |
| HTTP session API | 2 hours |
| Session persistence | 1 hour |
| Special token handling | 2 hours |
| Tests | 2 hours |
| **Total** | **~16 hours** |

Complexity rating: **Medium** — well-understood pattern; main challenges are context
window management and ensuring stop tokens work cleanly.

---

## 2026 Best Practices

- **Chat templates (Jinja2)**: Hugging Face `tokenizer.apply_chat_template()` uses Jinja2
  templates stored in `tokenizer_config.json`. Using this standard makes Cola-Coder
  compatible with the broader HF ecosystem.
- **Tool use / function calling**: in 2026, chat models are expected to support tool
  calls. Structure the system prompt to describe available tools; parse `<|tool_call|>`
  tokens from output to dispatch function execution.
- **Persistent KV-cache (prefix caching)**: cache the KV state for the system prompt
  so it doesn't need re-encoding on every turn. Called "prompt caching" by Anthropic.
  vLLM implements this as "prefix caching" with a radix tree.
- **Structured assistant responses**: for code-specific chat, always emit code in
  fenced code blocks (```python ... ```) for reliable parsing. Consider enforcing this
  via the system prompt or output constraints.
- **Retrieval-augmented context**: for long projects, include relevant code snippets
  from the codebase (retrieved via embedding similarity) in the context rather than
  full file contents. Reduces tokens while maintaining relevance.
