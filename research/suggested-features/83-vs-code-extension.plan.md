# Feature 83: VS Code Extension

**Status:** Proposed
**CLI Flag:** N/A (standalone extension — enabled by installing it)
**Complexity:** Medium-High

---

## Overview

A VS Code extension (`cola-coder-vscode`) that provides inline ghost-text code completions powered by the locally running Cola-Coder FastAPI server. The extension implements VS Code's `InlineCompletionProvider` API so suggestions appear as dimmed ghost text — identical UX to GitHub Copilot — and are accepted by pressing `Tab`.

Additional features: manual trigger keybinding, status bar item showing model/server state, and settings for temperature, max tokens, and server URL. Packaged as a `.vsix` file for local installation — no marketplace publishing required.

```
┌─────────────────────────────────────────────────────────────────┐
│ VS Code Editor                                                  │
│                                                                 │
│  function fibonacci(n: number) {                                │
│    if (n <= 1) return n;                                        │
│    return fibonacci(n - 1) + fibonacci(n - 2);░░░░░░░░░░░░░░░  │
│                                              ↑ ghost text       │
│  [Tab to accept]                                                │
│                                                                 │
│  Status bar: ● Cola-Coder (tiny · 50M) ▸                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Motivation

Cola-Coder's ultimate purpose is code generation. Having a VS Code integration means the model can be validated in real developer workflow, not just via CLI benchmarks. This is the most direct way to answer: "Is this model actually useful?"

For Josh specifically, as a TypeScript developer, VS Code is the primary editing environment. Being able to use Cola-Coder completions while writing TS code turns evaluation from an abstract benchmark into immediate, tactile feedback.

- Real-world test: does the model produce usable completions in actual editing context?
- Forces the FastAPI server to be robust (real concurrent requests, streaming).
- Status bar integration surfaces model health without leaving the editor.
- Ghost text UX is zero-friction — users see suggestions without requesting them.

---

## Architecture / Design

```
VS Code Extension (TypeScript)
  │
  ├── InlineCompletionProvider
  │     └── provideInlineCompletionItems(document, position, context)
  │           ├── extract prefix: last N lines before cursor
  │           ├── POST /complete → Cola-Coder FastAPI
  │           └── return InlineCompletionItem[]
  │
  ├── StatusBarItem
  │     ├── shows: model name, server state, last latency
  │     └── click → open settings or restart server
  │
  ├── Commands
  │     ├── cola-coder.triggerCompletion  (keybinding: Alt+\)
  │     ├── cola-coder.restartServer
  │     └── cola-coder.openSettings
  │
  └── Configuration
        ├── cola-coder.serverUrl        (default: http://localhost:8000)
        ├── cola-coder.temperature      (default: 0.2)
        ├── cola-coder.maxTokens        (default: 128)
        ├── cola-coder.triggerDelay     (default: 300ms)
        └── cola-coder.enabledLanguages (default: ["typescript","javascript","python"])

Cola-Coder FastAPI Server (Python)
  └── POST /complete
        ├── Request: {prompt, max_tokens, temperature, stop_sequences}
        └── Response: {completion, model, latency_ms}
```

### Debounce Strategy

Inline completions fire on every keystroke. The extension debounces requests: after the user stops typing for `triggerDelay` ms (default 300ms), the request is sent. In-flight requests are cancelled if a new keystroke arrives before the response returns.

```
keystroke → debounce timer reset → [300ms silence] → POST /complete → render ghost text
                                        ↑ new keystroke cancels previous request
```

This matches how Copilot behaves and prevents the server from being overwhelmed during fast typing.

---

## Implementation Steps

### Step 1: Extension scaffold

```bash
# In a new directory: extensions/cola-coder-vscode/
npm init -y
npm install --save-dev @types/vscode vscode typescript esbuild @vscode/vsce
```

```json
// package.json (key fields)
{
  "name": "cola-coder",
  "displayName": "Cola-Coder",
  "description": "Local AI code completions powered by Cola-Coder",
  "version": "0.1.0",
  "engines": { "vscode": "^1.85.0" },
  "activationEvents": ["onStartupFinished"],
  "main": "./dist/extension.js",
  "contributes": {
    "commands": [
      {
        "command": "cola-coder.triggerCompletion",
        "title": "Cola-Coder: Trigger Completion"
      },
      {
        "command": "cola-coder.restartServer",
        "title": "Cola-Coder: Restart Server"
      }
    ],
    "keybindings": [
      {
        "command": "cola-coder.triggerCompletion",
        "key": "alt+\\",
        "when": "editorTextFocus"
      }
    ],
    "configuration": {
      "title": "Cola-Coder",
      "properties": {
        "cola-coder.serverUrl": {
          "type": "string",
          "default": "http://localhost:8000",
          "description": "URL of the Cola-Coder FastAPI server"
        },
        "cola-coder.temperature": {
          "type": "number",
          "default": 0.2,
          "minimum": 0,
          "maximum": 2,
          "description": "Sampling temperature (lower = more deterministic)"
        },
        "cola-coder.maxTokens": {
          "type": "number",
          "default": 128,
          "minimum": 1,
          "maximum": 512,
          "description": "Maximum tokens to generate per completion"
        },
        "cola-coder.triggerDelay": {
          "type": "number",
          "default": 300,
          "description": "Debounce delay in milliseconds before requesting completion"
        },
        "cola-coder.enabledLanguages": {
          "type": "array",
          "items": { "type": "string" },
          "default": ["typescript", "javascript", "python", "typescriptreact"],
          "description": "Language IDs where completions are active"
        },
        "cola-coder.contextLines": {
          "type": "number",
          "default": 20,
          "description": "Number of lines before cursor to include as context"
        }
      }
    }
  }
}
```

### Step 2: Completion provider

```typescript
// src/completionProvider.ts
import * as vscode from 'vscode';

interface CompleteRequest {
  prompt: string;
  max_tokens: number;
  temperature: number;
  stop_sequences: string[];
}

interface CompleteResponse {
  completion: string;
  model: string;
  latency_ms: number;
}

export class ColaCoderCompletionProvider implements vscode.InlineCompletionItemProvider {
  private lastRequestController: AbortController | null = null;
  private debounceTimer: NodeJS.Timeout | null = null;

  async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    _context: vscode.InlineCompletionContext,
    token: vscode.CancellationToken
  ): Promise<vscode.InlineCompletionList | null> {
    const config = vscode.workspace.getConfiguration('cola-coder');
    const enabledLanguages: string[] = config.get('enabledLanguages', []);

    if (!enabledLanguages.includes(document.languageId)) {
      return null;
    }

    // Build prompt from context lines before cursor
    const contextLines: number = config.get('contextLines', 20);
    const startLine = Math.max(0, position.line - contextLines);
    const contextRange = new vscode.Range(startLine, 0, position.line, position.character);
    const prompt = document.getText(contextRange);

    if (prompt.trim().length < 3) return null;

    return new Promise<vscode.InlineCompletionList | null>((resolve) => {
      // Debounce
      if (this.debounceTimer) clearTimeout(this.debounceTimer);
      const delay: number = config.get('triggerDelay', 300);

      this.debounceTimer = setTimeout(async () => {
        if (token.isCancellationRequested) {
          resolve(null);
          return;
        }

        // Cancel previous in-flight request
        if (this.lastRequestController) {
          this.lastRequestController.abort();
        }
        this.lastRequestController = new AbortController();

        try {
          const completion = await this.fetchCompletion(prompt, this.lastRequestController.signal);
          if (!completion || token.isCancellationRequested) {
            resolve(null);
            return;
          }

          const item = new vscode.InlineCompletionItem(
            completion,
            new vscode.Range(position, position)
          );
          resolve({ items: [item] });
        } catch (err) {
          resolve(null);
        }
      }, delay);

      // If the cancellation token fires, resolve null
      token.onCancellationRequested(() => {
        if (this.debounceTimer) clearTimeout(this.debounceTimer);
        resolve(null);
      });
    });
  }

  private async fetchCompletion(prompt: string, signal: AbortSignal): Promise<string | null> {
    const config = vscode.workspace.getConfiguration('cola-coder');
    const serverUrl: string = config.get('serverUrl', 'http://localhost:8000');
    const maxTokens: number = config.get('maxTokens', 128);
    const temperature: number = config.get('temperature', 0.2);

    const body: CompleteRequest = {
      prompt,
      max_tokens: maxTokens,
      temperature,
      stop_sequences: ['\n\n', '```', '<|endoftext|>'],
    };

    const response = await fetch(`${serverUrl}/complete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    });

    if (!response.ok) return null;
    const data: CompleteResponse = await response.json();
    return data.completion || null;
  }
}
```

### Step 3: Status bar item

```typescript
// src/statusBar.ts
import * as vscode from 'vscode';

export class ColaCoderStatusBar {
  private item: vscode.StatusBarItem;
  private lastLatency: number | null = null;
  private serverConnected: boolean = false;

  constructor(context: vscode.ExtensionContext) {
    this.item = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100
    );
    this.item.command = 'cola-coder.openSettings';
    context.subscriptions.push(this.item);
    this.setDisconnected();
    this.item.show();
    this.startHealthCheck();
  }

  setConnected(modelName: string, latencyMs?: number): void {
    this.serverConnected = true;
    this.lastLatency = latencyMs ?? this.lastLatency;
    const latStr = this.lastLatency ? ` ${this.lastLatency}ms` : '';
    this.item.text = `$(check) Cola-Coder (${modelName})${latStr}`;
    this.item.backgroundColor = undefined;
    this.item.tooltip = `Cola-Coder connected\nModel: ${modelName}\nClick to open settings`;
  }

  setDisconnected(): void {
    this.serverConnected = false;
    this.item.text = '$(x) Cola-Coder (offline)';
    this.item.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
    this.item.tooltip = 'Cola-Coder server not reachable\nClick to open settings';
  }

  setLoading(): void {
    this.item.text = '$(sync~spin) Cola-Coder';
  }

  private startHealthCheck(): void {
    const check = async () => {
      const config = vscode.workspace.getConfiguration('cola-coder');
      const serverUrl: string = config.get('serverUrl', 'http://localhost:8000');
      try {
        const start = Date.now();
        const resp = await fetch(`${serverUrl}/health`, { signal: AbortSignal.timeout(2000) });
        if (resp.ok) {
          const data = await resp.json();
          this.setConnected(data.model ?? 'unknown', Date.now() - start);
        } else {
          this.setDisconnected();
        }
      } catch {
        this.setDisconnected();
      }
    };

    check();
    setInterval(check, 10_000); // health check every 10s
  }
}
```

### Step 4: Extension entry point

```typescript
// src/extension.ts
import * as vscode from 'vscode';
import { ColaCoderCompletionProvider } from './completionProvider';
import { ColaCoderStatusBar } from './statusBar';

export function activate(context: vscode.ExtensionContext): void {
  const statusBar = new ColaCoderStatusBar(context);
  const provider = new ColaCoderCompletionProvider();

  // Register inline completion provider for configured languages
  const selector: vscode.DocumentSelector = [
    { scheme: 'file', language: 'typescript' },
    { scheme: 'file', language: 'javascript' },
    { scheme: 'file', language: 'python' },
    { scheme: 'file', language: 'typescriptreact' },
  ];

  context.subscriptions.push(
    vscode.languages.registerInlineCompletionItemProvider(selector, provider)
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('cola-coder.triggerCompletion', async () => {
      await vscode.commands.executeCommand('editor.action.inlineSuggest.trigger');
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('cola-coder.openSettings', () => {
      vscode.commands.executeCommand('workbench.action.openSettings', 'cola-coder');
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('cola-coder.restartServer', async () => {
      // Show notification with instructions since we can't start the Python server from here
      const action = await vscode.window.showInformationMessage(
        'Start the Cola-Coder server from your terminal: cola-serve.ps1',
        'Copy Command'
      );
      if (action === 'Copy Command') {
        await vscode.env.clipboard.writeText('.venv/Scripts/python scripts/serve.py --checkpoint checkpoints/tiny/latest');
      }
    })
  );
}

export function deactivate(): void {}
```

### Step 5: FastAPI `/complete` endpoint additions

```python
# In src/cola_coder/inference/server.py — add these endpoints

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import time

class CompleteRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.2
    stop_sequences: list[str] = ["<|endoftext|>"]

class CompleteResponse(BaseModel):
    completion: str
    model: str
    latency_ms: int

class HealthResponse(BaseModel):
    status: str
    model: str
    vocab_size: int
    device: str

@app.post("/complete", response_model=CompleteResponse)
async def complete(req: CompleteRequest):
    start = time.monotonic()
    tokens = generator.generate(
        prompt=req.prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        stop_sequences=req.stop_sequences,
    )
    completion = tokenizer.decode(tokens, skip_special_tokens=True)
    latency_ms = int((time.monotonic() - start) * 1000)
    return CompleteResponse(
        completion=completion,
        model=config.model_name,
        latency_ms=latency_ms,
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model=config.model_name,
        vocab_size=config.vocab_size,
        device=str(next(model.parameters()).device),
    )
```

### Step 6: Build and package

```bash
# In extensions/cola-coder-vscode/
# Build TypeScript → JS
npx esbuild src/extension.ts --bundle --outfile=dist/extension.js \
  --external:vscode --format=cjs --platform=node --minify

# Package as .vsix
npx vsce package --no-dependencies

# Install locally
code --install-extension cola-coder-0.1.0.vsix
```

Add a `Makefile` or PowerShell build script:

```powershell
# build-extension.ps1 (in extensions/cola-coder-vscode/)
Write-Host "Building Cola-Coder VS Code Extension..."
npm install
npx esbuild src/extension.ts --bundle --outfile=dist/extension.js `
  --external:vscode --format=cjs --platform=node --minify
npx vsce package --no-dependencies
$vsix = Get-ChildItem *.vsix | Select-Object -First 1
Write-Host "Built: $($vsix.Name)"
Write-Host "Install with: code --install-extension $($vsix.Name)"
```

---

## Key Files to Modify / Create

**New (extension directory):**
- `extensions/cola-coder-vscode/src/extension.ts` — activation, command registration
- `extensions/cola-coder-vscode/src/completionProvider.ts` — `InlineCompletionProvider`
- `extensions/cola-coder-vscode/src/statusBar.ts` — status bar item + health check
- `extensions/cola-coder-vscode/package.json` — extension manifest
- `extensions/cola-coder-vscode/tsconfig.json` — TypeScript config
- `extensions/cola-coder-vscode/build-extension.ps1` — build script
- `extensions/cola-coder-vscode/README.md` — installation guide

**Modify (Python side):**
- `src/cola_coder/inference/server.py` — add `/complete` and `/health` endpoints, CORS headers

---

## Testing Strategy

**Unit tests (TypeScript):**
```typescript
// src/__tests__/completionProvider.test.ts
// Use jest + ts-jest

import { ColaCoderCompletionProvider } from '../completionProvider';

// Mock fetch
global.fetch = jest.fn().mockResolvedValue({
  ok: true,
  json: async () => ({ completion: '  return n * 2;', model: 'tiny', latency_ms: 45 }),
});

test('returns completion item on valid response', async () => {
  // ... mock vscode document/position, call provideInlineCompletionItems
});

test('returns null when language not in enabledLanguages', async () => {
  // ... mock document with languageId: 'rust'
});

test('cancels in-flight request on new keystroke', async () => {
  // ... verify AbortController.abort() called
});
```

**Integration test:**
1. Start `scripts/serve.py --checkpoint checkpoints/tiny/latest`
2. Install `.vsix` in VS Code
3. Open a `.ts` file, type a function signature, wait 300ms
4. Verify ghost text appears
5. Press Tab — verify completion is inserted
6. Stop server — verify status bar shows offline state

**API tests:**
```python
# tests/test_server.py
from fastapi.testclient import TestClient
from cola_coder.inference.server import app

def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_complete_endpoint(client: TestClient):
    resp = client.post("/complete", json={
        "prompt": "def add(a, b):\n    return",
        "max_tokens": 10,
        "temperature": 0.1,
    })
    assert resp.status_code == 200
    assert "completion" in resp.json()
    assert isinstance(resp.json()["latency_ms"], int)
```

---

## Performance Considerations

- **Debounce (300ms default)**: Prevents server saturation during fast typing. Users can lower to 100ms for faster response at the cost of more requests.
- **AbortController cancellation**: Abandoned requests are cancelled at the HTTP level — the server stops generating as soon as the connection drops (requires server-side streaming support or short timeout).
- **Context window**: Sending 20 lines of context (typically ~400-600 tokens) is sufficient for local completions and keeps inference fast. Full-file context would slow inference significantly.
- **esbuild bundling**: The extension bundle is ~50KB (no external dependencies besides vscode API). Load time is negligible.
- **Health check interval (10s)**: Balances responsiveness (user sees server come online quickly) with overhead (10 tiny requests per minute is negligible).
- **CORS headers required**: The FastAPI server must include `Access-Control-Allow-Origin: *` or the VS Code extension's fetch calls will fail. Add `CORSMiddleware` to FastAPI app.

---

## Dependencies

**Extension (TypeScript):**
| Package | Version | Purpose |
|---|---|---|
| `@types/vscode` | `^1.85.0` | VS Code API types |
| `typescript` | `^5.3` | TypeScript compiler |
| `esbuild` | `^0.20` | Fast bundler |
| `@vscode/vsce` | `^2.24` | VSIX packager |
| `jest` / `ts-jest` | dev | Unit tests |

No runtime npm dependencies (fetch is built into Node 18+, which VS Code 1.85+ uses).

**Server (Python):**
| Package | Purpose | Already present? |
|---|---|---|
| `fastapi` | API framework | Yes (feature 11) |
| `uvicorn` | ASGI server | Yes |
| `pydantic` | Request/response models | Yes |

---

## Estimated Complexity

**Medium-High.** The TypeScript extension code is moderate complexity (~400 lines), but requires familiarity with the VS Code extension API. The main tricky parts are:

1. **Debounce + cancellation**: Getting the AbortController + debounce interaction right so ghost text doesn't flicker.
2. **InlineCompletionProvider timing**: VS Code's inline completion API has quirks around when suggestions are dismissed vs. shown.
3. **Server compatibility**: The FastAPI server needs CORS headers and a clean `/complete` contract.

Estimated implementation time: 8-12 hours. Most time spent on the TypeScript side and testing in a real VS Code window.

---

## 2026 Best Practices

- **`InlineCompletionProvider` not `CompletionItemProvider`**: Ghost text (inline) is the 2026 standard UX for AI completions. The older `CompletionItemProvider` shows dropdown suggestions — avoid this for code generation.
- **`AbortController` for fetch cancellation**: Standard 2026 practice for cancellable fetch requests. No need for axios or other libraries.
- **esbuild over webpack**: esbuild is the de facto VS Code extension bundler in 2026. 10-100x faster builds.
- **`fetch` over `node-fetch`**: VS Code 1.85+ (Electron 28+) uses Node 18+ which has native `fetch`. No `node-fetch` polyfill needed.
- **Status bar health check**: A live server health indicator in the status bar is now standard practice for AI coding tools. Users expect to see connection state without hunting through settings.
- **Local-first, no cloud**: Framing the extension as "local FastAPI" avoids API key management, privacy concerns, and internet dependencies — increasingly valued in 2026.
