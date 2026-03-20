# 70 - VS Code Extension for Inline Completions

## Overview

A VS Code extension that provides inline ghost-text completions (like GitHub Copilot) by calling the local Cola-Coder FastAPI server. Features include inline completion on cursor pause, a manual trigger keybinding, a status bar indicator showing model info, and configuration for server URL, temperature, and max tokens.

**Feature flag:** Extension itself is always optional (separate install). FastAPI server feature flag: `--enable-completion-server`.

---

## Motivation

Cola-Coder's FastAPI server already exists for generation. Connecting it to VS Code as an inline completion provider closes the loop from "model I trained" to "model I use daily." This has an outsized motivational impact—seeing your trained model suggest completions while you write TypeScript makes the training quality tangible.

It also provides an organic, continuous evaluation: if you find yourself accepting or rejecting completions, that acceptance data is a future RLHF signal.

---

## Architecture / Design

### Component Overview

```
VS Code Extension (TypeScript/Node.js)
    │
    ├── InlineCompletionProvider
    │     └── on cursor pause (debounced 500ms)
    │           └── POST /complete → FastAPI server
    │
    ├── StatusBarItem
    │     └── shows: "Cola-Coder ✓ | step-5000 | 1.2B"
    │
    └── Configuration
          ├── cola-coder.serverUrl  (default: http://localhost:8000)
          ├── cola-coder.temperature (default: 0.3)
          ├── cola-coder.maxTokens  (default: 150)
          └── cola-coder.enabled    (default: true)

FastAPI Server (Python)
    └── POST /complete
          └── { prompt, temperature, max_tokens, language }
              → { completion, tokens_generated, model_info }
```

### FastAPI `/complete` Endpoint

```python
# server/completion_endpoint.py

from fastapi import FastAPI
from pydantic import BaseModel

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.3
    language: str = "typescript"
    stop_sequences: list[str] = ["\n\n", "```"]

class CompletionResponse(BaseModel):
    completion: str
    tokens_generated: int
    model_info: dict

@app.post("/complete", response_model=CompletionResponse)
async def complete(req: CompletionRequest):
    tokens = tokenizer.encode(req.prompt)
    # Truncate prompt to fit context window
    max_prompt_tokens = model_config.max_seq_len - req.max_tokens
    tokens = tokens[-max_prompt_tokens:]

    output = model.generate(
        torch.tensor([tokens]).to(device),
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        do_sample=req.temperature > 0,
        stop_sequences=[tokenizer.encode(s) for s in req.stop_sequences],
    )

    completion_tokens = output[0][len(tokens):]
    completion_text = tokenizer.decode(completion_tokens.tolist())

    # Stop at first stop sequence
    for stop in req.stop_sequences:
        idx = completion_text.find(stop)
        if idx != -1:
            completion_text = completion_text[:idx]

    return CompletionResponse(
        completion=completion_text,
        tokens_generated=len(completion_tokens),
        model_info={
            "checkpoint": current_checkpoint_name,
            "step": current_step,
        },
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
```

---

## Implementation Steps

### Step 1: Extension Scaffold

```
cola-coder-vscode/
├── package.json
├── tsconfig.json
├── src/
│   ├── extension.ts         # activation, registration
│   ├── completionProvider.ts
│   ├── statusBar.ts
│   └── config.ts
└── .vscodeignore
```

### Step 2: `package.json`

```json
{
  "name": "cola-coder",
  "displayName": "Cola-Coder",
  "description": "Inline completions from your locally trained Cola-Coder model",
  "version": "0.1.0",
  "engines": { "vscode": "^1.85.0" },
  "categories": ["Other"],
  "activationEvents": ["onLanguage:typescript", "onLanguage:typescriptreact"],
  "main": "./out/extension.js",
  "contributes": {
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
          "default": 0.3,
          "minimum": 0,
          "maximum": 2.0
        },
        "cola-coder.maxTokens": {
          "type": "number",
          "default": 150,
          "minimum": 10,
          "maximum": 1024
        },
        "cola-coder.enabled": {
          "type": "boolean",
          "default": true
        },
        "cola-coder.debounceMs": {
          "type": "number",
          "default": 500,
          "description": "Milliseconds to wait after typing before requesting completion"
        }
      }
    },
    "commands": [
      {
        "command": "cola-coder.triggerCompletion",
        "title": "Cola-Coder: Trigger Completion"
      },
      {
        "command": "cola-coder.toggle",
        "title": "Cola-Coder: Toggle Enable/Disable"
      }
    ],
    "keybindings": [
      {
        "command": "cola-coder.triggerCompletion",
        "key": "ctrl+alt+space",
        "when": "editorTextFocus && editorLangId == typescript"
      }
    ]
  },
  "scripts": {
    "compile": "tsc -p ./",
    "watch": "tsc -watch -p ./"
  },
  "devDependencies": {
    "@types/vscode": "^1.85.0",
    "typescript": "^5.3.0"
  }
}
```

### Step 3: Completion Provider (`src/completionProvider.ts`)

```typescript
import * as vscode from 'vscode';

interface CompletionResponse {
  completion: string;
  tokens_generated: number;
  model_info: { checkpoint: string; step: number };
}

export class ColaCoderCompletionProvider
  implements vscode.InlineCompletionItemProvider {

  private debounceTimer: NodeJS.Timeout | undefined;
  private lastRequestController: AbortController | undefined;

  async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    context: vscode.InlineCompletionContext,
    token: vscode.CancellationToken,
  ): Promise<vscode.InlineCompletionList | null> {

    const config = vscode.workspace.getConfiguration('cola-coder');
    if (!config.get<boolean>('enabled', true)) {
      return null;
    }

    // Only trigger on TypeScript/TSX files
    if (!['typescript', 'typescriptreact'].includes(document.languageId)) {
      return null;
    }

    // Build prompt: lines above cursor + current line up to cursor
    const prefix = this.buildPrompt(document, position);
    if (prefix.trim().length < 10) {
      return null;  // Too short to generate useful completion
    }

    // Debounce: cancel previous request if still pending
    if (this.lastRequestController) {
      this.lastRequestController.abort();
    }

    return new Promise((resolve) => {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = setTimeout(async () => {
        if (token.isCancellationRequested) {
          resolve(null);
          return;
        }

        const controller = new AbortController();
        this.lastRequestController = controller;

        try {
          const completion = await this.fetchCompletion(prefix, controller.signal);
          if (!completion || token.isCancellationRequested) {
            resolve(null);
            return;
          }

          const item = new vscode.InlineCompletionItem(
            completion,
            new vscode.Range(position, position),
          );
          resolve(new vscode.InlineCompletionList([item]));
        } catch (err) {
          // Server unavailable or request aborted: fail silently
          resolve(null);
        }
      }, config.get<number>('debounceMs', 500));
    });
  }

  private buildPrompt(doc: vscode.TextDocument, pos: vscode.Position): string {
    const MAX_PROMPT_LINES = 80;
    const startLine = Math.max(0, pos.line - MAX_PROMPT_LINES);
    const range = new vscode.Range(
      new vscode.Position(startLine, 0),
      pos,
    );
    return doc.getText(range);
  }

  private async fetchCompletion(
    prompt: string,
    signal: AbortSignal,
  ): Promise<string | null> {
    const config = vscode.workspace.getConfiguration('cola-coder');
    const serverUrl = config.get<string>('serverUrl', 'http://localhost:8000');
    const temperature = config.get<number>('temperature', 0.3);
    const maxTokens = config.get<number>('maxTokens', 150);

    const response = await fetch(`${serverUrl}/complete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, temperature, max_tokens: maxTokens }),
      signal,
    });

    if (!response.ok) {
      return null;
    }

    const data = await response.json() as CompletionResponse;
    return data.completion;
  }
}
```

### Step 4: Status Bar (`src/statusBar.ts`)

```typescript
import * as vscode from 'vscode';

export class ColaCoderStatusBar {
  private item: vscode.StatusBarItem;
  private serverUrl: string;
  private pollInterval: NodeJS.Timeout | undefined;

  constructor() {
    this.item = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right, 100
    );
    this.item.command = 'cola-coder.toggle';
    this.serverUrl = vscode.workspace.getConfiguration('cola-coder')
      .get<string>('serverUrl', 'http://localhost:8000');
    this.show();
    this.startPolling();
  }

  private async fetchServerInfo(): Promise<void> {
    try {
      const resp = await fetch(`${this.serverUrl}/health`);
      if (resp.ok) {
        const data = await resp.json() as { status: string; checkpoint?: string; step?: number };
        const enabled = vscode.workspace.getConfiguration('cola-coder').get<boolean>('enabled');
        const icon = enabled ? '$(sparkle)' : '$(circle-slash)';
        const step = data.step ? `step-${data.step}` : '';
        this.item.text = `${icon} Cola-Coder ${step}`;
        this.item.tooltip = `Cola-Coder server: ${this.serverUrl}\n${data.checkpoint || ''}`;
        this.item.backgroundColor = undefined;
      } else {
        this.setDisconnected();
      }
    } catch {
      this.setDisconnected();
    }
  }

  private setDisconnected(): void {
    this.item.text = '$(warning) Cola-Coder (offline)';
    this.item.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
  }

  private startPolling(): void {
    this.fetchServerInfo();  // immediate check
    this.pollInterval = setInterval(() => this.fetchServerInfo(), 10_000);
  }

  show() { this.item.show(); }

  dispose() {
    clearInterval(this.pollInterval);
    this.item.dispose();
  }
}
```

### Step 5: Extension Entry Point (`src/extension.ts`)

```typescript
import * as vscode from 'vscode';
import { ColaCoderCompletionProvider } from './completionProvider';
import { ColaCoderStatusBar } from './statusBar';

export function activate(context: vscode.ExtensionContext) {
  const provider = new ColaCoderCompletionProvider();
  const statusBar = new ColaCoderStatusBar();

  // Register inline completion provider for TS/TSX
  context.subscriptions.push(
    vscode.languages.registerInlineCompletionItemProvider(
      [{ language: 'typescript' }, { language: 'typescriptreact' }],
      provider,
    ),
    statusBar,
  );

  // Manual trigger command
  context.subscriptions.push(
    vscode.commands.registerCommand('cola-coder.triggerCompletion', () => {
      vscode.commands.executeCommand('editor.action.inlineSuggest.trigger');
    }),
  );

  // Toggle enable/disable
  context.subscriptions.push(
    vscode.commands.registerCommand('cola-coder.toggle', () => {
      const config = vscode.workspace.getConfiguration('cola-coder');
      const current = config.get<boolean>('enabled', true);
      config.update('enabled', !current, vscode.ConfigurationTarget.Global);
      vscode.window.showInformationMessage(
        `Cola-Coder completions ${!current ? 'enabled' : 'disabled'}`
      );
    }),
  );
}

export function deactivate() {}
```

### Step 6: FastAPI Server Enhancement (`server/app.py`)

Add the `/complete` endpoint (shown above) and a `/health` endpoint that reports current checkpoint info:

```python
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "checkpoint": os.environ.get("COLA_CHECKPOINT_NAME", "unknown"),
        "step": int(os.environ.get("COLA_CHECKPOINT_STEP", "0")),
    }
```

---

## Key Files to Modify

- `server/app.py` - Add `/complete` and `/health` endpoints
- `server/completion_endpoint.py` - New file: completion logic
- `cola-coder-vscode/` - New directory: entire VS Code extension
- `cola-coder-vscode/src/extension.ts` - Extension entry point
- `cola-coder-vscode/src/completionProvider.ts` - Inline completion provider
- `cola-coder-vscode/src/statusBar.ts` - Status bar item
- `cola-coder-vscode/package.json` - Extension manifest

---

## Testing Strategy

1. **Server health check**: start FastAPI server, GET `/health`, assert `status == "ok"`.
2. **Completion endpoint test**: POST `/complete` with a simple TypeScript prompt, assert response has `completion` field with non-empty string.
3. **Extension activation test**: load extension in VS Code Extension Development Host, assert status bar item appears.
4. **Debounce test**: trigger provider twice in rapid succession (< 500ms apart), assert only one HTTP request is made.
5. **Cancellation test**: trigger provider, immediately cancel (document change), assert no response is delivered.
6. **Offline graceful test**: stop FastAPI server, ensure no error notifications appear (fail silently).

---

## Performance Considerations

- Debounce at 500ms prevents a request on every keystroke. Tune downward (300ms) for faster networks, upward (800ms) for slow generation.
- Each completion request requires a model forward pass + generation: ~100-500ms on RTX 3080 for 150 tokens. This is fast enough to feel responsive.
- The `AbortController` pattern ensures stale requests are cancelled when the user keeps typing. Without this, multiple in-flight requests can return out of order.
- Status bar polling every 10s is negligible load.
- The extension uses the native `fetch` API (available in VS Code's Node.js 20+ runtime). No additional HTTP client libraries needed.

---

## Dependencies

**Extension**: `@types/vscode`, TypeScript. No runtime dependencies.
**Server**: FastAPI, pydantic (already required).

---

## Estimated Complexity

**Medium-High.** The FastAPI server additions are simple. The VS Code extension requires knowledge of the VS Code Extension API, especially `InlineCompletionItemProvider` which is a relatively new API. The debounce + cancellation pattern is easy to get subtly wrong. Estimated implementation time: 3-4 days.

---

## 2026 Best Practices

- **`InlineCompletionItemProvider` over `CompletionItemProvider`**: the inline (ghost text) API is the right interface for AI completions. The older `CompletionItemProvider` shows a dropdown, which is intrusive for long AI completions.
- **Fail silently**: never show error notifications for server failures. The user is in the middle of coding; an error dialog is maximally disruptive. Log to the Output channel instead.
- **Debounce, don't throttle**: debounce ensures a request fires only after the user pauses typing. Throttle fires at regular intervals and will generate completions mid-word. Always debounce.
- **Cursor position as prompt boundary**: build the prompt from the start of the file (or N lines back) to the current cursor position. Do not include text after the cursor—the model is completing, not replacing.
- **Respect editor language ID**: only activate for `typescript` and `typescriptreact`. Activating for all languages will generate TypeScript-biased completions in Python files.
