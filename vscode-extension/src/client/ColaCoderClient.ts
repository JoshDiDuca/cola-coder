/**
 * HTTP client for the Cola-Coder FastAPI inference server.
 *
 * Pure HTTP client — no VS Code dependencies. Uses Node's built-in fetch
 * (available in Node 18+, which is what VS Code ships).
 *
 * Endpoints mirrored from src/cola_coder/inference/server.py:
 *   GET  /health
 *   GET  /v1/models
 *   POST /v1/chat/completions  (streaming + non-streaming)
 *   POST /v1/completions       (non-streaming)
 *   POST /v1/fim
 *   POST /v1/context
 */

import type {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatDelta,
  ChatStreamChunk,
  CompletionRequest,
  CompletionResponse,
  ContextResponse,
  FimRequest,
  FimResponse,
  HealthStatus,
  ModelListResponse,
} from './types';

export class ColaCoderClient {
  private baseUrl: string;
  private connected: boolean = false;

  constructor(baseUrl: string) {
    // Normalise: strip trailing slash so all paths compose cleanly.
    this.baseUrl = baseUrl.replace(/\/+$/, '');
  }

  /** Update the server base URL at runtime (e.g. when the user changes settings). */
  setBaseUrl(url: string): void {
    this.baseUrl = url.replace(/\/+$/, '');
  }

  /**
   * Returns whether the last health check succeeded.
   * Does NOT issue a new request — call `healthCheck()` to refresh.
   */
  isConnected(): boolean {
    return this.connected;
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  private url(path: string): string {
    return `${this.baseUrl}${path}`;
  }

  private async get<T>(path: string, signal?: AbortSignal): Promise<T> {
    let response: Response;
    try {
      response = await fetch(this.url(path), {
        method: 'GET',
        headers: { Accept: 'application/json' },
        signal,
      });
    } catch (err) {
      this.connected = false;
      throw new Error(`Cola-Coder server unreachable at ${this.baseUrl}: ${String(err)}`);
    }

    if (!response.ok) {
      this.connected = false;
      const body = await response.text().catch(() => '');
      throw new Error(`Cola-Coder server error ${response.status}: ${body}`);
    }

    return response.json() as Promise<T>;
  }

  private async post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
    let response: Response;
    try {
      response = await fetch(this.url(path), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify(body),
        signal,
      });
    } catch (err) {
      this.connected = false;
      throw new Error(`Cola-Coder server unreachable at ${this.baseUrl}: ${String(err)}`);
    }

    if (!response.ok) {
      this.connected = false;
      const errorBody = await response.text().catch(() => '');
      throw new Error(`Cola-Coder server error ${response.status}: ${errorBody}`);
    }

    return response.json() as Promise<T>;
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * GET /health
   *
   * Fetches server health and GPU stats. Updates the internal connection
   * state that `isConnected()` reflects.
   */
  async healthCheck(): Promise<HealthStatus> {
    try {
      const status = await this.get<HealthStatus>('/health');
      this.connected = status.status === 'ok';
      return status;
    } catch (err) {
      this.connected = false;
      throw err;
    }
  }

  /**
   * GET /v1/models
   *
   * Lists models available on the server (OpenAI-compatible).
   */
  async getModels(): Promise<ModelListResponse> {
    return this.get<ModelListResponse>('/v1/models');
  }

  /**
   * POST /v1/chat/completions (non-streaming)
   *
   * Sends a chat completion request and waits for the full response.
   * For token-by-token delivery, use `chatStream` instead.
   */
  async chat(request: ChatCompletionRequest): Promise<ChatCompletionResponse> {
    return this.post<ChatCompletionResponse>('/v1/chat/completions', {
      ...request,
      stream: false,
    });
  }

  /**
   * POST /v1/chat/completions (streaming via SSE)
   *
   * Opens a streaming connection and calls `onChunk` for each incremental
   * `ChatDelta` received. Resolves when the stream ends (`[DONE]`).
   *
   * Pass an `AbortSignal` to cancel mid-stream (e.g. when the user presses
   * Escape or closes the editor).
   *
   * SSE wire format from the server:
   *   data: {"id":"...","choices":[{"delta":{"content":"token"},...}]}\n\n
   *   data: [DONE]\n\n
   */
  async chatStream(
    request: ChatCompletionRequest,
    onChunk: (delta: ChatDelta) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    let response: Response;
    try {
      response = await fetch(this.url('/v1/chat/completions'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify({ ...request, stream: true }),
        signal,
      });
    } catch (err) {
      this.connected = false;
      throw new Error(`Cola-Coder server unreachable at ${this.baseUrl}: ${String(err)}`);
    }

    if (!response.ok) {
      this.connected = false;
      const errorBody = await response.text().catch(() => '');
      throw new Error(`Cola-Coder server error ${response.status}: ${errorBody}`);
    }

    if (!response.body) {
      throw new Error('Cola-Coder server returned an empty response body for streaming request');
    }

    await this._consumeSseStream<ChatStreamChunk>(response.body, signal, (chunk) => {
      const delta = chunk.choices[0]?.delta;
      if (delta) {
        onChunk(delta);
      }
    });
  }

  /**
   * POST /v1/completions (non-streaming)
   *
   * OpenAI-compatible text completion endpoint.
   */
  async complete(request: CompletionRequest): Promise<CompletionResponse> {
    return this.post<CompletionResponse>('/v1/completions', {
      ...request,
      stream: false,
    });
  }

  /**
   * POST /v1/fim
   *
   * Fill-in-the-middle completion. Provide the text before (`prefix`) and
   * after (`suffix`) the cursor; the model generates the missing middle.
   */
  async fim(request: FimRequest, signal?: AbortSignal): Promise<FimResponse> {
    return this.post<FimResponse>('/v1/fim', request, signal);
  }

  /**
   * POST /v1/context
   *
   * Fetches repository context relevant to `filePath`.
   * Only available when the server was started with `--repo-root`.
   *
   * @param filePath  Absolute path to the file being edited.
   * @param maxTokens Maximum context tokens to return (default: server decides).
   */
  async getContext(filePath: string, maxTokens?: number): Promise<ContextResponse> {
    const body: Record<string, unknown> = { file_path: filePath };
    if (maxTokens !== undefined) {
      body.max_tokens = maxTokens;
    }
    return this.post<ContextResponse>('/v1/context', body);
  }

  // ── SSE stream parser ─────────────────────────────────────────────────────

  /**
   * Reads an SSE ReadableStream line-by-line, deserialises each `data: ...`
   * payload as JSON, and calls `onEvent` with the parsed value.
   * Stops when `data: [DONE]` is received or the signal is aborted.
   *
   * The server emits chunks in the form:
   *   data: <json>\n\n
   *   data: [DONE]\n\n
   *
   * We buffer incomplete lines across network chunks so no data is dropped
   * even if a `\n\n` boundary arrives split across two reads.
   */
  private async _consumeSseStream<T>(
    body: ReadableStream<Uint8Array>,
    signal: AbortSignal | undefined,
    onEvent: (event: T) => void,
  ): Promise<void> {
    const reader = body.getReader();
    const decoder = new TextDecoder('utf-8');
    // Accumulate bytes that haven't yet formed a complete line.
    let buffer = '';

    try {
      while (true) {
        if (signal?.aborted) {
          break;
        }

        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });

        // Process all complete SSE messages in the buffer.
        // Each message ends with \n\n (double newline).
        let boundary: number;
        while ((boundary = buffer.indexOf('\n\n')) !== -1) {
          const message = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);

          // An SSE message may contain multiple lines; we care about `data:`.
          for (const line of message.split('\n')) {
            if (!line.startsWith('data:')) {
              continue;
            }

            const data = line.slice('data:'.length).trim();

            if (data === '[DONE]') {
              return;
            }

            if (!data) {
              continue;
            }

            try {
              const parsed = JSON.parse(data) as T;
              onEvent(parsed);
            } catch {
              // Malformed JSON — skip the chunk rather than crashing.
              // This can happen if the server sends a keep-alive comment.
            }
          }
        }
      }
    } finally {
      // Always release the reader lock so the underlying stream can be GC'd.
      reader.releaseLock();
    }
  }
}

export default ColaCoderClient;
