import { useCallback, useEffect, useRef, useState } from 'react';
import type { InferenceRequest, GenStreamChunk } from '../types';
import { openGenerateStream } from '../api';

/**
 * Streams a code generation token-by-token from `/api/generate/stream` (SSE) and
 * exposes the accumulating text + status + a stop() to abort. Powers a live
 * "watch it type" inference UX.
 *
 * Wire protocol: the response body is a stream of SSE frames, each `data: {json}\n\n`
 * where the JSON is a `GenStreamChunk` ({ delta, done, error }). Deltas are appended
 * in order; a frame with `done:true` finishes the stream; a frame (or 409 body) with
 * an `error` surfaces it. While a training run is live the endpoint refuses with a
 * non-OK status (409) and a plain JSON `{error}` body instead of an SSE stream.
 */

const SSE_FRAME_SEPARATOR = '\n\n';
const SSE_DATA_PREFIX = 'data: ';

interface StreamState {
  /** Accumulated completion so far. */
  text: string;
  /** True while a stream is in flight. */
  streaming: boolean;
  error: string | null;
  /** True after a clean finish. */
  done: boolean;
}

/** Body returned by the endpoint when it refuses with a non-OK status (e.g. 409 training live). */
interface StreamErrorBody {
  error?: string;
}

export interface UseStreamingGeneration {
  state: StreamState;
  /** Resets text/error/done and begins streaming the given request. */
  start: (req: InferenceRequest) => void;
  /** Aborts the in-flight stream (no error surfaced — user-initiated). */
  stop: () => void;
  /** Clears text/error/done back to idle. */
  reset: () => void;
}

const IDLE_STATE: StreamState = {
  text: '',
  streaming: false,
  error: null,
  done: false,
};

const STREAMING_STATE: StreamState = {
  text: '',
  streaming: true,
  error: null,
  done: false,
};

/** Type guard for the AbortController's `AbortError` (DOMException) — not a real failure. */
function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

/** Extract a `{ error }` field from a parsed non-OK JSON body without trusting its shape. */
function readErrorBody(body: StreamErrorBody): string | null {
  return typeof body.error === 'string' && body.error.length > 0 ? body.error : null;
}

export function useStreamingGeneration(): UseStreamingGeneration {
  const [state, setState] = useState<StreamState>(IDLE_STATE);
  const controllerRef = useRef<AbortController | null>(null);

  const stop = useCallback((): void => {
    const controller = controllerRef.current;
    if (controller !== null) {
      controller.abort();
      controllerRef.current = null;
    }
    setState((prev): StreamState => ({ ...prev, streaming: false }));
  }, []);

  const reset = useCallback((): void => {
    const controller = controllerRef.current;
    if (controller !== null) {
      controller.abort();
      controllerRef.current = null;
    }
    setState(IDLE_STATE);
  }, []);

  const start = useCallback((req: InferenceRequest): void => {
    // Abort any prior in-flight stream before starting a fresh one.
    const previous = controllerRef.current;
    if (previous !== null) {
      previous.abort();
    }
    const controller = new AbortController();
    controllerRef.current = controller;
    setState(STREAMING_STATE);

    const run = async (): Promise<void> => {
      let res: Response;
      try {
        res = await openGenerateStream(req, controller.signal);
      } catch (err: unknown) {
        if (isAbortError(err)) {
          return;
        }
        const message = err instanceof Error ? err.message : 'stream request failed';
        setState((prev): StreamState => ({ ...prev, streaming: false, error: message }));
        return;
      }

      // Non-OK (e.g. 409 while training is live): a JSON {error} body, not an SSE stream.
      if (!res.ok) {
        let message = `${res.status}`;
        try {
          const body = (await res.json()) as StreamErrorBody;
          message = readErrorBody(body) ?? message;
        } catch {
          // Body was not JSON — keep the status-code fallback.
        }
        setState((prev): StreamState => ({ ...prev, streaming: false, error: message }));
        return;
      }

      if (res.body === null) {
        setState((prev): StreamState => ({
          ...prev,
          streaming: false,
          error: 'response body is empty',
        }));
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      const handleFrame = (frame: string): boolean => {
        const trimmed = frame.trim();
        if (trimmed.length === 0) {
          return false;
        }
        const payload = trimmed.startsWith(SSE_DATA_PREFIX)
          ? trimmed.slice(SSE_DATA_PREFIX.length)
          : trimmed;
        const chunk = JSON.parse(payload) as GenStreamChunk;
        if (chunk.delta.length > 0) {
          setState((prev): StreamState => ({ ...prev, text: prev.text + chunk.delta }));
        }
        if (chunk.error != null) {
          const errText: string = chunk.error;
          setState((prev): StreamState => ({
            ...prev,
            streaming: false,
            error: errText,
          }));
          return true;
        }
        if (chunk.done) {
          setState((prev): StreamState => ({ ...prev, streaming: false, done: true }));
          return true;
        }
        return false;
      };

      try {
        for (;;) {
          const { value, done: readerDone } = await reader.read();
          if (readerDone) {
            break;
          }
          buffer += decoder.decode(value, { stream: true });
          let sep = buffer.indexOf(SSE_FRAME_SEPARATOR);
          while (sep !== -1) {
            const frame = buffer.slice(0, sep);
            buffer = buffer.slice(sep + SSE_FRAME_SEPARATOR.length);
            if (handleFrame(frame)) {
              return;
            }
            sep = buffer.indexOf(SSE_FRAME_SEPARATOR);
          }
        }
        // Flush any trailing buffered frame (server may omit the final separator).
        const tail = buffer + decoder.decode();
        if (tail.trim().length > 0) {
          handleFrame(tail);
        }
      } catch (err: unknown) {
        if (isAbortError(err)) {
          return; // User-initiated stop — not an error.
        }
        const message = err instanceof Error ? err.message : 'stream read failed';
        setState((prev): StreamState => ({ ...prev, streaming: false, error: message }));
      } finally {
        reader.releaseLock();
        if (controllerRef.current === controller) {
          controllerRef.current = null;
        }
      }
    };

    void run();
  }, []);

  // Abort any in-flight stream on unmount.
  useEffect(() => {
    return () => {
      const controller = controllerRef.current;
      if (controller !== null) {
        controller.abort();
        controllerRef.current = null;
      }
    };
  }, []);

  return { state, start, stop, reset };
}
