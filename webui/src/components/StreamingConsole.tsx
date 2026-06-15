import { useEffect, useRef } from 'react';

// ── Streaming console (presentational) ───────────────────────────────────────
// The live, streaming sibling of InferenceScreen's static completion view. It
// renders accumulating generation output in a monospace block with a blinking
// caret while tokens arrive, a Stop button, and explicit error / idle states.
//
// PURE PRESENTATIONAL: no data fetching, no state. A parent owns a streaming
// hook and passes its snapshot down through props. The only hook used here is a
// ref + effect to keep the output pane auto-scrolled to the newest token.

interface StreamingConsoleProps {
  // Accumulated completion text so far.
  text: string;
  // True while tokens are arriving.
  streaming: boolean;
  // Non-null → render the error state (partial `text` is still shown above it).
  error: string | null;
  // True after a clean finish.
  done: boolean;
  // Called when the user clicks Stop (only relevant while streaming).
  onStop?: () => void;
  // Shown when idle with no text yet.
  emptyHint?: string;
}

const DEFAULT_EMPTY_HINT = 'Output will stream here.';

// Discriminated status of the header pill — keeps the className/label mapping
// exhaustive and type-safe (no string probing).
type ConsoleStatus = 'streaming' | 'error' | 'done' | 'idle';

function deriveStatus(streaming: boolean, error: string | null, done: boolean): ConsoleStatus {
  if (error !== null) return 'error';
  if (streaming) return 'streaming';
  if (done) return 'done';
  return 'idle';
}

interface StatusPill {
  label: string;
  className: string;
}

function statusPill(status: ConsoleStatus): StatusPill | null {
  switch (status) {
    case 'streaming':
      return { label: 'streaming…', className: 'tag running' };
    case 'done':
      return { label: 'done', className: 'tag done' };
    case 'error':
      return { label: 'error', className: 'tag failed' };
    case 'idle':
      return null;
    default: {
      const _exhaustive: never = status;
      return _exhaustive;
    }
  }
}

export default function StreamingConsole({
  text,
  streaming,
  error,
  done,
  onStop,
  emptyHint = DEFAULT_EMPTY_HINT,
}: StreamingConsoleProps): JSX.Element {
  const outRef = useRef<HTMLPreElement>(null);

  // Auto-scroll the output pane to the bottom as text grows so the newest
  // tokens stay in view during a stream.
  useEffect(() => {
    const pre = outRef.current;
    if (pre !== null) {
      pre.scrollTop = pre.scrollHeight;
    }
  }, [text]);

  const status = deriveStatus(streaming, error, done);
  const pill = statusPill(status);
  const hasText = text !== '';
  const showStop = streaming && onStop !== undefined;
  const showOutput = hasText || streaming;

  return (
    <div className="stream-console">
      <div className="stream-head">
        {pill !== null && <span className={pill.className}>{pill.label}</span>}
        {showStop && (
          <button type="button" className="btn stream-stop" onClick={onStop}>
            Stop
          </button>
        )}
      </div>

      {showOutput && (
        <pre ref={outRef} className="stream-out mono scroll">
          {text}
          {streaming && (
            <span className="stream-caret" aria-hidden="true">
              ▋
            </span>
          )}
        </pre>
      )}

      {error !== null && <div className="stream-error err">{error}</div>}

      {status === 'idle' && !hasText && <div className="stream-empty muted">{emptyHint}</div>}
    </div>
  );
}
