import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ChatRequest, ChatMessage, ConfigFile } from '../../types';
import { getConfigs, openChatStream } from '../../api';
import { useStreamingGeneration } from '../../hooks/useStreamingGeneration';
import StreamingConsole from '../StreamingConsole';

// ── Chat playground (R10) ─────────────────────────────────────────────────────
// A multi-turn conversation with a trained checkpoint. Sibling to the Inference
// playground — same controls, same training-alive guard, same error handling.
//
// The assistant reply STREAMS in token-by-token (useStreamingGeneration →
// /api/chat/stream) so you watch the model type into an in-progress bubble; on a
// clean finish the accumulated text is committed once as a final assistant turn.
//
// CRITICAL SAFETY: chat generation contends for the GPU with the live training
// run, so the backend refuses with HTTP 409 while training is alive. We never
// even attempt to send when `trainingAlive` is true, surface a prominent amber
// banner, and still surface a 409 {error} body from the stream opener in case the
// snapshot is briefly stale. Each generation reloads the model server-side (a few
// seconds) — that's expected; the streaming bubble covers it.

interface ChatScreenProps {
  // Checkpoint paths from the live snapshot (App passes snap.checkpoints.map(c => c.path)).
  checkpoints: string[];
  // When true, sending is disabled with a clear banner.
  trainingAlive: boolean;
}

// Sampling defaults (see Behavior spec).
const DEFAULT_TEMPERATURE = 0.7;
const DEFAULT_MAX_TOKENS = 256;
const DEFAULT_TOP_P = 0.9;
const DEFAULT_TOP_K = 50;

const TRAINING_GUARD_MESSAGE =
  'A training run is live — chat is disabled to protect the GPU. ' +
  'It will be available when training finishes.';

// Typed form state: numbers stay numbers (parsed on change), never strings.
interface SamplingState {
  temperature: number;
  maxTokens: number;
  topP: number;
  topK: number;
}

const DEFAULT_SAMPLING: SamplingState = {
  temperature: DEFAULT_TEMPERATURE,
  maxTokens: DEFAULT_MAX_TOKENS,
  topP: DEFAULT_TOP_P,
  topK: DEFAULT_TOP_K,
};

/** Parse a numeric <input> value, falling back to a default when blank/NaN. */
function parseNumber(raw: string, fallback: number): number {
  const n = Number(raw);
  return raw.trim() === '' || Number.isNaN(n) ? fallback : n;
}

/** A thrown error whose message contains "409" is the training-guard refusal. */
function isTrainingGuardError(message: string): boolean {
  return message.includes('409');
}

function lastSegment(path: string): string {
  const parts = path.split(/[\\/]/).filter((p) => p !== '');
  return parts.length === 0 ? path : parts.slice(-2).join('/');
}

// ── Transcript ──────────────────────────────────────────────────────────────

// The streaming snapshot of the in-progress assistant reply, rendered as a live
// bubble at the tail of the transcript while tokens arrive (or while an error is
// being surfaced for the most recent send). Mirrors the hook's StreamState.
interface LiveReply {
  text: string;
  streaming: boolean;
  error: string | null;
  done: boolean;
  onStop: () => void;
}

function Transcript({
  messages,
  live,
}: {
  messages: ChatMessage[];
  // Non-null while a reply is streaming/erroring — rendered as a live tail bubble.
  live: LiveReply | null;
}): JSX.Element {
  // The system turn is shown as a subtle preamble, not a chat bubble.
  const visible = messages.filter((m) => m.role !== 'system');
  // Show the live bubble whenever a stream is active or has produced text/error.
  const showLive = live !== null && (live.streaming || live.text !== '' || live.error !== null);
  if (visible.length === 0 && !showLive) {
    return (
      <div className="chat-empty muted">
        No messages yet — pick a checkpoint and config, then say something below.
      </div>
    );
  }
  return (
    <div className="chat-transcript scroll">
      {messages.map((msg, index) => {
        if (msg.role === 'system') return null;
        return (
          <div key={index} className={`chat-msg ${msg.role}`}>
            <div className="chat-msg-role">{msg.role === 'user' ? 'You' : 'Assistant'}</div>
            <div className="chat-msg-body mono">{msg.content}</div>
          </div>
        );
      })}
      {showLive && live !== null && (
        <div className="chat-msg assistant chat-msg-streaming">
          <div className="chat-msg-role">Assistant</div>
          <div className="chat-msg-stream">
            <StreamingConsole
              text={live.text}
              streaming={live.streaming}
              error={live.error}
              done={live.done}
              onStop={live.onStop}
              emptyHint="Waiting for the first token…"
            />
          </div>
        </div>
      )}
    </div>
  );
}

// ── Screen ────────────────────────────────────────────────────────────────────

export default function ChatScreen({ checkpoints, trainingAlive }: ChatScreenProps): JSX.Element {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState<string>('');
  const [systemPrompt, setSystemPrompt] = useState<string>('');
  const [showSystem, setShowSystem] = useState<boolean>(false);
  const [showSettings, setShowSettings] = useState<boolean>(false);

  const [checkpoint, setCheckpoint] = useState<string>('');
  const [config, setConfig] = useState<string>('');
  const [useChatTemplate, setUseChatTemplate] = useState<boolean>(true);
  const [sampling, setSampling] = useState<SamplingState>(DEFAULT_SAMPLING);

  const [configs, setConfigs] = useState<ConfigFile[]>([]);
  const [configsError, setConfigsError] = useState<string | null>(null);

  // The assistant reply streams in token-by-token via this hook; `stream.text`
  // is the running reply, `stream.done` flips true on a clean finish, and
  // `stream.error` carries a 409/transport error (no bogus turn is committed then).
  const { state: stream, start, stop, reset } = useStreamingGeneration();
  const busy = stream.streaming;

  // Commit-once guard: each `start()` opens a new stream and resets the hook's
  // `done` to false, so we record the identity of the stream we last committed.
  // A fresh send bumps `streamIdRef`; the effect only commits once per id.
  const streamIdRef = useRef<number>(0);
  const committedStreamRef = useRef<number>(-1);

  // Load configs once. Non-fatal: a failure leaves the select empty with a hint.
  useEffect(() => {
    let active = true;
    void (async (): Promise<void> => {
      try {
        const list = await getConfigs();
        if (!active) return;
        setConfigs(list);
        setConfig((prev) => (prev === '' && list[0] ? list[0].path : prev));
      } catch (e) {
        if (active) setConfigsError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  // Default-select the first checkpoint; keep the selection valid as the live
  // snapshot updates (a checkpoint may appear/disappear between polls).
  useEffect(() => {
    setCheckpoint((prev) => {
      if (prev !== '' && checkpoints.includes(prev)) return prev;
      return checkpoints[0] ?? prev;
    });
  }, [checkpoints]);

  const hasCheckpointList = checkpoints.length > 0;

  const draftMissing = draft.trim() === '';
  const checkpointMissing = checkpoint.trim() === '';
  const configMissing = config.trim() === '';
  const canSend =
    !trainingAlive && !busy && !draftMissing && !checkpointMissing && !configMissing;

  /** Build the full history sent to the backend, with the optional system seed first. */
  const buildHistory = useCallback(
    (priorTurns: ChatMessage[], userTurn: ChatMessage): ChatMessage[] => {
      const seed: ChatMessage[] =
        systemPrompt.trim() === '' ? [] : [{ role: 'system', content: systemPrompt }];
      return [...seed, ...priorTurns, userTurn];
    },
    [systemPrompt],
  );

  const onSend = useCallback((): void => {
    // Hard guard: never call the API while training is live.
    if (trainingAlive) return;
    if (draftMissing || checkpointMissing || configMissing) return;

    const userTurn: ChatMessage = { role: 'user', content: draft };
    // Conversation turns visible to the user (excludes the system seed, which
    // we re-prepend on every request from `systemPrompt`).
    const priorTurns = messages.filter((m) => m.role !== 'system');
    const nextVisible = [...priorTurns, userTurn];

    setMessages(nextVisible);
    setDraft('');

    // Mark a new stream identity so the commit-once effect treats this reply as
    // fresh (the hook's `start` resets text/done/error for us).
    streamIdRef.current += 1;

    const req: ChatRequest = {
      messages: buildHistory(priorTurns, userTurn),
      checkpoint: checkpoint,
      config: config,
      use_chat_template: useChatTemplate,
      max_tokens: sampling.maxTokens,
      temperature: sampling.temperature,
      top_p: sampling.topP,
      top_k: sampling.topK,
    };
    start((signal) => openChatStream(req, signal)); // streams token-by-token
  }, [
    trainingAlive,
    draftMissing,
    checkpointMissing,
    configMissing,
    draft,
    messages,
    buildHistory,
    checkpoint,
    config,
    useChatTemplate,
    sampling,
    start,
  ]);

  // Commit-once on a clean finish: when the stream is done (and not still
  // streaming) with non-empty text, append it as a final assistant turn and
  // reset the hook so the next send starts clean. Guarded by stream identity so
  // a re-render after commit (or a `done` that lingers in state) can't double-add.
  useEffect(() => {
    if (!stream.done || stream.streaming) return;
    if (committedStreamRef.current === streamIdRef.current) return; // already committed
    if (stream.text === '') {
      // Clean finish with no tokens — nothing to commit, but mark handled.
      committedStreamRef.current = streamIdRef.current;
      reset();
      return;
    }
    committedStreamRef.current = streamIdRef.current;
    const assistantTurn: ChatMessage = { role: 'assistant', content: stream.text };
    setMessages((prev) => [...prev, assistantTurn]);
    reset();
  }, [stream.done, stream.streaming, stream.text, reset]);

  const onClear = useCallback((): void => {
    reset(); // abort any in-flight stream + clear streamed text/error
    streamIdRef.current += 1; // invalidate any pending commit for the aborted stream
    setMessages([]);
  }, [reset]);

  const validationHint = useMemo<string | null>(() => {
    if (trainingAlive) return null;
    const missing: string[] = [];
    if (draftMissing) missing.push('a message');
    if (checkpointMissing) missing.push('a checkpoint');
    if (configMissing) missing.push('a config');
    if (missing.length === 0) return null;
    return `Need ${missing.join(', ')} before sending.`;
  }, [trainingAlive, draftMissing, checkpointMissing, configMissing]);

  const hasConversation = messages.some((m) => m.role !== 'system');

  // Map a 409 (training live) stream error to the friendly guard message; other
  // errors pass through verbatim. No assistant turn is committed when error≠null.
  const liveError: string | null =
    stream.error === null
      ? null
      : isTrainingGuardError(stream.error)
        ? TRAINING_GUARD_MESSAGE
        : stream.error;

  // The in-progress assistant reply rendered as a live tail bubble in the
  // transcript (streaming text + caret, Stop button, or surfaced error).
  const liveReply: LiveReply = {
    text: stream.text,
    streaming: stream.streaming,
    error: liveError,
    done: stream.done,
    onStop: stop,
  };

  return (
    <div className="card card-wide chat-screen">
      <div className="md-toolbar chat-head">
        <h1 className="md-detail-title">Chat Playground</h1>
        <span className="muted">Multi-turn conversation with a trained checkpoint</span>
      </div>

      {trainingAlive && (
        <div className="chat-banner" role="status">
          {TRAINING_GUARD_MESSAGE}
        </div>
      )}

      {/* ── Settings (collapsible) ── */}
      <div className="chat-settings-bar">
        <button
          type="button"
          className="btn chat-toggle"
          onClick={() => setShowSettings((s) => !s)}
        >
          {showSettings ? '▾ Settings' : '▸ Settings'}
        </button>
        <button
          type="button"
          className="btn chat-toggle"
          onClick={() => setShowSystem((s) => !s)}
        >
          {showSystem ? '▾ System prompt' : '▸ System prompt'}
        </button>
        <button
          type="button"
          className="btn chat-clear"
          onClick={onClear}
          disabled={busy || !hasConversation}
        >
          Clear conversation
        </button>
      </div>

      {showSystem && (
        <label className="chat-field">
          <span className="chat-label">System prompt (optional)</span>
          <textarea
            className="textarea chat-system"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            placeholder="You are a helpful coding assistant."
            spellCheck={false}
            rows={3}
          />
        </label>
      )}

      {showSettings && (
        <div className="chat-settings">
          <label className="chat-field">
            <span className="chat-label">Checkpoint</span>
            {hasCheckpointList ? (
              <select
                className="select"
                value={checkpoint}
                onChange={(e) => setCheckpoint(e.target.value)}
              >
                {checkpoints.map((path) => (
                  <option key={path} value={path}>
                    {lastSegment(path)}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                className="input mono"
                value={checkpoint}
                onChange={(e) => setCheckpoint(e.target.value)}
                placeholder="checkpoints/small_sft/latest"
                spellCheck={false}
              />
            )}
          </label>

          <label className="chat-field">
            <span className="chat-label">Config</span>
            <select className="select" value={config} onChange={(e) => setConfig(e.target.value)}>
              <option value="">(none)</option>
              {configs.map((c) => (
                <option key={c.path} value={c.path}>
                  {c.rel}
                </option>
              ))}
            </select>
            {configsError !== null && (
              <span className="chat-hint err">Could not load configs: {configsError}</span>
            )}
          </label>

          <label className="chat-field chat-field-check">
            <input
              type="checkbox"
              checked={useChatTemplate}
              onChange={(e) => setUseChatTemplate(e.target.checked)}
            />
            <span className="chat-label">ChatML formatting (instruction-tuned models)</span>
          </label>

          <div className="chat-sampling">
            <label className="chat-field">
              <span className="chat-label">Temperature</span>
              <input
                type="number"
                step="any"
                min={0}
                className="input"
                value={sampling.temperature}
                onChange={(e) =>
                  setSampling((s) => ({
                    ...s,
                    temperature: parseNumber(e.target.value, DEFAULT_TEMPERATURE),
                  }))
                }
              />
            </label>
            <label className="chat-field">
              <span className="chat-label">Max tokens</span>
              <input
                type="number"
                step={1}
                min={1}
                className="input"
                value={sampling.maxTokens}
                onChange={(e) =>
                  setSampling((s) => ({
                    ...s,
                    maxTokens: parseNumber(e.target.value, DEFAULT_MAX_TOKENS),
                  }))
                }
              />
            </label>
            <label className="chat-field">
              <span className="chat-label">Top-p</span>
              <input
                type="number"
                step="any"
                min={0}
                max={1}
                className="input"
                value={sampling.topP}
                onChange={(e) =>
                  setSampling((s) => ({
                    ...s,
                    topP: parseNumber(e.target.value, DEFAULT_TOP_P),
                  }))
                }
              />
            </label>
            <label className="chat-field">
              <span className="chat-label">Top-k</span>
              <input
                type="number"
                step={1}
                min={0}
                className="input"
                value={sampling.topK}
                onChange={(e) =>
                  setSampling((s) => ({
                    ...s,
                    topK: parseNumber(e.target.value, DEFAULT_TOP_K),
                  }))
                }
              />
            </label>
          </div>
        </div>
      )}

      {/* ── Transcript (streamed assistant reply renders as a live tail bubble) ── */}
      <Transcript messages={messages} live={liveReply} />

      {/* ── Composer ── */}
      <div className="chat-composer">
        <textarea
          className="textarea chat-input"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={
            trainingAlive ? 'Chat is disabled while training is live.' : 'Type a message…'
          }
          spellCheck={false}
          rows={3}
          disabled={trainingAlive || busy}
        />
        <div className="chat-composer-actions">
          {busy ? (
            <button type="button" className="btn chat-stop" onClick={stop}>
              Stop
            </button>
          ) : (
            <button
              type="button"
              className="btn btn-primary chat-send"
              onClick={onSend}
              disabled={!canSend}
              title={trainingAlive ? TRAINING_GUARD_MESSAGE : undefined}
            >
              Send
            </button>
          )}
          {validationHint !== null && <span className="chat-hint muted">{validationHint}</span>}
        </div>
      </div>
    </div>
  );
}
