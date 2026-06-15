import { useCallback, useEffect, useMemo, useState } from 'react';
import type { ChatRequest, ChatMessage, InferenceResult, ConfigFile } from '../../types';
import { isApiError } from '../../types';
import { getConfigs, chatGenerate } from '../../api';
import { formatInteger, formatDuration, formatFloat } from '../../format';

// ── Chat playground (R10) ─────────────────────────────────────────────────────
// A multi-turn conversation with a trained checkpoint. Sibling to the Inference
// playground — same controls, same training-alive guard, same error handling.
//
// CRITICAL SAFETY: chat generation contends for the GPU with the live training
// run, so the backend refuses with HTTP 409 while training is alive. We never
// even attempt to send when `trainingAlive` is true, surface a prominent amber
// banner, and still defensively handle a thrown "409" (or a resolved ApiError)
// in case the snapshot is briefly stale. Each generation reloads the model
// server-side (a few seconds) — that's expected; the busy indicator covers it.

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

// A per-reply stat shown beneath the assistant turn that produced it.
interface ReplyStat {
  tokensGenerated: number;
  elapsedS: number;
}

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

function StatLine({ stat }: { stat: ReplyStat }): JSX.Element {
  return (
    <div className="chat-stats">
      <span className="chat-stat">
        <span className="chat-stat-label">tokens</span>
        <span className="chat-stat-value mono">{formatInteger(stat.tokensGenerated)}</span>
      </span>
      <span className="chat-stat">
        <span className="chat-stat-label">elapsed</span>
        <span className="chat-stat-value mono">{formatDuration(stat.elapsedS)}</span>
      </span>
      <span className="chat-stat">
        <span className="chat-stat-label">tok/s</span>
        <span className="chat-stat-value mono">
          {stat.elapsedS > 0 ? formatFloat(stat.tokensGenerated / stat.elapsedS, 1) : '—'}
        </span>
      </span>
    </div>
  );
}

function Transcript({
  messages,
  stats,
  busy,
}: {
  messages: ChatMessage[];
  // Per-message stats keyed by message index (only assistant turns have one).
  stats: Record<number, ReplyStat>;
  busy: boolean;
}): JSX.Element {
  // The system turn is shown as a subtle preamble, not a chat bubble.
  const visible = messages.filter((m) => m.role !== 'system');
  if (visible.length === 0 && !busy) {
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
        const stat = stats[index];
        return (
          <div key={index} className={`chat-msg ${msg.role}`}>
            <div className="chat-msg-role">{msg.role === 'user' ? 'You' : 'Assistant'}</div>
            <div className="chat-msg-body mono">{msg.content}</div>
            {msg.role === 'assistant' && stat !== undefined && <StatLine stat={stat} />}
          </div>
        );
      })}
      {busy && <div className="chat-msg assistant chat-thinking muted">Thinking…</div>}
    </div>
  );
}

// ── Screen ────────────────────────────────────────────────────────────────────

export default function ChatScreen({ checkpoints, trainingAlive }: ChatScreenProps): JSX.Element {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [stats, setStats] = useState<Record<number, ReplyStat>>({});
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

  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

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

  const onSend = useCallback(async (): Promise<void> => {
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
    setBusy(true);
    setError(null);
    try {
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
      const resp = await chatGenerate(req);
      if (isApiError(resp)) {
        setError(isTrainingGuardError(resp.error) ? TRAINING_GUARD_MESSAGE : resp.error);
        return;
      }
      const result: InferenceResult = resp;
      const assistantTurn: ChatMessage = { role: 'assistant', content: result.completion };
      const assistantIndex = nextVisible.length;
      setMessages([...nextVisible, assistantTurn]);
      setStats((prev) => ({
        ...prev,
        [assistantIndex]: {
          tokensGenerated: result.tokens_generated,
          elapsedS: result.elapsed_s,
        },
      }));
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setError(isTrainingGuardError(message) ? TRAINING_GUARD_MESSAGE : message);
    } finally {
      setBusy(false);
    }
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
  ]);

  const onClear = useCallback((): void => {
    setMessages([]);
    setStats({});
    setError(null);
  }, []);

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

      {/* ── Transcript ── */}
      <Transcript messages={messages} stats={stats} busy={busy} />

      {error !== null && <div className="err chat-error">{error}</div>}

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
          disabled={trainingAlive}
        />
        <div className="chat-composer-actions">
          <button
            type="button"
            className="btn btn-primary chat-send"
            onClick={() => void onSend()}
            disabled={!canSend}
            title={trainingAlive ? TRAINING_GUARD_MESSAGE : undefined}
          >
            {busy ? 'Thinking…' : 'Send'}
          </button>
          {validationHint !== null && <span className="chat-hint muted">{validationHint}</span>}
        </div>
      </div>
    </div>
  );
}
