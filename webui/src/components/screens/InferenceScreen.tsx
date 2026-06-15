import { useCallback, useEffect, useMemo, useState } from 'react';
import type { ConfigFile, InferenceRequest, InferenceResult } from '../../types';
import { isApiError } from '../../types';
import { getConfigs, generateText } from '../../api';
import { formatInteger, formatDuration, formatFloat } from '../../format';

// ── Inference playground (R10) ───────────────────────────────────────────────
// Left: prompt + checkpoint/config + sampling controls + a Generate button.
// Right: the model completion with a tokens/elapsed stat line.
//
// CRITICAL SAFETY: generation contends for the GPU with the live training run,
// so the backend refuses with HTTP 409 while training is alive. We never even
// attempt to generate when `trainingAlive` is true, surface a prominent amber
// banner, and still defensively handle a thrown "409" (or a resolved ApiError)
// in case the snapshot is briefly stale.

interface InferenceScreenProps {
  // Checkpoint paths from the live snapshot (App passes snap.checkpoints.map(c => c.path)).
  checkpoints: string[];
  // When true, generation is disabled with a clear banner.
  trainingAlive: boolean;
}

// Sampling defaults (see Behavior spec).
const DEFAULT_TEMPERATURE = 0.8;
const DEFAULT_MAX_TOKENS = 256;
const DEFAULT_TOP_P = 0.9;
const DEFAULT_TOP_K = 50;

const TRAINING_GUARD_MESSAGE =
  'A training run is live — generation is disabled to protect the GPU. ' +
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

// ── Output pane ───────────────────────────────────────────────────────────────

function OutputPane({
  result,
  busy,
  error,
}: {
  result: InferenceResult | null;
  busy: boolean;
  error: string | null;
}): JSX.Element {
  if (busy) {
    return <div className="infer-empty muted">Generating…</div>;
  }
  if (error !== null) {
    return <div className="err infer-error">{error}</div>;
  }
  if (result === null) {
    return (
      <div className="infer-empty muted">
        No output yet — write a prompt, pick a checkpoint and config, then Generate.
      </div>
    );
  }
  return (
    <div className="infer-output">
      <div className="infer-stats">
        <span className="infer-stat">
          <span className="infer-stat-label">tokens</span>
          <span className="infer-stat-value mono">{formatInteger(result.tokens_generated)}</span>
        </span>
        <span className="infer-stat">
          <span className="infer-stat-label">elapsed</span>
          <span className="infer-stat-value mono">{formatDuration(result.elapsed_s)}</span>
        </span>
        <span className="infer-stat">
          <span className="infer-stat-label">tok/s</span>
          <span className="infer-stat-value mono">
            {result.elapsed_s > 0
              ? formatFloat(result.tokens_generated / result.elapsed_s, 1)
              : '—'}
          </span>
        </span>
      </div>
      <pre className="infer-completion mono scroll">{result.completion}</pre>
    </div>
  );
}

// ── Screen ────────────────────────────────────────────────────────────────────

export default function InferenceScreen({
  checkpoints,
  trainingAlive,
}: InferenceScreenProps): JSX.Element {
  const [prompt, setPrompt] = useState<string>('');
  const [checkpoint, setCheckpoint] = useState<string>('');
  const [config, setConfig] = useState<string>('');
  const [sampling, setSampling] = useState<SamplingState>(DEFAULT_SAMPLING);

  const [configs, setConfigs] = useState<ConfigFile[]>([]);
  const [configsError, setConfigsError] = useState<string | null>(null);

  const [result, setResult] = useState<InferenceResult | null>(null);
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

  const promptMissing = prompt.trim() === '';
  const checkpointMissing = checkpoint.trim() === '';
  const configMissing = config.trim() === '';
  const canGenerate =
    !trainingAlive && !busy && !promptMissing && !checkpointMissing && !configMissing;

  const onGenerate = useCallback(async (): Promise<void> => {
    // Hard guard: never call the API while training is live.
    if (trainingAlive) return;
    if (promptMissing || checkpointMissing || configMissing) return;

    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const req: InferenceRequest = {
        prompt: prompt,
        checkpoint: checkpoint,
        config: config,
        max_tokens: sampling.maxTokens,
        temperature: sampling.temperature,
        top_p: sampling.topP,
        top_k: sampling.topK,
      };
      const resp = await generateText(req);
      if (isApiError(resp)) {
        setError(isTrainingGuardError(resp.error) ? TRAINING_GUARD_MESSAGE : resp.error);
        return;
      }
      setResult(resp);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setError(isTrainingGuardError(message) ? TRAINING_GUARD_MESSAGE : message);
    } finally {
      setBusy(false);
    }
  }, [
    trainingAlive,
    promptMissing,
    checkpointMissing,
    configMissing,
    prompt,
    checkpoint,
    config,
    sampling,
  ]);

  const validationHint = useMemo<string | null>(() => {
    if (trainingAlive) return null;
    const missing: string[] = [];
    if (promptMissing) missing.push('a prompt');
    if (checkpointMissing) missing.push('a checkpoint');
    if (configMissing) missing.push('a config');
    if (missing.length === 0) return null;
    return `Need ${missing.join(', ')} before generating.`;
  }, [trainingAlive, promptMissing, checkpointMissing, configMissing]);

  return (
    <div className="card card-wide infer-screen">
      <div className="md-toolbar infer-head">
        <h1 className="md-detail-title">Inference Playground</h1>
        <span className="muted">Run a trained model — prompt in, completion out</span>
      </div>

      {trainingAlive && (
        <div className="infer-banner" role="status">
          {TRAINING_GUARD_MESSAGE}
        </div>
      )}

      <div className="infer-grid">
        {/* ── Controls ── */}
        <div className="infer-controls">
          <label className="infer-field">
            <span className="infer-label">Prompt</span>
            <textarea
              className="textarea infer-prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="def fibonacci(n):"
              spellCheck={false}
              rows={8}
            />
          </label>

          <label className="infer-field">
            <span className="infer-label">Checkpoint</span>
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
                placeholder="checkpoints/small/latest"
                spellCheck={false}
              />
            )}
          </label>

          <label className="infer-field">
            <span className="infer-label">Config</span>
            <select
              className="select"
              value={config}
              onChange={(e) => setConfig(e.target.value)}
            >
              <option value="">(none)</option>
              {configs.map((c) => (
                <option key={c.path} value={c.path}>
                  {c.rel}
                </option>
              ))}
            </select>
            {configsError !== null && (
              <span className="infer-hint err">Could not load configs: {configsError}</span>
            )}
          </label>

          <div className="infer-sampling">
            <label className="infer-field">
              <span className="infer-label">Temperature</span>
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
            <label className="infer-field">
              <span className="infer-label">Max tokens</span>
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
            <label className="infer-field">
              <span className="infer-label">Top-p</span>
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
            <label className="infer-field">
              <span className="infer-label">Top-k</span>
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

          <div className="infer-actions">
            <button
              type="button"
              className="btn btn-primary infer-generate"
              onClick={() => void onGenerate()}
              disabled={!canGenerate}
              title={trainingAlive ? TRAINING_GUARD_MESSAGE : undefined}
            >
              {busy ? '…generating' : 'Generate'}
            </button>
            {validationHint !== null && (
              <span className="infer-hint muted">{validationHint}</span>
            )}
          </div>
        </div>

        {/* ── Output ── */}
        <div className="infer-output-pane">
          <OutputPane result={result} busy={busy} error={error} />
        </div>
      </div>
    </div>
  );
}
