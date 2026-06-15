import { useCallback, useEffect, useMemo, useState } from 'react';
import type { ConfigFile, FimRequest, InferenceResult } from '../../types';
import { isApiError } from '../../types';
import { getConfigs, fimGenerate } from '../../api';
import { formatInteger, formatDuration, formatFloat } from '../../format';

// ── Fill-In-the-Middle playground (R10) ─────────────────────────────────────
// The core code-completion primitive: a PREFIX (code before the hole) and a
// SUFFIX (code after the hole); the model fills the middle. We show the raw
// infill AND a stitched preview (prefix + [infill] + suffix) so the result is
// visible in context.
//
// CRITICAL SAFETY: FIM generation contends for the GPU with the live training
// run, so the backend refuses with HTTP 409 while training is alive. We never
// even attempt to generate when `trainingAlive` is true, surface a prominent
// amber banner, and still defensively handle a thrown "409" (or a resolved
// ApiError) in case the snapshot is briefly stale.
//
// A second expected case: a checkpoint whose tokenizer was trained WITHOUT the
// `<|fim_*|>` tokens returns a 400. That is common and benign — we surface it
// as a helpful note, not a scary error.

interface FimScreenProps {
  // Checkpoint paths from the live snapshot (App passes snap.checkpoints.map(c => c.path)).
  checkpoints: string[];
  // When true, FIM generation is disabled with a clear banner.
  trainingAlive: boolean;
}

// Sampling defaults — FIM wants low temperature for deterministic infills.
const DEFAULT_TEMPERATURE = 0.2;
const DEFAULT_MAX_TOKENS = 128;
const DEFAULT_TOP_P = 0.9;
const DEFAULT_TOP_K = 50;

const TRAINING_GUARD_MESSAGE =
  'A training run is live — FIM generation is disabled to protect the GPU. ' +
  'It will be available when training finishes.';

const NO_FIM_TOKENS_MESSAGE =
  "This checkpoint's tokenizer was not trained with FIM support " +
  '(no <|fim_*|> tokens). Pick a FIM-capable checkpoint, or retrain the ' +
  'tokenizer with the fill-in-the-middle tokens to use this playground.';

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

/** A thrown error (or ApiError) whose message contains "409" is the training-guard refusal. */
function isTrainingGuardMessage(message: string): boolean {
  return message.includes('409');
}

/** A 400 mentioning missing fim tokens — the tokenizer lacks FIM support. */
function isMissingFimTokensMessage(message: string): boolean {
  const lower = message.toLowerCase();
  return lower.includes('fim_') || (lower.includes('fim') && lower.includes('token'));
}

/** Map any error string to the friendliest message we can show. */
function friendlyError(message: string): string {
  if (isTrainingGuardMessage(message)) return TRAINING_GUARD_MESSAGE;
  if (isMissingFimTokensMessage(message)) return NO_FIM_TOKENS_MESSAGE;
  return message;
}

function lastSegment(path: string): string {
  const parts = path.split(/[\\/]/).filter((p) => p !== '');
  return parts.length === 0 ? path : parts.slice(-2).join('/');
}

// ── Output pane ─────────────────────────────────────────────────────────────

function OutputPane({
  result,
  prefix,
  suffix,
  busy,
  error,
}: {
  result: InferenceResult | null;
  prefix: string;
  suffix: string;
  busy: boolean;
  error: string | null;
}): JSX.Element {
  if (busy) {
    return <div className="fim-empty muted">Filling…</div>;
  }
  if (error !== null) {
    return <div className="err fim-error">{error}</div>;
  }
  if (result === null) {
    return (
      <div className="fim-empty muted">
        No infill yet — write a prefix (and optionally a suffix), pick a checkpoint and
        config, then Complete.
      </div>
    );
  }
  return (
    <div className="fim-output">
      <div className="fim-stats">
        <span className="fim-stat">
          <span className="fim-stat-label">tokens</span>
          <span className="fim-stat-value mono">{formatInteger(result.tokens_generated)}</span>
        </span>
        <span className="fim-stat">
          <span className="fim-stat-label">elapsed</span>
          <span className="fim-stat-value mono">{formatDuration(result.elapsed_s)}</span>
        </span>
        <span className="fim-stat">
          <span className="fim-stat-label">tok/s</span>
          <span className="fim-stat-value mono">
            {result.elapsed_s > 0
              ? formatFloat(result.tokens_generated / result.elapsed_s, 1)
              : '—'}
          </span>
        </span>
      </div>

      <div className="fim-section-label muted">infill</div>
      <pre className="fim-infill-block mono scroll">{result.completion}</pre>

      <div className="fim-section-label muted">stitched preview</div>
      <pre className="fim-stitched mono scroll">
        <span className="fim-context">{prefix}</span>
        <span className="fim-infill">{result.completion}</span>
        <span className="fim-context">{suffix}</span>
      </pre>
    </div>
  );
}

// ── Screen ────────────────────────────────────────────────────────────────────

export default function FimScreen({ checkpoints, trainingAlive }: FimScreenProps): JSX.Element {
  const [prefix, setPrefix] = useState<string>('');
  const [suffix, setSuffix] = useState<string>('');
  const [checkpoint, setCheckpoint] = useState<string>('');
  const [config, setConfig] = useState<string>('');
  const [sampling, setSampling] = useState<SamplingState>(DEFAULT_SAMPLING);

  const [configs, setConfigs] = useState<ConfigFile[]>([]);
  const [configsError, setConfigsError] = useState<string | null>(null);

  // Capture the prefix/suffix that produced the current result so the stitched
  // preview stays consistent even if the user keeps editing the textareas.
  const [resultPrefix, setResultPrefix] = useState<string>('');
  const [resultSuffix, setResultSuffix] = useState<string>('');
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

  // Suffix may be empty (degenerates to plain completion — allowed).
  const prefixMissing = prefix.trim() === '';
  const checkpointMissing = checkpoint.trim() === '';
  const configMissing = config.trim() === '';
  const canComplete =
    !trainingAlive && !busy && !prefixMissing && !checkpointMissing && !configMissing;

  const onComplete = useCallback(async (): Promise<void> => {
    // Hard guard: never call the API while training is live.
    if (trainingAlive) return;
    if (prefixMissing || checkpointMissing || configMissing) return;

    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const req: FimRequest = {
        prefix: prefix,
        suffix: suffix,
        checkpoint: checkpoint,
        config: config,
        max_tokens: sampling.maxTokens,
        temperature: sampling.temperature,
        top_p: sampling.topP,
        top_k: sampling.topK,
      };
      const resp = await fimGenerate(req);
      if (isApiError(resp)) {
        setError(friendlyError(resp.error));
        return;
      }
      setResultPrefix(prefix);
      setResultSuffix(suffix);
      setResult(resp);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setError(friendlyError(message));
    } finally {
      setBusy(false);
    }
  }, [
    trainingAlive,
    prefixMissing,
    checkpointMissing,
    configMissing,
    prefix,
    suffix,
    checkpoint,
    config,
    sampling,
  ]);

  const validationHint = useMemo<string | null>(() => {
    if (trainingAlive) return null;
    const missing: string[] = [];
    if (prefixMissing) missing.push('a prefix');
    if (checkpointMissing) missing.push('a checkpoint');
    if (configMissing) missing.push('a config');
    if (missing.length === 0) return null;
    return `Need ${missing.join(', ')} before completing.`;
  }, [trainingAlive, prefixMissing, checkpointMissing, configMissing]);

  return (
    <div className="card card-wide fim-screen">
      <div className="md-toolbar fim-head">
        <h1 className="md-detail-title">Fill-In-the-Middle</h1>
        <span className="muted">Prefix + suffix in, the model fills the hole</span>
      </div>

      {trainingAlive && (
        <div className="fim-banner" role="status">
          {TRAINING_GUARD_MESSAGE}
        </div>
      )}

      <div className="fim-grid">
        {/* ── Controls ── */}
        <div className="fim-controls">
          <label className="fim-field">
            <span className="fim-label">Prefix — code before the hole</span>
            <textarea
              className="textarea fim-code mono"
              value={prefix}
              onChange={(e) => setPrefix(e.target.value)}
              placeholder={'function add(a: number, b: number): number {'}
              spellCheck={false}
              rows={6}
            />
          </label>

          <div className="fim-divider" aria-hidden="true">
            <span className="fim-divider-line" />
            <span className="fim-divider-label mono">▢ fill here ▢</span>
            <span className="fim-divider-line" />
          </div>

          <label className="fim-field">
            <span className="fim-label">Suffix — code after the hole (optional)</span>
            <textarea
              className="textarea fim-code mono"
              value={suffix}
              onChange={(e) => setSuffix(e.target.value)}
              placeholder={'\n}'}
              spellCheck={false}
              rows={6}
            />
          </label>

          <label className="fim-field">
            <span className="fim-label">Checkpoint</span>
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

          <label className="fim-field">
            <span className="fim-label">Config</span>
            <select className="select" value={config} onChange={(e) => setConfig(e.target.value)}>
              <option value="">(none)</option>
              {configs.map((c) => (
                <option key={c.path} value={c.path}>
                  {c.rel}
                </option>
              ))}
            </select>
            {configsError !== null && (
              <span className="fim-hint err">Could not load configs: {configsError}</span>
            )}
          </label>

          <div className="fim-sampling">
            <label className="fim-field">
              <span className="fim-label">Temperature</span>
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
            <label className="fim-field">
              <span className="fim-label">Max tokens</span>
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
            <label className="fim-field">
              <span className="fim-label">Top-p</span>
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
            <label className="fim-field">
              <span className="fim-label">Top-k</span>
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

          <div className="fim-actions">
            <button
              type="button"
              className="btn btn-primary fim-complete"
              onClick={() => void onComplete()}
              disabled={!canComplete}
              title={trainingAlive ? TRAINING_GUARD_MESSAGE : undefined}
            >
              {busy ? 'Filling…' : 'Complete'}
            </button>
            {validationHint !== null && <span className="fim-hint muted">{validationHint}</span>}
          </div>
        </div>

        {/* ── Output ── */}
        <div className="fim-output-pane">
          <OutputPane
            result={result}
            prefix={resultPrefix}
            suffix={resultSuffix}
            busy={busy}
            error={error}
          />
        </div>
      </div>
    </div>
  );
}
