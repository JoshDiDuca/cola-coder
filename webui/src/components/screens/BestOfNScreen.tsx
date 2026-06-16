import { useCallback, useEffect, useMemo, useState } from 'react';
import type { ChangeEvent } from 'react';
import type {
  BestOfNRequest,
  BestOfNCandidate,
  BestOfNResponse,
  ConfigFile,
} from '../../types';
import { isApiError } from '../../types';
import { bestOfN, getConfigs } from '../../api';
import { formatDuration, formatInteger, formatPercent } from '../../format';

// ── Best-of-N playground (R10) ──────────────────────────────────────────────
// The batch-verified sibling of InferenceScreen. We generate N candidates for a
// single prompt; the backend VERIFIES each one (tsc --noEmit / exec / parse, all
// sandboxed) and returns the best completion plus every candidate ranked by
// score with its verdict. This is a single batch call (NOT a token stream) — it
// is slow, so the busy state sets expectations explicitly.
//
// CRITICAL SAFETY: verified generation contends for the GPU with the live
// training run, so the backend refuses with HTTP 409 while training is alive. We
// never even attempt the call when `trainingAlive` is true and show a prominent
// amber banner; the bestOfN() helper's j() THROWS on non-OK HTTP (a 409 message
// contains "409"), and may ALSO resolve an ApiError body — we handle both.

interface BestOfNScreenProps {
  // Checkpoint paths from the live snapshot (App passes snap.checkpoints.map(c => c.path)).
  checkpoints: string[];
  // When true, best-of-N is disabled with a clear banner.
  trainingAlive: boolean;
}

// The exact request union for `language` — never widened to string.
type GenLanguage = 'auto' | 'python' | 'typescript';

const LANGUAGE_OPTIONS: readonly GenLanguage[] = ['auto', 'python', 'typescript'];

// Sampling + batch defaults (see Behavior spec).
const DEFAULT_NUM_CANDIDATES = 4;
const MIN_NUM_CANDIDATES = 1;
const MAX_NUM_CANDIDATES = 16;
const DEFAULT_TEMPERATURE = 0.8;
const DEFAULT_MAX_TOKENS = 256;
const DEFAULT_TOP_P = 0.9;
const DEFAULT_TOP_K = 50;
const DEFAULT_MIN_P = 0; // 0 = disabled

const TRAINING_GUARD_MESSAGE =
  'A training run is live — best-of-N is disabled to protect the GPU. ' +
  'It will be available when training finishes.';

// Typed form state: numbers stay numbers (parsed on change), never strings.
interface BestOfNFormState {
  numCandidates: number;
  language: GenLanguage;
  temperature: number;
  maxTokens: number;
  topP: number;
  topK: number;
  minP: number;
}

const DEFAULT_FORM: BestOfNFormState = {
  numCandidates: DEFAULT_NUM_CANDIDATES,
  language: 'auto',
  temperature: DEFAULT_TEMPERATURE,
  maxTokens: DEFAULT_MAX_TOKENS,
  topP: DEFAULT_TOP_P,
  topK: DEFAULT_TOP_K,
  minP: DEFAULT_MIN_P,
};

/** Parse a numeric <input> value, falling back to a default when blank/NaN. */
function parseNumber(raw: string, fallback: number): number {
  const n = Number(raw);
  return raw.trim() === '' || Number.isNaN(n) ? fallback : n;
}

/** Clamp the candidate count into the allowed [min, max] band. */
function clampCandidates(n: number): number {
  const rounded = Math.round(n);
  if (rounded < MIN_NUM_CANDIDATES) return MIN_NUM_CANDIDATES;
  if (rounded > MAX_NUM_CANDIDATES) return MAX_NUM_CANDIDATES;
  return rounded;
}

/** Narrow an arbitrary select value to the exact GenLanguage union. */
function toLanguage(raw: string): GenLanguage {
  switch (raw) {
    case 'python':
      return 'python';
    case 'typescript':
      return 'typescript';
    case 'auto':
      return 'auto';
    default:
      return 'auto';
  }
}

function lastSegment(path: string): string {
  const parts = path.split(/[\\/]/).filter((p) => p !== '');
  return parts.length === 0 ? path : parts.slice(-2).join('/');
}

// ── Screen ────────────────────────────────────────────────────────────────────

export default function BestOfNScreen({
  checkpoints,
  trainingAlive,
}: BestOfNScreenProps): JSX.Element {
  const [prompt, setPrompt] = useState<string>('');
  const [checkpoint, setCheckpoint] = useState<string>('');
  const [config, setConfig] = useState<string>('');
  const [form, setForm] = useState<BestOfNFormState>(DEFAULT_FORM);

  const [configs, setConfigs] = useState<ConfigFile[]>([]);
  const [configsError, setConfigsError] = useState<string | null>(null);

  const [busy, setBusy] = useState<boolean>(false);
  const [result, setResult] = useState<BestOfNResponse | null>(null);
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

  const onGenerate = useCallback((): void => {
    // Hard guard: never call the API while training is live.
    if (trainingAlive) return;
    if (promptMissing || checkpointMissing || configMissing) return;
    if (busy) return;

    const req: BestOfNRequest = {
      prompt: prompt,
      checkpoint: checkpoint,
      config: config,
      num_candidates: form.numCandidates,
      language: form.language,
      max_tokens: form.maxTokens,
      temperature: form.temperature,
      top_p: form.topP,
      top_k: form.topK,
      min_p: form.minP,
    };

    setBusy(true);
    setError(null);
    setResult(null);

    void (async (): Promise<void> => {
      try {
        const res = await bestOfN(req);
        if (isApiError(res)) {
          // Resolved error body (e.g. verifier/load failure, or a stale-snapshot 409).
          setError(res.error);
          return;
        }
        setResult(res);
      } catch (e) {
        // j() threw on non-OK HTTP. A 409 (training live, snapshot briefly stale)
        // surfaces the GPU-guard message; anything else shows the raw error text.
        const message = e instanceof Error ? e.message : String(e);
        setError(message.includes('409') ? TRAINING_GUARD_MESSAGE : message);
      } finally {
        setBusy(false);
      }
    })();
  }, [
    trainingAlive,
    promptMissing,
    checkpointMissing,
    configMissing,
    busy,
    prompt,
    checkpoint,
    config,
    form,
  ]);

  const validationHint = useMemo<string | null>(() => {
    if (trainingAlive) return null;
    const missing: string[] = [];
    if (promptMissing) missing.push('a prompt');
    if (checkpointMissing) missing.push('a checkpoint');
    if (configMissing) missing.push('a config');
    if (missing.length === 0) return null;
    return `Need ${missing.join(', ')} before verifying.`;
  }, [trainingAlive, promptMissing, checkpointMissing, configMissing]);

  const buttonLabel = busy
    ? `Verifying ${formatInteger(form.numCandidates)} candidates…`
    : 'Generate & verify';

  return (
    <div className="card card-wide bon-screen">
      <div className="md-toolbar bon-head">
        <h1 className="md-detail-title">Best-of-N Playground</h1>
        <span className="muted">
          Generate N candidates — the sandbox verifies each and ranks them
        </span>
      </div>

      {trainingAlive && (
        <div className="bon-banner" role="status">
          {TRAINING_GUARD_MESSAGE}
        </div>
      )}

      <div className="bon-grid">
        {/* ── Controls ── */}
        <div className="bon-controls">
          <label className="bon-field">
            <span className="bon-label">Prompt</span>
            <textarea
              className="textarea bon-prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="def is_prime(n: int) -> bool:"
              spellCheck={false}
              rows={8}
            />
          </label>

          <label className="bon-field">
            <span className="bon-label">Checkpoint</span>
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

          <label className="bon-field">
            <span className="bon-label">Config</span>
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
              <span className="bon-hint err">Could not load configs: {configsError}</span>
            )}
          </label>

          <div className="bon-batch">
            <label className="bon-field">
              <span className="bon-label">Candidates (N)</span>
              <input
                type="number"
                step={1}
                min={MIN_NUM_CANDIDATES}
                max={MAX_NUM_CANDIDATES}
                className="input"
                value={form.numCandidates}
                onChange={(e) =>
                  setForm((f) => ({
                    ...f,
                    numCandidates: clampCandidates(
                      parseNumber(e.target.value, DEFAULT_NUM_CANDIDATES),
                    ),
                  }))
                }
              />
            </label>
            <label className="bon-field">
              <span className="bon-label">Language</span>
              <select
                className="select"
                value={form.language}
                onChange={(e) =>
                  setForm((f) => ({ ...f, language: toLanguage(e.target.value) }))
                }
              >
                {LANGUAGE_OPTIONS.map((lang) => (
                  <option key={lang} value={lang}>
                    {lang}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="bon-sampling">
            <label className="bon-field">
              <span className="bon-label">Temperature</span>
              <input
                type="number"
                step="any"
                min={0}
                className="input"
                value={form.temperature}
                onChange={(e) =>
                  setForm((f) => ({
                    ...f,
                    temperature: parseNumber(e.target.value, DEFAULT_TEMPERATURE),
                  }))
                }
              />
            </label>
            <label className="bon-field">
              <span className="bon-label">Max tokens</span>
              <input
                type="number"
                step={1}
                min={1}
                className="input"
                value={form.maxTokens}
                onChange={(e) =>
                  setForm((f) => ({
                    ...f,
                    maxTokens: parseNumber(e.target.value, DEFAULT_MAX_TOKENS),
                  }))
                }
              />
            </label>
            <label className="bon-field">
              <span className="bon-label">Top-p</span>
              <input
                type="number"
                step="any"
                min={0}
                max={1}
                className="input"
                value={form.topP}
                onChange={(e) =>
                  setForm((f) => ({
                    ...f,
                    topP: parseNumber(e.target.value, DEFAULT_TOP_P),
                  }))
                }
              />
            </label>
            <label className="bon-field">
              <span className="bon-label">Top-k</span>
              <input
                type="number"
                step={1}
                min={0}
                className="input"
                value={form.topK}
                onChange={(e) =>
                  setForm((f) => ({
                    ...f,
                    topK: parseNumber(e.target.value, DEFAULT_TOP_K),
                  }))
                }
              />
            </label>
            <label className="bon-field">
              <span className="bon-label">min-p</span>
              <input
                type="number"
                step={0.01}
                min={0}
                max={1}
                className="input"
                value={form.minP}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  setForm((f) => ({
                    ...f,
                    minP: parseNumber(e.target.value, DEFAULT_MIN_P),
                  }))
                }
              />
              <span className="bon-hint muted">0 = off</span>
            </label>
          </div>

          <div className="bon-actions">
            <button
              type="button"
              className="btn btn-primary bon-generate"
              onClick={onGenerate}
              disabled={!canGenerate}
              title={trainingAlive ? TRAINING_GUARD_MESSAGE : undefined}
            >
              {buttonLabel}
            </button>
            {validationHint !== null && (
              <span className="bon-hint muted">{validationHint}</span>
            )}
          </div>
        </div>

        {/* ── Output (batch — best + ranked candidates) ── */}
        <div className="bon-output-pane">
          {error !== null && (
            <div className="bon-error" role="alert">
              {error}
            </div>
          )}

          {error === null && result === null && !busy && (
            <div className="bon-empty muted">
              No output yet — write a prompt, pick a checkpoint and config, choose how many
              candidates to verify, then Generate &amp; verify.
            </div>
          )}

          {error === null && busy && (
            <div className="bon-empty muted">
              Verifying {formatInteger(form.numCandidates)} candidates in the sandbox — this is
              slower than plain generation. Hang tight.
            </div>
          )}

          {error === null && result !== null && (
            <BestOfNResult result={result} />
          )}
        </div>
      </div>
    </div>
  );
}

// ── Result panel ───────────────────────────────────────────────────────────────

interface BestOfNResultProps {
  result: BestOfNResponse;
}

function BestOfNResult({ result }: BestOfNResultProps): JSX.Element {
  return (
    <div className="bon-result">
      <div className="bon-result-head">
        <span className={result.solved ? 'tag bon-tag-good' : 'tag bon-tag-bad'}>
          {result.solved ? 'verified ✓' : 'unverified'}
        </span>
        <span className="muted">
          verifier <span className="mono">{result.verifier}</span>
        </span>
        <span className="muted">
          {formatInteger(result.candidates_used)} candidates
        </span>
        <span className="muted">{formatDuration(result.elapsed_s)}</span>
      </div>

      <div className="bon-best">
        <div className="bon-section-label">Best completion</div>
        <pre className="mono scroll bon-best-body">{result.best_completion}</pre>
      </div>

      <div className="bon-candidates">
        <div className="bon-section-label">
          All {formatInteger(result.candidates.length)} candidates (ranked)
        </div>
        {result.candidates.map((candidate, index) => (
          <CandidateRow
            key={index}
            candidate={candidate}
            index={index}
            isBest={index === 0}
          />
        ))}
      </div>
    </div>
  );
}

interface CandidateRowProps {
  candidate: BestOfNCandidate;
  index: number;
  isBest: boolean;
}

function CandidateRow({ candidate, index, isBest }: CandidateRowProps): JSX.Element {
  return (
    <div className={isBest ? 'bon-cand bon-cand-best' : 'bon-cand'}>
      <div className="bon-cand-head">
        <span className="bon-cand-rank mono">#{index + 1}</span>
        {isBest && <span className="tag bon-tag-best">best</span>}
        <span className="bon-cand-score mono">{formatPercent(candidate.score)}</span>
        <span className={candidate.verified ? 'tag bon-tag-good' : 'tag bon-tag-muted'}>
          {candidate.verified ? 'verified ✓' : 'unverified'}
        </span>
      </div>
      <pre className="mono scroll bon-cand-body">{candidate.completion}</pre>
    </div>
  );
}
