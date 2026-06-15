import { useCallback, useEffect, useRef, useState } from 'react';
import type { TrainingStatus } from '../types';
import { isApiError } from '../types';
import { formatInteger, formatFloat, formatPercent } from '../format';
import { getLog } from '../api';

// The training run is written to this log file (confirmed present in getLogs()).
// The dashboard tails it live so the user can watch training as it happens.
const TRAIN_LOG_PATH = 'train_small_react_best.log';
const TAIL_LINES = 40;
const REFRESH_MS = 5000;

type RunState = 'running' | 'idle' | 'offline';

interface StatTile {
  label: string;
  value: string;
}

/** Derive a friendly run name from the log file name: train_X.log → X. */
function runNameFromLog(logPath: string): string {
  const base = logPath.replace(/\.log$/i, '');
  return base.replace(/^train_/i, '');
}

function runState(training: TrainingStatus | null): RunState {
  if (training === null) return 'offline';
  return training.alive ? 'running' : 'idle';
}

function stateLabel(state: RunState): string {
  switch (state) {
    case 'running':
      return 'running';
    case 'idle':
      return 'idle';
    case 'offline':
      return 'offline';
    default: {
      const _exhaustive: never = state;
      return _exhaustive;
    }
  }
}

// The live log tail is one of three things: still loading, an error message we
// surfaced gracefully, or the joined log body.
type LogTailState =
  | { kind: 'loading' }
  | { kind: 'error'; message: string }
  | { kind: 'text'; body: string };

export default function LiveTrainingPanel({
  training,
}: {
  training: TrainingStatus | null;
}): JSX.Element {
  const state = runState(training);
  const live = state === 'running';
  const runName = runNameFromLog(TRAIN_LOG_PATH);

  const step = training?.step ?? null;
  const totalSteps = training?.total_steps ?? null;
  const progressFraction = training?.progress_pct ?? null;
  // progress_pct is already a 0..100 percentage (matches TrainingPanel's bar usage).
  const fillPct = Math.max(0, Math.min(100, progressFraction ?? 0));

  const tiles: StatTile[] = [
    { label: 'loss', value: formatFloat(training?.loss ?? null, 4) },
    { label: 'perplexity', value: formatFloat(training?.ppl ?? null, 2) },
    { label: 'tok / s', value: formatInteger(training?.tok_per_s ?? null) },
    { label: 's / it', value: formatFloat(training?.s_per_it ?? null, 1) },
  ];

  const [tail, setTail] = useState<LogTailState>({ kind: 'loading' });
  const preRef = useRef<HTMLPreElement | null>(null);

  const loadTail = useCallback(async () => {
    try {
      const result = await getLog(TRAIN_LOG_PATH, TAIL_LINES);
      if (isApiError(result)) {
        // The most common case here is the log not existing yet — show a
        // friendly note rather than a raw error.
        setTail({
          kind: 'error',
          message: `training log not found yet — ${result.error}`,
        });
        return;
      }
      const body = result.lines.join('\n');
      setTail({ kind: 'text', body: result.truncated ? `…(truncated)\n${body}` : body });
    } catch (e) {
      setTail({ kind: 'error', message: e instanceof Error ? e.message : String(e) });
    }
  }, []);

  // Auto-refresh the log tail every ~5s; clear the interval on unmount.
  useEffect(() => {
    let active = true;
    void (async () => {
      await loadTail();
      if (!active) return;
    })();
    const id = window.setInterval(() => {
      void loadTail();
    }, REFRESH_MS);
    return () => {
      active = false;
      window.clearInterval(id);
    };
  }, [loadTail]);

  // Auto-scroll the log box to the bottom whenever new content arrives.
  useEffect(() => {
    const el = preRef.current;
    if (el !== null) {
      el.scrollTop = el.scrollHeight;
    }
  }, [tail]);

  return (
    <div className={`card card-wide live-train${live ? '' : ' live-train-dim'}`}>
      <div className="row live-train-head">
        <div className="card-title live-train-title">
          <span
            className={live ? 'dot live' : 'dot dead'}
            role="img"
            aria-label={stateLabel(state)}
            title={stateLabel(state)}
          />
          Live Training
          <span className="tag mono live-train-run">{runName}</span>
        </div>
        <span className={`tag${live ? ' live' : ''}`}>{stateLabel(state)}</span>
      </div>

      <div className="live-train-progress">
        <div className="live-train-steps">
          <span className="stat-big mono">{formatInteger(step)}</span>
          <span className="muted mono live-train-steps-total">
            / {formatInteger(totalSteps)} steps
          </span>
          <span className="muted mono live-train-pct">{formatPercent(progressFraction)}</span>
        </div>
        <div
          className="bar live-train-bar"
          role="progressbar"
          aria-valuenow={Math.round(fillPct)}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div className="fill" style={{ width: `${fillPct}%` }} />
        </div>
      </div>

      <div className="stat-tiles">
        {tiles.map((tile) => (
          <div className="stat-tile" key={tile.label}>
            <div className="stat-tile-label">{tile.label}</div>
            <div className="stat-tile-value mono">{tile.value}</div>
          </div>
        ))}
      </div>

      <div className="live-train-log">
        <div className="live-train-log-head">
          <span className="mono muted">{TRAIN_LOG_PATH}</span>
          <span className="muted live-train-log-note">auto-refreshing every 5s</span>
        </div>
        <pre className="pre scroll live-train-log-body mono" ref={preRef}>
          {tail.kind === 'loading' ? (
            'loading…'
          ) : tail.kind === 'error' ? (
            <span className="muted">{tail.message}</span>
          ) : tail.body !== '' ? (
            tail.body
          ) : (
            <span className="muted">no log output yet</span>
          )}
        </pre>
      </div>
    </div>
  );
}
