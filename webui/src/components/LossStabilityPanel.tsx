import { useCallback, useEffect, useState } from 'react';
import type { LossStability } from '../types';
import { isApiError } from '../types';
import { formatFloat, formatInteger } from '../format';
import { getLossStability } from '../api';

const REFRESH_MS = 15000;

type Verdict = LossStability['verdict'];
type Trend = LossStability['trend'];

// The fetched stability summary is one of three things: still loading, an error
// message we surfaced gracefully, or the resolved stability payload.
type StabilityState =
  | { kind: 'loading' }
  | { kind: 'error'; message: string }
  | { kind: 'ready'; data: LossStability };

/** Map the verdict to a status-tag CSS class, exhaustively. */
function verdictTagClass(verdict: Verdict): string {
  switch (verdict) {
    case 'stable':
      return 'tag done';
    case 'watch':
      return 'tag warn';
    case 'spiking':
      return 'tag failed';
    case 'insufficient_data':
      return 'tag';
    default: {
      const _exhaustive: never = verdict;
      return _exhaustive;
    }
  }
}

/** A small directional glyph for the loss trend, exhaustively. */
function trendArrow(trend: Trend): string {
  switch (trend) {
    case 'improving':
      return '▼';
    case 'worsening':
      return '▲';
    case 'flat':
      return '▬';
    case 'unknown':
      return '—';
    default: {
      const _exhaustive: never = trend;
      return _exhaustive;
    }
  }
}

export default function LossStabilityPanel(): JSX.Element {
  const [state, setState] = useState<StabilityState>({ kind: 'loading' });

  const loadStability = useCallback(async () => {
    try {
      const result = await getLossStability();
      if (isApiError(result)) {
        setState({ kind: 'error', message: result.error });
        return;
      }
      setState({ kind: 'ready', data: result });
    } catch (e) {
      setState({ kind: 'error', message: e instanceof Error ? e.message : String(e) });
    }
  }, []);

  // Refresh every ~15s; clear the interval and guard against late updates on unmount.
  useEffect(() => {
    let active = true;
    void (async () => {
      await loadStability();
      if (!active) return;
    })();
    const id = window.setInterval(() => {
      void loadStability();
    }, REFRESH_MS);
    return () => {
      active = false;
      window.clearInterval(id);
    };
  }, [loadStability]);

  return (
    <div className="card">
      <div className="row">
        <div className="card-title">Loss stability</div>
        {state.kind === 'ready' && (
          <span className={verdictTagClass(state.data.verdict)}>{state.data.verdict}</span>
        )}
      </div>

      {state.kind === 'loading' && <div className="muted">loading…</div>}

      {state.kind === 'error' && <div className="err">{state.message}</div>}

      {state.kind === 'ready' &&
        (state.data.verdict === 'insufficient_data' ? (
          <div className="muted">not enough log history yet</div>
        ) : (
          <>
            <div className="row">
              <span className="muted">trend</span>
              <span className="mono">
                {trendArrow(state.data.trend)} {state.data.trend}
              </span>
            </div>
            <div className="row">
              <span className="muted">current loss</span>
              <span className="mono">
                {state.data.current_loss != null ? formatFloat(state.data.current_loss, 4) : '—'}
              </span>
            </div>
            <div className="row">
              <span className="muted">EMA loss</span>
              <span className="mono">
                {state.data.ema_loss != null ? formatFloat(state.data.ema_loss, 4) : '—'}
              </span>
            </div>
            <div className="row">
              <span className="muted">spike count</span>
              <span className="mono">{formatInteger(state.data.spike_count)}</span>
            </div>
            <div className="row">
              <span className="muted">recent max z</span>
              <span className="mono">
                {state.data.recent_max_z != null ? formatFloat(state.data.recent_max_z, 2) : '—'}
              </span>
            </div>
            <div className="row">
              <span className="muted">points used</span>
              <span className="mono">{formatInteger(state.data.points_used)}</span>
            </div>
          </>
        ))}

      <div className="muted">z-score spike detection over the logged loss curve</div>
    </div>
  );
}
