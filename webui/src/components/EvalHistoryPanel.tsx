import { useCallback, useEffect, useMemo, useState } from 'react';
import type { EvalHistoryView, EvalSnapshot, JsonValue } from '../types';
import { isApiError } from '../types';
import { getEvalHistory } from '../api';
import { formatFloat, formatRelativeTime } from '../format';
import Sparkline from './Sparkline';

function asNumber(value: JsonValue): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  return null;
}

interface MetricRow {
  snap: EvalSnapshot;
  value: number | null;
  delta: number | null;
}

/** Class + glyph for a step-over-step delta of the selected metric. */
function deltaParts(delta: number | null): { cls: string; text: string } {
  if (delta === null || delta === 0) return { cls: 'evalhist-delta', text: '—' };
  const sign = delta > 0 ? '▲' : '▼';
  const cls = delta > 0 ? 'evalhist-delta evalhist-delta--up' : 'evalhist-delta evalhist-delta--down';
  return { cls, text: `${sign} ${formatFloat(Math.abs(delta), 4)}` };
}

export default function EvalHistoryPanel() {
  const [view, setView] = useState<EvalHistoryView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [metric, setMetric] = useState<string>('');

  const load = useCallback(async () => {
    setError(null);
    try {
      const resp = await getEvalHistory();
      if (isApiError(resp)) {
        setError(resp.error);
        setView(null);
      } else {
        setView(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const resp = await getEvalHistory();
        if (!active) return;
        if (isApiError(resp)) setError(resp.error);
        else setView(resp);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  // Default the selected metric to the first available metric key.
  useEffect(() => {
    const keys = view?.metric_keys ?? [];
    if (keys.length > 0 && !keys.includes(metric)) {
      setMetric(keys[0]);
    }
  }, [view, metric]);

  const snapshots = view?.snapshots ?? [];

  const rows = useMemo(() => {
    if (!metric) return [];
    const out: MetricRow[] = [];
    let prev: number | null = null;
    for (const snap of snapshots) {
      const value = asNumber(snap.metrics[metric]);
      const delta = value !== null && prev !== null ? value - prev : null;
      out.push({ snap, value, delta });
      if (value !== null) prev = value;
    }
    return out;
  }, [snapshots, metric]);

  const series = useMemo(() => {
    const out: number[] = [];
    for (const r of rows) {
      if (r.value !== null) out.push(r.value);
    }
    return out;
  }, [rows]);

  const range = useMemo(() => {
    if (series.length === 0) return null;
    let min = series[0];
    let max = series[0];
    let last = series[series.length - 1];
    for (const v of series) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    return { min, max, last };
  }, [series]);

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Eval History</div>
        <button className="btn" onClick={() => void load()}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {view && snapshots.length === 0 && !error && (
        <div className="muted">no eval-over-training snapshots yet</div>
      )}

      {snapshots.length > 0 && (
        <>
          <div className="row">
            <span className="muted mono">{view?.count ?? snapshots.length} snapshots</span>
            <select
              className="select"
              value={metric}
              onChange={(e) => setMetric(e.target.value)}
              disabled={(view?.metric_keys.length ?? 0) === 0}
            >
              {(view?.metric_keys ?? []).length === 0 ? (
                <option value="">no numeric metrics</option>
              ) : (
                view?.metric_keys.map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))
              )}
            </select>
          </div>

          {metric && (
            <>
              {range && (
                <div className="stat-tiles evalhist-tiles">
                  <div className="stat-tile">
                    <div className="stat-tile-label">latest</div>
                    <div className="stat-tile-value mono">{formatFloat(range.last, 4)}</div>
                  </div>
                  <div className="stat-tile">
                    <div className="stat-tile-label">min</div>
                    <div className="stat-tile-value mono">{formatFloat(range.min, 4)}</div>
                  </div>
                  <div className="stat-tile">
                    <div className="stat-tile-label">max</div>
                    <div className="stat-tile-value mono">{formatFloat(range.max, 4)}</div>
                  </div>
                </div>
              )}

              <div className="evalhist-chart">
                <div className="evalhist-chart-title muted mono">{metric}</div>
                <Sparkline points={series} stroke="var(--accent)" width={640} height={96} />
              </div>

              <table className="tbl">
                <thead>
                  <tr>
                    <th className="right">step</th>
                    <th className="right">{metric}</th>
                    <th className="right">Δ</th>
                    <th className="right">when</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => {
                    const d = deltaParts(r.delta);
                    return (
                      <tr key={r.snap.path}>
                        <td className="right mono">{r.snap.step ?? '—'}</td>
                        <td className="right mono">{formatFloat(r.value, 4)}</td>
                        <td className="right mono">
                          <span className={d.cls}>{d.text}</span>
                        </td>
                        <td className="right mono muted">{formatRelativeTime(r.snap.mtime)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </>
          )}
        </>
      )}
    </div>
  );
}
