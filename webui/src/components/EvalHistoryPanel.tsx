import { useCallback, useEffect, useMemo, useState } from 'react';
import type { EvalHistoryView, EvalSnapshot } from '../types';
import { isApiError } from '../types';
import { getEvalHistory } from '../api';
import Sparkline from './Sparkline';

function asNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  return null;
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
    const out: { snap: EvalSnapshot; value: number | null }[] = [];
    for (const snap of snapshots) {
      out.push({ snap, value: asNumber(snap.metrics[metric]) });
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
              <div className="card-title">{metric}</div>
              <Sparkline points={series} stroke="#4f9cf9" />
              <table className="tbl">
                <thead>
                  <tr>
                    <th className="right">step</th>
                    <th className="right">{metric}</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.snap.path}>
                      <td className="right mono">{r.snap.step ?? '—'}</td>
                      <td className="right mono">
                        {r.value === null ? '—' : r.value.toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </>
      )}
    </div>
  );
}
