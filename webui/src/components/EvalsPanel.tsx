import { useCallback, useEffect, useState } from 'react';
import type { EvalResult, EvalDetail } from '../types';
import { isApiError } from '../types';
import { getEvals, getEval } from '../api';
import { formatJsonValue, formatRelativeTime } from '../format';

function formatDetail(d: EvalDetail): string {
  if (d.parsed !== null) {
    const text = formatJsonValue(d.parsed);
    return d.truncated ? `${text}\n…(truncated)` : text;
  }
  const body = d.content ?? '';
  return d.truncated ? `${body}\n…(truncated)` : body;
}

export default function EvalsPanel() {
  const [evals, setEvals] = useState<EvalResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [detailText, setDetailText] = useState<string>('');
  const [detailLoading, setDetailLoading] = useState(false);

  const load = useCallback(async () => {
    setError(null);
    try {
      setEvals(await getEvals());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const next = await getEvals();
        if (active) setEvals(next);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const onView = useCallback(async (ev: EvalResult) => {
    setSelectedPath(ev.path);
    setDetailLoading(true);
    setDetailText('');

    try {
      const d = await getEval(ev.path);
      setDetailText(isApiError(d) ? `error: ${d.error}` : formatDetail(d));
    } catch (e) {
      setDetailText(`error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setDetailLoading(false);
    }
  }, []);

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Evals</div>
        <button className="btn" onClick={() => void load()}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {evals.length === 0 && !error ? (
        <div className="muted">no eval artifacts found</div>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>name</th>
              <th>kind</th>
              <th>summary</th>
              <th className="right">when</th>
              <th className="right">view</th>
            </tr>
          </thead>
          <tbody>
            {evals.map((ev) => (
              <tr key={ev.path}>
                <td className="mono">{ev.name}</td>
                <td>
                  <span className="tag">{ev.kind}</span>
                </td>
                <td>{ev.summary}</td>
                <td className="right mono">{formatRelativeTime(ev.mtime)}</td>
                <td className="right">
                  <button className="btn" onClick={() => void onView(ev)}>
                    view
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {selectedPath !== null && (
        <pre className="pre scroll">{detailLoading ? 'loading…' : detailText}</pre>
      )}
    </div>
  );
}
