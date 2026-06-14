import { useCallback, useEffect, useState } from 'react';
import type { DataSourcesView } from '../types';
import { isApiError } from '../types';
import { getDataSources } from '../api';

function pct(weight: number | null): string {
  if (weight === null || !Number.isFinite(weight)) return '—';
  return `${(weight * 100).toFixed(1)}%`;
}

export default function DataSourcesPanel() {
  const [view, setView] = useState<DataSourcesView | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const resp = await getDataSources();
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
        const resp = await getDataSources();
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

  const sources = view?.sources ?? [];

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Data Sources</div>
        <button className="btn" onClick={() => void load()}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {view && (
        <div className="muted mono">
          {view.summary} · total weight {pct(view.total_weight)} · {view.path}
        </div>
      )}

      {view && sources.length === 0 && !error && (
        <div className="muted">no data sources configured</div>
      )}

      {sources.length > 0 && (
        <table className="tbl">
          <thead>
            <tr>
              <th>name</th>
              <th>kind</th>
              <th className="right">weight</th>
              <th>dataset</th>
              <th>languages</th>
            </tr>
          </thead>
          <tbody>
            {sources.map((s) => (
              <tr key={s.name}>
                <td>{s.name}</td>
                <td>
                  {s.kind ? <span className="tag">{s.kind}</span> : <span className="muted">—</span>}
                </td>
                <td className="right mono">{pct(s.weight)}</td>
                <td className="mono">{s.dataset ?? '—'}</td>
                <td>
                  {s.languages.length === 0 ? (
                    <span className="muted">—</span>
                  ) : (
                    s.languages.map((lang) => (
                      <span key={lang} className="tag" style={{ marginRight: 4 }}>
                        {lang}
                      </span>
                    ))
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
