import { useCallback, useEffect, useState } from 'react';
import type { RouterOverview } from '../types';
import { isApiError } from '../types';
import { getRouter } from '../api';

export default function RouterPanel() {
  const [view, setView] = useState<RouterOverview | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const resp = await getRouter();
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
        const resp = await getRouter();
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

  const empty = view !== null && !view.has_router && view.checkpoints.length === 0;

  return (
    <div className="card">
      <div className="row">
        <div className="card-title">Router</div>
        <button className="btn" onClick={() => void load()}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {empty && !error && <div className="muted">router not trained yet</div>}

      {view && !empty && (
        <>
          <div className="row">
            <span className="k">
              <span className={`dot ${view.has_router ? 'live' : 'dead'}`} /> trained
            </span>
            <span className={`tag ${view.has_router ? 'done' : 'failed'}`}>
              {view.has_router ? 'yes' : 'no'}
            </span>
          </div>

          {view.domains.length > 0 && (
            <div className="row">
              <span className="k">domains</span>
              <span>
                {view.domains.map((d) => (
                  <span key={d} className="tag" style={{ marginLeft: 4 }}>
                    {d}
                  </span>
                ))}
              </span>
            </div>
          )}

          {view.checkpoints.length > 0 && (
            <table className="tbl">
              <thead>
                <tr>
                  <th>checkpoint</th>
                  <th className="right">step</th>
                </tr>
              </thead>
              <tbody>
                {view.checkpoints.map((c) => (
                  <tr key={c.path}>
                    <td className="mono">{c.name}</td>
                    <td className="right mono">{c.step ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
    </div>
  );
}
