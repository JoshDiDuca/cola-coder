import { useCallback, useEffect, useState } from 'react';
import type { StorageView } from '../types';
import { isApiError } from '../types';
import { getStorage } from '../api';
import { formatBytes } from '../format';

export default function StoragePanel() {
  const [view, setView] = useState<StorageView | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await getStorage();
      if (isApiError(resp)) {
        setError(resp.error);
        setView(null);
      } else {
        setView(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="card">
      <div className="row" style={{ borderBottom: 'none', padding: 0 }}>
        <span className="card-title">Storage</span>
        <button className="btn" onClick={() => void load()} disabled={loading}>
          Refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}
      {loading && !view && <div className="muted">loading…</div>}

      {view && (
        <>
          <div className="muted mono">{view.path}</div>

          <div className="row">
            <span className="k">tokenizer_path</span>
            <span className="v">{view.tokenizer_path ?? '—'}</span>
          </div>
          <div className="row">
            <span className="k">data_dir</span>
            <span className="v">{view.data_dir ?? '—'}</span>
          </div>
          <div className="row">
            <span className="k">checkpoint_dir</span>
            <span className="v">{view.checkpoint_dir ?? '—'}</span>
          </div>

          <div className="card-title">entries</div>
          {view.entries.length === 0 ? (
            <div className="muted">none</div>
          ) : (
            <table className="tbl">
              <thead>
                <tr>
                  <th>name</th>
                  <th>exists</th>
                  <th>path</th>
                  <th className="right">size</th>
                </tr>
              </thead>
              <tbody>
                {view.entries.map((e) => (
                  <tr key={e.path}>
                    <td>{e.name}</td>
                    <td>
                      <span className={`dot ${e.exists ? 'live' : 'dead'}`} />
                    </td>
                    <td className="mono">{e.path}</td>
                    <td className="right mono">{formatBytes(e.size_bytes)}</td>
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
