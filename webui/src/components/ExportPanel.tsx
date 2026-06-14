import { useCallback, useEffect, useState } from 'react';
import type { ExportOverview } from '../types';
import { isApiError } from '../types';
import { formatBytes, formatRelativeTime } from '../format';
import { getExports } from '../api';

export default function ExportPanel() {
  const [view, setView] = useState<ExportOverview | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const resp = await getExports();
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
        const resp = await getExports();
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

  const empty =
    view !== null &&
    view.checkpoints.length === 0 &&
    view.formats.length === 0 &&
    view.existing.length === 0;

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Export</div>
        <button className="btn" onClick={() => void load()}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {empty && !error && <div className="muted">nothing to export yet</div>}

      {view && !empty && (
        <>
          <div className="card-title">exportable checkpoints</div>
          {view.checkpoints.length === 0 ? (
            <div className="muted">no checkpoints</div>
          ) : (
            <table className="tbl">
              <thead>
                <tr>
                  <th>model</th>
                  <th>name</th>
                  <th className="right">step</th>
                </tr>
              </thead>
              <tbody>
                {view.checkpoints.map((c) => (
                  <tr key={c.path}>
                    <td className="mono">{c.model}</td>
                    <td className="mono">{c.name}</td>
                    <td className="right mono">{c.step ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <div className="card-title">supported formats</div>
          {view.formats.length === 0 ? (
            <div className="muted">none</div>
          ) : (
            view.formats.map((f) => (
              <div className="row" key={f.key}>
                <span className="k">
                  <span className="tag">{f.label}</span>
                </span>
                <span className="muted">{f.desc}</span>
              </div>
            ))
          )}

          <div className="card-title">existing exports</div>
          {view.existing.length === 0 ? (
            <div className="muted">none yet</div>
          ) : (
            <table className="tbl">
              <thead>
                <tr>
                  <th>path</th>
                  <th>format</th>
                  <th className="right">size</th>
                  <th className="right">modified</th>
                </tr>
              </thead>
              <tbody>
                {view.existing.map((e) => (
                  <tr key={e.path}>
                    <td className="mono">{e.path}</td>
                    <td>
                      <span className="tag">{e.format}</span>
                    </td>
                    <td className="right mono">{formatBytes(e.size_bytes)}</td>
                    <td className="right mono">{formatRelativeTime(e.mtime)}</td>
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
