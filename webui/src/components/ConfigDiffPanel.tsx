import { useCallback, useEffect, useState } from 'react';
import type { ConfigFile, ConfigDiff } from '../types';
import { isApiError } from '../types';
import { getConfigs, getConfigDiff } from '../api';
import { formatJsonValue } from '../format';

export default function ConfigDiffPanel() {
  const [configs, setConfigs] = useState<ConfigFile[]>([]);
  const [aPath, setAPath] = useState<string>('');
  const [bPath, setBPath] = useState<string>('');
  const [diff, setDiff] = useState<ConfigDiff | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const next = await getConfigs();
        if (active) setConfigs(next);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const canCompare = aPath !== '' && bPath !== '' && aPath !== bPath;

  const onCompare = useCallback(async () => {
    if (!canCompare) return;
    setDiff(null);
    setError(null);
    setLoading(true);
    try {
      const resp = await getConfigDiff(aPath, bPath);
      if (isApiError(resp)) setError(resp.error);
      else setDiff(resp);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [aPath, bPath, canCompare]);

  return (
    <div className="card card-wide">
      <div className="card-title">Config Diff</div>

      {configs.length < 2 && !error ? (
        <div className="muted">need two distinct configs</div>
      ) : (
        <>
          <div className="row" style={{ borderBottom: 'none', flexWrap: 'wrap' }}>
            <select
              className="select"
              value={aPath}
              onChange={(e) => setAPath(e.target.value)}
              style={{ flex: 1, minWidth: 180 }}
            >
              <option value="">A: select config…</option>
              {configs.map((cfg) => (
                <option key={cfg.path} value={cfg.path}>
                  {cfg.rel}
                </option>
              ))}
            </select>
            <select
              className="select"
              value={bPath}
              onChange={(e) => setBPath(e.target.value)}
              style={{ flex: 1, minWidth: 180 }}
            >
              <option value="">B: select config…</option>
              {configs.map((cfg) => (
                <option key={cfg.path} value={cfg.path}>
                  {cfg.rel}
                </option>
              ))}
            </select>
            <button
              className="btn btn-primary"
              onClick={() => void onCompare()}
              disabled={!canCompare || loading}
            >
              Compare
            </button>
          </div>

          {aPath !== '' && bPath !== '' && aPath === bPath && (
            <div className="muted">pick two distinct configs</div>
          )}
          {error && <div className="err">{error}</div>}
          {loading && <div className="muted">loading…</div>}

          {diff && (
            <>
              <div className="card-title">changed</div>
              {diff.changed.length === 0 ? (
                <div className="muted">no changed keys</div>
              ) : (
                <table className="tbl">
                  <thead>
                    <tr>
                      <th>key</th>
                      <th>A</th>
                      <th>B</th>
                    </tr>
                  </thead>
                  <tbody>
                    {diff.changed.map((c) => (
                      <tr key={c.key}>
                        <td className="mono">{c.key}</td>
                        <td className="mono">{formatJsonValue(c.a)}</td>
                        <td className="mono">{formatJsonValue(c.b)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}

              <div className="row" style={{ flexWrap: 'wrap', gap: 6 }}>
                <span className="k">only_a</span>
                {diff.only_a.length === 0 ? (
                  <span className="muted">none</span>
                ) : (
                  diff.only_a.map((k) => (
                    <span key={k} className="tag mono">
                      {k}
                    </span>
                  ))
                )}
              </div>

              <div className="row" style={{ flexWrap: 'wrap', gap: 6, borderBottom: 'none' }}>
                <span className="k">only_b</span>
                {diff.only_b.length === 0 ? (
                  <span className="muted">none</span>
                ) : (
                  diff.only_b.map((k) => (
                    <span key={k} className="tag mono">
                      {k}
                    </span>
                  ))
                )}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
