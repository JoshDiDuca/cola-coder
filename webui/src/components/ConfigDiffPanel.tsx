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

  const sameSelected = aPath !== '' && bPath !== '' && aPath === bPath;
  const noDifferences =
    diff !== null &&
    diff.changed.length === 0 &&
    diff.only_a.length === 0 &&
    diff.only_b.length === 0;

  return (
    <div className="card card-wide">
      <div className="card-title">Config Diff</div>

      {configs.length < 2 && !error ? (
        <div className="muted">need two distinct configs</div>
      ) : (
        <>
          <div className="cfg-pick">
            <select
              className="select"
              value={aPath}
              onChange={(e) => setAPath(e.target.value)}
            >
              <option value="">A: select config…</option>
              {configs.map((cfg) => (
                <option key={cfg.path} value={cfg.path}>
                  {cfg.rel}
                </option>
              ))}
            </select>
            <span className="cfg-pick-vs muted mono">vs</span>
            <select
              className="select"
              value={bPath}
              onChange={(e) => setBPath(e.target.value)}
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
              {loading ? 'comparing…' : 'Compare'}
            </button>
          </div>

          {sameSelected && <div className="muted">pick two distinct configs</div>}
          {error && <div className="err">{error}</div>}

          {diff && noDifferences && <div className="muted">configs are identical</div>}

          {diff && !noDifferences && (
            <div className="diff-list scroll">
              {diff.changed.map((c) => (
                <div key={`chg-${c.key}`} className="diff-row diff-chg">
                  <span className="diff-mark mono">~</span>
                  <span className="diff-key mono">{c.key}</span>
                  <span className="diff-val">
                    <span className="diff-a mono">{formatJsonValue(c.a)}</span>
                    <span className="diff-arrow muted mono">→</span>
                    <span className="diff-b mono">{formatJsonValue(c.b)}</span>
                  </span>
                </div>
              ))}

              {diff.only_a.map((k) => (
                <div key={`a-${k}`} className="diff-row diff-del">
                  <span className="diff-mark mono">−</span>
                  <span className="diff-key mono">{k}</span>
                  <span className="diff-val muted">only in A</span>
                </div>
              ))}

              {diff.only_b.map((k) => (
                <div key={`b-${k}`} className="diff-row diff-add">
                  <span className="diff-mark mono">+</span>
                  <span className="diff-key mono">{k}</span>
                  <span className="diff-val muted">only in B</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
