import { useCallback, useEffect, useMemo, useState } from 'react';
import { isApiError, type ScoringConfig, type ScorerConfigEntry } from '../types';
import { getScoringConfig } from '../api';
import { formatFloat, formatInteger } from '../format';

export default function ScoringConfigPanel() {
  const [config, setConfig] = useState<ScoringConfig | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const c = await getScoringConfig();
      if (isApiError(c)) {
        setConfig(null);
        setError(c.error);
      } else {
        setConfig(c);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const scorers = useMemo<ScorerConfigEntry[]>(
    () => (config ? config.scorers : []),
    [config],
  );

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Scoring Config</div>
        <button className="btn" onClick={() => void load()}>
          Refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {config && (
        <>
          <div className="row">
            <span className="muted mono">
              {formatInteger(config.count)} scorers · {formatInteger(config.enabled_count)} enabled
              {config.curriculum ? ` · curriculum: ${config.curriculum}` : ''}
            </span>
            <span className="muted mono">{config.path}</span>
          </div>

          {scorers.length === 0 ? (
            <div className="muted">no scorers configured</div>
          ) : (
            <table className="tbl">
              <thead>
                <tr>
                  <th>scorer</th>
                  <th>enabled</th>
                  <th>available</th>
                  <th>weight</th>
                  <th>purpose</th>
                </tr>
              </thead>
              <tbody>
                {scorers.map((s) => (
                  <tr key={s.name}>
                    <td className="mono">{s.name}</td>
                    <td>
                      <span
                        className={`dot ${s.enabled ? 'live' : 'dead'}`}
                        title={s.enabled ? 'enabled' : 'disabled'}
                      />
                    </td>
                    <td>
                      <span
                        className={`dot ${s.available ? 'live' : 'dead'}`}
                        title={s.available ? 'deps present — can run' : 'unavailable (missing deps)'}
                      />
                    </td>
                    <td className="mono">{formatFloat(s.weight)}</td>
                    <td className="muted">{s.purpose}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}

      {!config && !error && <div className="muted">no scoring config</div>}
    </div>
  );
}
