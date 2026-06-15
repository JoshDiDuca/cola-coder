import { useCallback, useEffect, useState } from 'react';
import type { ConfigFile, VramEstimate, VramComponent } from '../types';
import { isApiError } from '../types';
import { getConfigs, getVramEstimate } from '../api';
import { formatInteger, formatFloat } from '../format';

function fitsBadgeClass(fits: boolean): string {
  return fits ? 'tag done' : 'tag failed';
}

function ComponentRow({ component }: { component: VramComponent }) {
  return (
    <div className="row">
      <span className="k">{component.name}</span>
      <span className="v mono">{formatFloat(component.mb)} MB</span>
    </div>
  );
}

export default function VramEstimatePanel() {
  const [configs, setConfigs] = useState<ConfigFile[]>([]);
  const [selected, setSelected] = useState<string>('');
  const [estimate, setEstimate] = useState<VramEstimate | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch the config list once; default to the first config.
  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const next = await getConfigs();
        if (!active) return;
        setConfigs(next);
        if (next.length > 0) setSelected(next[0].name);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const load = useCallback(async (config: string): Promise<void> => {
    if (config === '') return;
    setError(null);
    setLoading(true);
    try {
      const resp = await getVramEstimate(config);
      if (isApiError(resp)) {
        setError(resp.error);
        setEstimate(null);
      } else {
        setEstimate(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setEstimate(null);
    } finally {
      setLoading(false);
    }
  }, []);

  // Estimate whenever the selected config changes.
  useEffect(() => {
    void load(selected);
  }, [selected, load]);

  return (
    <div className="card">
      <div className="row">
        <div className="card-title">VRAM Estimate</div>
        <div className="row">
          <select
            className="select"
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
            disabled={configs.length === 0}
          >
            {configs.length === 0 && <option value="">no configs</option>}
            {configs.map((cfg) => (
              <option key={cfg.path} value={cfg.name}>
                {cfg.name}
              </option>
            ))}
          </select>
          <button className="btn" onClick={() => void load(selected)} disabled={loading}>
            refresh
          </button>
        </div>
      </div>

      {error && <div className="err">{error}</div>}
      {loading && !estimate && <div className="muted">loading…</div>}

      {estimate && (
        <>
          <div className="row">
            <span className={fitsBadgeClass(estimate.fits)}>
              {estimate.fits ? 'fits' : 'over budget'}
            </span>
            <span className="v muted">
              {formatFloat(estimate.total_mb)} / {formatFloat(estimate.budget_mb)} MB
            </span>
          </div>

          <div className="row">
            <span className="k">config</span>
            <span className="v mono">{estimate.config}</span>
          </div>
          <div className="row">
            <span className="k">params</span>
            <span className="v mono">{formatFloat(estimate.params_millions, 1)}M</span>
          </div>
          <div className="row">
            <span className="k">precision</span>
            <span className="v mono">{estimate.precision}</span>
          </div>
          <div className="row">
            <span className="k">batch / seq_len</span>
            <span className="v mono">
              {formatInteger(estimate.batch_size)} / {formatInteger(estimate.seq_len)}
            </span>
          </div>

          <div className="tbl">
            {estimate.components.length === 0 ? (
              <div className="muted">no components</div>
            ) : (
              estimate.components.map((component) => (
                <ComponentRow key={component.name} component={component} />
              ))
            )}
            <div className="row">
              <span className="k">
                <strong>training total</strong>
              </span>
              <span className="v mono">
                <strong>{formatFloat(estimate.total_mb)} MB</strong>
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
