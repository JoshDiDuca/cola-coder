import { useCallback, useEffect, useState } from 'react';
import type { ConfigFile, DataSource, Job } from '../types';
import { isApiError } from '../types';
import { getConfigs, getDataSources, runAction } from '../api';
import { formatPercent } from '../format';

const DEFAULT_CONFIG = 'configs/small.yaml';

type LoadState = 'loading' | 'ready' | 'error';

/**
 * Data Collection launcher — pick a model config and launch
 * `scripts/collect_data.py` as a background job. This is the multi-source
 * (code/text/math) collection step that reads `configs/data_sources.yaml`,
 * downloads from HuggingFace, and tokenizes into .npy files.
 *
 * It is a CPU/network job (download + tokenize), NOT a GPU trainer, so there
 * is no `trainingAlive` guard. The configured source weights are shown
 * read-only so the user sees what mix will be collected before launching.
 */
export default function CollectDataPanel() {
  const [configs, setConfigs] = useState<ConfigFile[]>([]);
  const [sources, setSources] = useState<DataSource[]>([]);
  const [sourcesError, setSourcesError] = useState<string | null>(null);

  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [loadError, setLoadError] = useState<string | null>(null);

  const [selectedConfig, setSelectedConfig] = useState<string>(DEFAULT_CONFIG);

  const [pending, setPending] = useState<boolean>(false);
  const [launched, setLaunched] = useState<Job | null>(null);
  const [launchError, setLaunchError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const [cfgList, ds] = await Promise.all([getConfigs(), getDataSources()]);
        if (!active) return;

        setConfigs(cfgList);
        // Prefer the default config if present, else the first available.
        const hasDefault = cfgList.some((c) => c.path === DEFAULT_CONFIG);
        if (!hasDefault && cfgList.length > 0) {
          setSelectedConfig(cfgList[0].path);
        }

        if (isApiError(ds)) {
          setSourcesError(ds.error);
        } else {
          setSources(ds.sources);
        }

        setLoadState('ready');
      } catch (e) {
        if (!active) return;
        setLoadError(e instanceof Error ? e.message : String(e));
        setLoadState('error');
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const onCollect = useCallback(async (): Promise<void> => {
    setPending(true);
    setLaunched(null);
    setLaunchError(null);
    try {
      // collect_data.py's required flag is `--config <path>`.
      const job = await runAction('collect_data', ['--config', selectedConfig]);
      setLaunched(job);
    } catch (e) {
      setLaunchError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(false);
    }
  }, [selectedConfig]);

  return (
    <div className="card card-wide">
      <div className="card-title">Collect Data</div>

      {loadState === 'loading' && <div className="muted">loading configs…</div>}
      {loadState === 'error' && (
        <div className="err">{loadError ?? 'failed to load configs'}</div>
      )}

      {loadState === 'ready' && (
        <>
          <div className="muted" style={{ marginBottom: 8 }}>
            Multi-source collection (code + text + math): downloads from HuggingFace
            and tokenizes into .npy files. Network + CPU job — may take a while and
            needs <span className="mono">HF_TOKEN</span> for gated datasets.
          </div>

          <label className="muted" htmlFor="collect-config">
            config
          </label>
          <select
            id="collect-config"
            className="input"
            style={{ width: '100%', marginTop: 4 }}
            value={selectedConfig}
            onChange={(e) => setSelectedConfig(e.target.value)}
            disabled={pending || configs.length === 0}
          >
            {configs.length === 0 && <option value={DEFAULT_CONFIG}>{DEFAULT_CONFIG}</option>}
            {configs.map((c) => (
              <option key={c.path} value={c.path}>
                {c.rel}
              </option>
            ))}
          </select>

          <div className="card-title" style={{ marginTop: 12 }}>
            Configured source weights
          </div>
          {sourcesError !== null && <div className="err">{sourcesError}</div>}
          {sourcesError === null && sources.length === 0 && (
            <div className="muted">no data sources configured</div>
          )}
          {sources.length > 0 && (
            <table className="tbl">
              <thead>
                <tr>
                  <th>source</th>
                  <th>dataset</th>
                  <th className="right">weight</th>
                </tr>
              </thead>
              <tbody>
                {sources.map((s) => (
                  <tr key={s.name}>
                    <td>
                      {s.name}
                      {s.kind !== null && (
                        <span className="tag" style={{ marginLeft: 8 }}>
                          {s.kind}
                        </span>
                      )}
                    </td>
                    <td className="mono">{s.dataset ?? '—'}</td>
                    <td className="right mono">{formatPercent(s.weight)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <div style={{ marginTop: 12 }}>
            <button
              className="btn btn-primary"
              onClick={() => void onCollect()}
              disabled={pending}
            >
              {pending ? '…launching' : '▶ Collect Data'}
            </button>
          </div>

          {launched !== null && (
            <div className="muted mono" style={{ marginTop: 8 }}>
              launched {launched.name} ({launched.id}) — {launched.status}
            </div>
          )}
          {launchError !== null && (
            <div className="err" style={{ marginTop: 8 }}>
              {launchError}
            </div>
          )}
        </>
      )}
    </div>
  );
}
