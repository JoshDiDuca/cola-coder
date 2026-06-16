import { useCallback, useEffect, useState } from 'react';
import type { ConfigFile, ConfigSummary, ConfigGroup } from '../types';
import { isApiError } from '../types';
import { getConfigs, getConfigSummary } from '../api';
import EmptyState from './EmptyState';
import LoadingSpinner from './LoadingSpinner';

/**
 * Pick the default config path: prefer the live "small_react_best" run if it
 * is present, otherwise fall back to the first available config.
 */
function pickDefaultPath(configs: ConfigFile[]): string {
  if (configs.length === 0) return '';
  const live = configs.find((cfg) => cfg.rel.includes('small_react_best'));
  return live !== undefined ? live.path : configs[0].path;
}

function SummaryGroup({ group }: { group: ConfigGroup }): JSX.Element {
  return (
    <div className="tbl">
      <div className="card-title">{group.title}</div>
      {group.items.map((item) => (
        <div key={item.label} className="row">
          <span className="k">{item.label}</span>
          <span className="v mono">{item.value}</span>
        </div>
      ))}
    </div>
  );
}

export default function ConfigSummaryPanel(): JSX.Element {
  const [configs, setConfigs] = useState<ConfigFile[]>([]);
  const [selected, setSelected] = useState<string>('');
  const [summary, setSummary] = useState<ConfigSummary | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch the config list once; default to the live run (or first config).
  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const next = await getConfigs();
        if (!active) return;
        setConfigs(next);
        setSelected(pickDefaultPath(next));
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const load = useCallback(async (path: string): Promise<void> => {
    if (path === '') return;
    setError(null);
    setLoading(true);
    try {
      const resp = await getConfigSummary(path);
      if (isApiError(resp)) {
        setError(resp.error);
        setSummary(null);
      } else {
        setSummary(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSummary(null);
    } finally {
      setLoading(false);
    }
  }, []);

  // Load the summary whenever the selected config changes.
  useEffect(() => {
    void load(selected);
  }, [selected, load]);

  return (
    <>
      <div className="row">
        <div className="card-title">Config Summary</div>
        <select
          className="select"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          disabled={configs.length === 0}
        >
          {configs.length === 0 && <option value="">no configs</option>}
          {configs.map((cfg) => (
            <option key={cfg.path} value={cfg.path}>
              {cfg.rel}
            </option>
          ))}
        </select>
      </div>

      <div className="muted">Key hyperparameters at a glance (read-only).</div>

      {error && <div className="err">{error}</div>}
      {loading && summary === null && <LoadingSpinner label="Loading summary…" />}

      {summary !== null && !summary.exists && (
        <EmptyState title="Config not found" hint={summary.path} />
      )}

      {summary !== null &&
        summary.exists &&
        summary.groups.map((group) => <SummaryGroup key={group.title} group={group} />)}
    </>
  );
}
