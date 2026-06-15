import { useCallback, useEffect, useState } from 'react';
import type { ConfigFile, Job } from '../types';
import { getConfigs, runAction } from '../api';

const DEFAULT_CONFIG = 'configs/small.yaml';

type LoadState = 'loading' | 'ready' | 'error';

/** Quality-filter aggressiveness. Maps 1:1 to prepare_data.py's mutually
 *  exclusive flag group: 'default' passes no flag (conservative filtering),
 *  'none' → --no-filter, 'strict' → --filter-strict. */
type FilterMode = 'default' | 'none' | 'strict';

/** Dedup mode. prepare_data.py supports none/exact/minhash/semantic with exact
 *  as the default. This launcher exposes the two common choices: 'exact'
 *  (default, passes no flag) and 'none' (→ --no-dedup). */
type DedupMode = 'exact' | 'none';

/**
 * Prepare Data launcher — pick a model config plus the real data-prep options
 * and launch `scripts/prepare_data.py` as a background job (UI-017 wizard gap).
 *
 * This is the tokenize/filter/dedup/score step that turns collected source code
 * into chunked `.npy` arrays ready for training. It is a CPU job (no GPU /
 * `trainingAlive` guard); its output is reusable and only needs re-running when
 * the tokenizer, seq_len, dataset, languages, filter, or dedup mode changes.
 *
 * Args are assembled from ONLY the real `prepare_data.py` flags:
 *   --config <path>           always
 *   --score                   when "score quality weights" is checked
 *   --no-filter | --filter-strict   from the filter mode (default = no flag)
 *   --no-dedup                when dedup mode is "none" (exact = default, no flag)
 *   --workers <n>             when a positive worker count is entered
 */
export default function PrepareDataPanel() {
  const [configs, setConfigs] = useState<ConfigFile[]>([]);
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [loadError, setLoadError] = useState<string | null>(null);

  const [selectedConfig, setSelectedConfig] = useState<string>(DEFAULT_CONFIG);
  const [score, setScore] = useState<boolean>(false);
  const [filterMode, setFilterMode] = useState<FilterMode>('default');
  const [dedupMode, setDedupMode] = useState<DedupMode>('exact');
  const [workers, setWorkers] = useState<string>('');

  const [pending, setPending] = useState<boolean>(false);
  const [launched, setLaunched] = useState<Job | null>(null);
  const [launchError, setLaunchError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const cfgList = await getConfigs();
        if (!active) return;
        setConfigs(cfgList);
        const hasDefault = cfgList.some((c) => c.path === DEFAULT_CONFIG);
        if (!hasDefault && cfgList.length > 0) {
          setSelectedConfig(cfgList[0].path);
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

  const buildArgs = useCallback((): string[] => {
    const args: string[] = ['--config', selectedConfig];
    if (score) args.push('--score');
    if (filterMode === 'none') args.push('--no-filter');
    else if (filterMode === 'strict') args.push('--filter-strict');
    if (dedupMode === 'none') args.push('--no-dedup');
    const n = Number.parseInt(workers, 10);
    if (Number.isFinite(n) && n > 0) {
      args.push('--workers', String(n));
    }
    return args;
  }, [selectedConfig, score, filterMode, dedupMode, workers]);

  const onPrepare = useCallback(async (): Promise<void> => {
    setPending(true);
    setLaunched(null);
    setLaunchError(null);
    try {
      const job = await runAction('prepare_data', buildArgs());
      setLaunched(job);
    } catch (e) {
      setLaunchError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(false);
    }
  }, [buildArgs]);

  return (
    <div className="card card-wide">
      <div className="card-title">Prepare Data</div>

      {loadState === 'loading' && <div className="muted">loading configs…</div>}
      {loadState === 'error' && (
        <div className="err">{loadError ?? 'failed to load configs'}</div>
      )}

      {loadState === 'ready' && (
        <>
          <div className="muted" style={{ marginBottom: 8 }}>
            Tokenizes and quality-filters collected code into chunked{' '}
            <span className="mono">.npy</span> arrays ready for training. CPU job —
            can take a while. Output is reusable; only re-run when the tokenizer,
            seq_len, dataset, languages, filter, or dedup mode changes.
          </div>

          <label className="muted" htmlFor="prepare-config">
            config
          </label>
          <select
            id="prepare-config"
            className="input"
            style={{ width: '100%', marginTop: 4 }}
            value={selectedConfig}
            onChange={(e) => setSelectedConfig(e.target.value)}
            disabled={pending || configs.length === 0}
          >
            {configs.length === 0 && (
              <option value={DEFAULT_CONFIG}>{DEFAULT_CONFIG}</option>
            )}
            {configs.map((c) => (
              <option key={c.path} value={c.path}>
                {c.rel}
              </option>
            ))}
          </select>

          <label className="muted" htmlFor="prepare-filter" style={{ marginTop: 12, display: 'block' }}>
            quality filter
          </label>
          <select
            id="prepare-filter"
            className="input"
            style={{ width: '100%', marginTop: 4 }}
            value={filterMode}
            onChange={(e) => setFilterMode(e.target.value as FilterMode)}
            disabled={pending}
          >
            <option value="default">conservative (default)</option>
            <option value="strict">strict (--filter-strict, 60-75% rejected)</option>
            <option value="none">disabled (--no-filter)</option>
          </select>

          <label className="muted" htmlFor="prepare-dedup" style={{ marginTop: 12, display: 'block' }}>
            deduplication
          </label>
          <select
            id="prepare-dedup"
            className="input"
            style={{ width: '100%', marginTop: 4 }}
            value={dedupMode}
            onChange={(e) => setDedupMode(e.target.value as DedupMode)}
            disabled={pending}
          >
            <option value="exact">exact (default)</option>
            <option value="none">none (--no-dedup, keep all chunks)</option>
          </select>

          <label className="muted" htmlFor="prepare-workers" style={{ marginTop: 12, display: 'block' }}>
            workers (blank = all CPU cores, capped at 16)
          </label>
          <input
            id="prepare-workers"
            className="input"
            type="number"
            min={1}
            style={{ width: '100%', marginTop: 4 }}
            value={workers}
            onChange={(e) => setWorkers(e.target.value)}
            placeholder="auto"
            disabled={pending}
          />

          <label className="muted" style={{ marginTop: 12, display: 'block' }}>
            <input
              type="checkbox"
              checked={score}
              onChange={(e) => setScore(e.target.checked)}
              disabled={pending}
              style={{ marginRight: 6 }}
            />
            score quality weights (--score, adds ~30% time)
          </label>

          <div style={{ marginTop: 12 }}>
            <button
              className="btn btn-primary"
              onClick={() => void onPrepare()}
              disabled={pending}
            >
              {pending ? '…launching' : '▶ Prepare Data'}
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
