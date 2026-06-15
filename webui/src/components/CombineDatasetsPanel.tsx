import { useCallback, useEffect, useMemo, useState } from 'react';
import type { Dataset, Job } from '../types';
import { getDatasets, runAction } from '../api';
import { formatBytes, formatInteger } from '../format';

type LoadState = 'loading' | 'ready' | 'error';

/** A selectable input row: the dataset plus its user-assigned mix weight. */
interface SelectedInput {
  dataset: Dataset;
  weight: number;
}

const DEFAULT_WEIGHT = 1.0;
const ACTION_KEY = 'combine_datasets';

/**
 * Combine Datasets — launcher panel.
 *
 * Pick 2+ prepared .npy datasets, assign each a mix weight, set an output path,
 * and launch `scripts/combine_datasets.py` as a background job via the
 * `combine_datasets` allow-listed action.
 *
 * The script's non-interactive interface is `--datasets PATH[:WEIGHT] ...
 * --output PATH` (weights are a per-path `:weight` suffix, normalised by the
 * script; the CLI path performs weighted sampling — MinHash/near-dup dedup is
 * only available in the script's interactive TUI, so it is not exposed here).
 */
export default function CombineDatasetsPanel() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [loadError, setLoadError] = useState<string | null>(null);

  // Keyed by dataset path → its weight (presence in the map == selected).
  const [selected, setSelected] = useState<Map<string, number>>(new Map());
  const [output, setOutput] = useState<string>('');

  const [pending, setPending] = useState<boolean>(false);
  const [launched, setLaunched] = useState<Job | null>(null);
  const [launchError, setLaunchError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const next = await getDatasets();
        if (!active) return;
        setDatasets(next);
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

  // Only .npy datasets are combinable (the script loads 2-D token arrays).
  const npyDatasets = useMemo<Dataset[]>(
    () => datasets.filter((ds) => ds.kind === 'npy'),
    [datasets],
  );

  const toggle = useCallback((ds: Dataset): void => {
    setSelected((prev) => {
      const next = new Map(prev);
      if (next.has(ds.path)) {
        next.delete(ds.path);
      } else {
        next.set(ds.path, DEFAULT_WEIGHT);
      }
      return next;
    });
    setLaunched(null);
    setLaunchError(null);
  }, []);

  const setWeight = useCallback((path: string, weight: number): void => {
    setSelected((prev) => {
      if (!prev.has(path)) return prev;
      const next = new Map(prev);
      next.set(path, weight);
      return next;
    });
  }, []);

  const selectedInputs = useMemo<SelectedInput[]>(() => {
    const rows: SelectedInput[] = [];
    for (const ds of npyDatasets) {
      const weight = selected.get(ds.path);
      if (weight !== undefined) rows.push({ dataset: ds, weight });
    }
    return rows;
  }, [npyDatasets, selected]);

  const canLaunch: boolean =
    selectedInputs.length >= 2 && output.trim().length > 0 && !pending;

  const onLaunch = useCallback(async (): Promise<void> => {
    if (selectedInputs.length < 2 || output.trim().length === 0) return;
    setPending(true);
    setLaunched(null);
    setLaunchError(null);

    // Build: --datasets path1:w1 path2:w2 ... --output out
    // Weight suffix is parsed on the script's LAST colon, so Windows drive
    // paths (C:\...) survive the round-trip.
    const datasetArgs: string[] = selectedInputs.map(
      ({ dataset, weight }) => `${dataset.path}:${weight}`,
    );
    const args: string[] = ['--datasets', ...datasetArgs, '--output', output.trim()];

    try {
      const job = await runAction(ACTION_KEY, args);
      setLaunched(job);
    } catch (e) {
      setLaunchError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(false);
    }
  }, [selectedInputs, output]);

  return (
    <div className="card card-wide">
      <div className="card-title">Combine Datasets</div>
      <div className="muted" style={{ marginBottom: 8 }}>
        Pick 2+ prepared .npy datasets, set per-dataset weights and an output
        path, then launch a weighted mix as a background job. CPU job — runs
        alongside training.
      </div>

      {loadState === 'loading' && <div className="muted">loading datasets…</div>}
      {loadState === 'error' && (
        <div className="err">{loadError ?? 'failed to load datasets'}</div>
      )}

      {loadState === 'ready' && npyDatasets.length === 0 && (
        <div className="muted">no .npy datasets in data/</div>
      )}

      {loadState === 'ready' && npyDatasets.length > 0 && (
        <>
          <table className="tbl">
            <thead>
              <tr>
                <th>pick</th>
                <th>name</th>
                <th className="right">samples</th>
                <th className="right">size</th>
                <th className="right">weight</th>
              </tr>
            </thead>
            <tbody>
              {npyDatasets.map((ds) => {
                const weight = selected.get(ds.path);
                const isSelected = weight !== undefined;
                return (
                  <tr key={ds.path}>
                    <td>
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => toggle(ds)}
                        disabled={pending}
                        aria-label={`select ${ds.name}`}
                      />
                    </td>
                    <td>{ds.name}</td>
                    <td className="right mono">{formatInteger(ds.num_samples)}</td>
                    <td className="right mono">{formatBytes(ds.size_bytes)}</td>
                    <td className="right">
                      <input
                        className="input mono"
                        type="number"
                        min={0}
                        step={0.1}
                        style={{ width: 72, textAlign: 'right' }}
                        value={isSelected ? weight : ''}
                        onChange={(e) =>
                          setWeight(ds.path, Number(e.target.value))
                        }
                        disabled={!isSelected || pending}
                        aria-label={`weight for ${ds.name}`}
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <div style={{ marginTop: 12 }}>
            <label className="muted" htmlFor="combine-output">
              output path (.npy)
            </label>
            <input
              id="combine-output"
              className="input mono"
              style={{ width: '100%', marginTop: 4 }}
              value={output}
              onChange={(e) => setOutput(e.target.value)}
              placeholder="data/processed/combined.npy"
              spellCheck={false}
              disabled={pending}
            />
          </div>

          {selectedInputs.length < 2 && (
            <div className="muted" style={{ marginTop: 8 }}>
              select at least 2 datasets to combine
            </div>
          )}

          <div style={{ marginTop: 8 }}>
            <button
              className="btn btn-primary"
              onClick={() => void onLaunch()}
              disabled={!canLaunch}
            >
              {pending ? '…launching' : '▶ Combine'}
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
