import { useCallback, useEffect, useMemo, useState } from 'react';
import type {
  ConfigFile,
  Dataset,
  DataStats,
  Job,
  JsonValue,
  Preview,
  ScoreSummary,
} from '../../types';
import { isApiError } from '../../types';
import {
  getConfigs,
  getDataStats,
  getDatasets,
  getPreview,
  getScores,
  runAction,
} from '../../api';
import {
  formatBytes,
  formatFloat,
  formatInteger,
  formatJsonValue,
} from '../../format';
import MasterDetail, { type MasterItem } from '../MasterDetail';
import LoadingSpinner from '../LoadingSpinner';
import EmptyState from '../EmptyState';

// ─────────────────────────────────────────────────────────────────────────────
// DataScreen — ONE coherent master-detail "Data" screen replacing the old grid
// of ~12 data cards. Left list = datasets; detail = stats + preview + scores for
// the selected dataset. A compact launcher (Collect / Prepare) lives in the list
// aside so the screen is action-capable without giant cards.
//
// All API calls and flows are reused verbatim from the existing panels:
//   - getDatasets / getPreview / getScores  → DatasetsPanel
//   - getDataStats                          → DataStatsPanel
//   - getConfigs + runAction('collect_data'|'prepare_data') → Collect/PrepareDataPanel
// ─────────────────────────────────────────────────────────────────────────────

const PREVIEW_N = 12;
const PREVIEW_MAX_CHARS = 4000;
const DEFAULT_CONFIG = 'configs/small.yaml';

/** The prepared-training array `getDataStats()` reports on by default. Only this
 *  dataset gets the extra token-level stat tiles. */
const TRAIN_DATA_BASENAME = 'train_data.npy';

/** Which detail sub-view is showing. */
type DetailTab = 'preview' | 'scores';

/** Which launcher mini-form is open in the list aside (null = none). */
type LauncherMode = 'collect' | 'prepare' | null;

/** prepare_data.py mutually-exclusive filter group (matches PrepareDataPanel). */
type FilterMode = 'default' | 'none' | 'strict';

/** A dataset's sidecar weights path (DatasetsPanel convention). */
function weightsPathFor(datasetPath: string): string {
  return datasetPath.replace(/\.npy$/, '.weights.npy');
}

/** Render preview rows the same way DatasetsPanel does. */
function formatPreview(p: Preview): string {
  if (p.error) return `error: ${p.error}`;
  const rows: JsonValue[] = p.preview ?? [];
  const text = rows.map((row) => formatJsonValue(row)).join('\n');
  return text.length > PREVIEW_MAX_CHARS
    ? `${text.slice(0, PREVIEW_MAX_CHARS)}\n…(truncated)`
    : text;
}

interface StatTile {
  label: string;
  value: string;
}

/** Token-level tiles for the prepared train array (matches DataStatsPanel). */
function buildStatsTiles(stats: DataStats): StatTile[] {
  return [
    { label: 'file size', value: `${formatFloat(stats.file_size_mb, 1)} MB` },
    { label: 'shape', value: stats.shape.join(' × ') },
    { label: 'total tokens', value: formatInteger(stats.total_tokens) },
    {
      label: 'token range',
      value: `${formatInteger(stats.token_min)}–${formatInteger(stats.token_max)}`,
    },
    { label: 'token mean', value: formatFloat(stats.token_mean) },
    { label: 'est. unique', value: formatInteger(stats.est_unique_tokens ?? null) },
  ];
}

// ── Compact launcher (Collect / Prepare) — lives in the list aside ───────────

interface LauncherProps {
  configs: ConfigFile[];
  selectedConfig: string;
  onSelectConfig: (path: string) => void;
}

/** Collect launcher: `collect_data.py --config <path>` (CollectDataPanel flow). */
function CollectForm({ configs, selectedConfig, onSelectConfig }: LauncherProps): JSX.Element {
  const [pending, setPending] = useState<boolean>(false);
  const [launched, setLaunched] = useState<Job | null>(null);
  const [launchError, setLaunchError] = useState<string | null>(null);

  const onCollect = useCallback(async (): Promise<void> => {
    setPending(true);
    setLaunched(null);
    setLaunchError(null);
    try {
      const job = await runAction('collect_data', ['--config', selectedConfig]);
      setLaunched(job);
    } catch (e) {
      setLaunchError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(false);
    }
  }, [selectedConfig]);

  return (
    <div className="data-launch-form">
      <div className="muted data-launch-hint">
        Multi-source collection (code + text + math) into <span className="mono">.npy</span>.
        Network + CPU job; needs <span className="mono">HF_TOKEN</span> for gated datasets.
      </div>
      <label className="muted" htmlFor="data-collect-config">config</label>
      <select
        id="data-collect-config"
        className="select"
        value={selectedConfig}
        onChange={(e) => onSelectConfig(e.target.value)}
        disabled={pending || configs.length === 0}
      >
        {configs.length === 0 && <option value={DEFAULT_CONFIG}>{DEFAULT_CONFIG}</option>}
        {configs.map((c) => (
          <option key={c.path} value={c.path}>{c.rel}</option>
        ))}
      </select>
      <button className="btn btn-primary" onClick={() => void onCollect()} disabled={pending}>
        {pending ? '…launching' : '▶ Collect'}
      </button>
      {launched !== null && (
        <div className="muted mono data-launch-note">
          launched {launched.name} ({launched.id}) — {launched.status}
        </div>
      )}
      {launchError !== null && <div className="err">{launchError}</div>}
    </div>
  );
}

/** Prepare launcher: `prepare_data.py --config [--score] [--no-filter|--filter-strict]`
 *  (subset of PrepareDataPanel's real flags). */
function PrepareForm({ configs, selectedConfig, onSelectConfig }: LauncherProps): JSX.Element {
  const [score, setScore] = useState<boolean>(false);
  const [filterMode, setFilterMode] = useState<FilterMode>('default');
  const [pending, setPending] = useState<boolean>(false);
  const [launched, setLaunched] = useState<Job | null>(null);
  const [launchError, setLaunchError] = useState<string | null>(null);

  const buildArgs = useCallback((): string[] => {
    const args: string[] = ['--config', selectedConfig];
    if (score) args.push('--score');
    if (filterMode === 'none') args.push('--no-filter');
    else if (filterMode === 'strict') args.push('--filter-strict');
    return args;
  }, [selectedConfig, score, filterMode]);

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
    <div className="data-launch-form">
      <div className="muted data-launch-hint">
        Tokenize + quality-filter collected code into chunked{' '}
        <span className="mono">.npy</span>. CPU job; reusable output.
      </div>
      <label className="muted" htmlFor="data-prepare-config">config</label>
      <select
        id="data-prepare-config"
        className="select"
        value={selectedConfig}
        onChange={(e) => onSelectConfig(e.target.value)}
        disabled={pending || configs.length === 0}
      >
        {configs.length === 0 && <option value={DEFAULT_CONFIG}>{DEFAULT_CONFIG}</option>}
        {configs.map((c) => (
          <option key={c.path} value={c.path}>{c.rel}</option>
        ))}
      </select>
      <label className="muted" htmlFor="data-prepare-filter">quality filter</label>
      <select
        id="data-prepare-filter"
        className="select"
        value={filterMode}
        onChange={(e) => setFilterMode(e.target.value as FilterMode)}
        disabled={pending}
      >
        <option value="default">conservative (default)</option>
        <option value="strict">strict (--filter-strict)</option>
        <option value="none">disabled (--no-filter)</option>
      </select>
      <label className="muted data-launch-check">
        <input
          type="checkbox"
          checked={score}
          onChange={(e) => setScore(e.target.checked)}
          disabled={pending}
        />
        score quality weights (--score)
      </label>
      <button className="btn btn-primary" onClick={() => void onPrepare()} disabled={pending}>
        {pending ? '…launching' : '▶ Prepare'}
      </button>
      {launched !== null && (
        <div className="muted mono data-launch-note">
          launched {launched.name} ({launched.id}) — {launched.status}
        </div>
      )}
      {launchError !== null && <div className="err">{launchError}</div>}
    </div>
  );
}

// ── Detail pane for the selected dataset ─────────────────────────────────────

interface DetailProps {
  dataset: Dataset;
}

function DatasetDetail({ dataset }: DetailProps): JSX.Element {
  const [tab, setTab] = useState<DetailTab>('preview');

  const [previewText, setPreviewText] = useState<string>('');
  const [previewLoading, setPreviewLoading] = useState<boolean>(false);

  const [scores, setScores] = useState<ScoreSummary | null>(null);
  const [scoreError, setScoreError] = useState<string | null>(null);
  const [scoreLoading, setScoreLoading] = useState<boolean>(false);

  const [stats, setStats] = useState<DataStats | null>(null);
  const [statsError, setStatsError] = useState<string | null>(null);

  const isTrainData = dataset.path.replace(/\\/g, '/').endsWith(TRAIN_DATA_BASENAME);

  // Reset to the preview tab whenever the selected dataset changes.
  useEffect(() => {
    setTab('preview');
  }, [dataset.path]);

  // Preview (always) — DatasetsPanel.onView flow.
  useEffect(() => {
    let active = true;
    setPreviewLoading(true);
    setPreviewText('');
    void (async () => {
      try {
        const p = await getPreview(dataset.path, PREVIEW_N);
        if (active) setPreviewText(formatPreview(p));
      } catch (e) {
        if (active) setPreviewText(`error: ${e instanceof Error ? e.message : String(e)}`);
      } finally {
        if (active) setPreviewLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, [dataset.path]);

  // Scores (only when the dataset has weights) — DatasetsPanel.onView flow.
  useEffect(() => {
    let active = true;
    setScores(null);
    setScoreError(null);
    if (!dataset.has_weights) return;
    setScoreLoading(true);
    void (async () => {
      try {
        const s = await getScores(weightsPathFor(dataset.path));
        if (active) setScores(s);
      } catch (e) {
        if (active) setScoreError(e instanceof Error ? e.message : String(e));
      } finally {
        if (active) setScoreLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, [dataset.path, dataset.has_weights]);

  // Token-level stats — only for the prepared train array (DataStatsPanel flow).
  useEffect(() => {
    let active = true;
    setStats(null);
    setStatsError(null);
    if (!isTrainData) return;
    void (async () => {
      try {
        const resp = await getDataStats(dataset.path);
        if (!active) return;
        if (isApiError(resp)) setStatsError(resp.error);
        else setStats(resp);
      } catch (e) {
        if (active) setStatsError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, [dataset.path, isTrainData]);

  const histMax = useMemo<number>(() => {
    if (!scores || scores.histogram.length === 0) return 0;
    return Math.max(...scores.histogram);
  }, [scores]);

  return (
    <>
      <div className="md-detail-head">
        <div className="data-detail-id">
          <h2 className="md-detail-title">{dataset.name}</h2>
          <div className="muted mono data-detail-path" title={dataset.path}>{dataset.path}</div>
        </div>
        <div className="md-toolbar">
          <button
            className={`btn${tab === 'preview' ? ' btn-primary' : ''}`}
            onClick={() => setTab('preview')}
          >
            Preview
          </button>
          {dataset.has_weights && (
            <button
              className={`btn${tab === 'scores' ? ' btn-primary' : ''}`}
              onClick={() => setTab('scores')}
            >
              Scores
            </button>
          )}
        </div>
      </div>

      {/* Always-visible summary tiles. */}
      <div className="stat-tiles">
        <div className="stat-tile">
          <div className="stat-tile-label">kind</div>
          <div className="stat-tile-value mono">{dataset.kind}</div>
        </div>
        <div className="stat-tile">
          <div className="stat-tile-label">samples</div>
          <div className="stat-tile-value mono">{formatInteger(dataset.num_samples)}</div>
        </div>
        <div className="stat-tile">
          <div className="stat-tile-label">size</div>
          <div className="stat-tile-value mono">{formatBytes(dataset.size_bytes)}</div>
        </div>
        <div className="stat-tile">
          <div className="stat-tile-label">weights</div>
          <div className="stat-tile-value mono">{dataset.has_weights ? 'yes' : 'no'}</div>
        </div>
      </div>

      {/* Token-level stats for the prepared train array only. */}
      {isTrainData && (
        <div className="data-section">
          <div className="card-title">Token statistics</div>
          {statsError && <div className="err">{statsError}</div>}
          {!stats && !statsError && <LoadingSpinner label="Loading token statistics…" />}
          {stats && (
            <div className="stat-tiles">
              {buildStatsTiles(stats).map((tile) => (
                <div className="stat-tile" key={tile.label}>
                  <div className="stat-tile-label">{tile.label}</div>
                  <div className="stat-tile-value mono">{tile.value}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Preview tab. */}
      {tab === 'preview' && (
        <div className="data-section">
          <div className="card-title">
            Preview · <span className="mono">first {PREVIEW_N} rows</span>
          </div>
          {previewLoading ? (
            <LoadingSpinner label="Loading preview…" />
          ) : (
            <pre className="pre scroll">{previewText}</pre>
          )}
        </div>
      )}

      {/* Scores tab (weighted datasets only). */}
      {tab === 'scores' && dataset.has_weights && (
        <div className="data-section">
          <div className="card-title">Quality weights</div>
          {scoreError && <div className="err">{scoreError}</div>}
          {scoreLoading && !scores && <LoadingSpinner label="Loading scores…" />}
          {scores && (
            <>
              <div className="hist">
                {scores.histogram.map((count, i) => (
                  <div
                    key={i}
                    className="b"
                    title={`${scores.bins[i] ?? ''}: ${count}`}
                    style={{ height: `${histMax > 0 ? (count / histMax) * 100 : 0}%` }}
                  />
                ))}
              </div>
              <div className="stat-tiles">
                <div className="stat-tile">
                  <div className="stat-tile-label">n</div>
                  <div className="stat-tile-value mono">{formatInteger(scores.n)}</div>
                </div>
                <div className="stat-tile">
                  <div className="stat-tile-label">mean</div>
                  <div className="stat-tile-value mono">{formatFloat(scores.mean, 3)}</div>
                </div>
                <div className="stat-tile">
                  <div className="stat-tile-label">min</div>
                  <div className="stat-tile-value mono">{formatFloat(scores.min, 3)}</div>
                </div>
                <div className="stat-tile">
                  <div className="stat-tile-label">max</div>
                  <div className="stat-tile-value mono">{formatFloat(scores.max, 3)}</div>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
}

// ── Screen ───────────────────────────────────────────────────────────────────

export default function DataScreen(): JSX.Element {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);

  const [configs, setConfigs] = useState<ConfigFile[]>([]);
  const [selectedConfig, setSelectedConfig] = useState<string>(DEFAULT_CONFIG);
  const [launcher, setLauncher] = useState<LauncherMode>(null);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const [next, cfgList] = await Promise.all([getDatasets(), getConfigs()]);
        if (!active) return;
        setDatasets(next);
        if (next.length > 0) setSelectedPath((cur) => cur ?? next[0].path);
        setConfigs(cfgList);
        const hasDefault = cfgList.some((c) => c.path === DEFAULT_CONFIG);
        if (!hasDefault && cfgList.length > 0) setSelectedConfig(cfgList[0].path);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const items = useMemo<MasterItem[]>(
    () =>
      datasets.map((ds) => ({
        id: ds.path,
        title: ds.name,
        subtitle: ds.kind,
        meta: formatBytes(ds.size_bytes),
        badge: ds.has_weights ? <span className="tag done">weighted</span> : null,
      })),
    [datasets],
  );

  const selected = useMemo<Dataset | null>(
    () => datasets.find((d) => d.path === selectedPath) ?? null,
    [datasets, selectedPath],
  );

  const toggleLauncher = useCallback((mode: 'collect' | 'prepare'): void => {
    setLauncher((cur) => (cur === mode ? null : mode));
  }, []);

  const listAside = (
    <div className="data-launch-buttons">
      <button
        className={`btn${launcher === 'collect' ? ' btn-primary' : ''}`}
        onClick={() => toggleLauncher('collect')}
      >
        + Collect
      </button>
      <button
        className={`btn${launcher === 'prepare' ? ' btn-primary' : ''}`}
        onClick={() => toggleLauncher('prepare')}
      >
        + Prepare
      </button>
    </div>
  );

  const launcherForm =
    launcher === 'collect' ? (
      <CollectForm
        configs={configs}
        selectedConfig={selectedConfig}
        onSelectConfig={setSelectedConfig}
      />
    ) : launcher === 'prepare' ? (
      <PrepareForm
        configs={configs}
        selectedConfig={selectedConfig}
        onSelectConfig={setSelectedConfig}
      />
    ) : null;

  return (
    <div className="data-screen">
      {error && <div className="err">{error}</div>}
      {launcherForm}
      <MasterDetail
        items={items}
        selectedId={selectedPath}
        onSelect={setSelectedPath}
        listLabel="Datasets"
        listAside={listAside}
        emptyList={<EmptyState title="No datasets yet" hint="Collect or Prepare data to populate data/." icon="▦" />}
        emptyDetail={<EmptyState title="No dataset selected" hint="Pick a dataset to see its stats, preview, and quality scores." icon="▦" />}
        detail={selected !== null ? <DatasetDetail dataset={selected} /> : null}
      />
    </div>
  );
}
