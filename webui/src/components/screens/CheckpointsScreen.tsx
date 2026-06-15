import { useCallback, useEffect, useMemo, useState } from 'react';
import type { Checkpoint, CheckpointHealth, CompareResult, Job } from '../../types';
import { isApiError } from '../../types';
import { getCheckpointHealth, getCheckpointCompare, runAction } from '../../api';
import {
  formatInteger,
  formatFloat,
  formatBytes,
  formatRelativeTime,
  formatParams,
} from '../../format';
import MasterDetail, { type MasterItem } from '../MasterDetail';

// ── Master-detail "Checkpoints" screen ──────────────────────────────────────
// Left: every checkpoint (newest step first, "latest" badge per model).
// Right: the selected checkpoint's health, an inline compare, and an export
// launcher — composed as a small tabbed detail so it stays scannable.

type DetailTab = 'health' | 'compare' | 'export';

type ExportFormat = 'gguf-f16' | 'q8' | 'q4' | 'ollama' | 'quantize';

const EXPORT_FORMATS: readonly { value: ExportFormat; label: string }[] = [
  { value: 'gguf-f16', label: 'GGUF f16' },
  { value: 'q8', label: 'Quantize q8' },
  { value: 'q4', label: 'Quantize q4' },
  { value: 'ollama', label: 'Ollama' },
  { value: 'quantize', label: 'Quantize' },
];

function checkpointId(ckpt: Checkpoint): string {
  return ckpt.path;
}

/** Newest step first; ties broken by most-recent mtime. */
function sortByNewest(checkpoints: Checkpoint[]): Checkpoint[] {
  return [...checkpoints].sort((a, b) => b.step - a.step || b.mtime - a.mtime);
}

/** The path of the latest (highest-step) checkpoint for each model. */
function latestPathByModel(checkpoints: Checkpoint[]): Map<string, string> {
  const best = new Map<string, Checkpoint>();
  for (const ckpt of checkpoints) {
    const current = best.get(ckpt.model);
    if (!current || ckpt.step > current.step) best.set(ckpt.model, ckpt);
  }
  const out = new Map<string, string>();
  for (const [model, ckpt] of best) out.set(model, ckpt.path);
  return out;
}

function compareLabel(ckpt: Checkpoint): string {
  return `${ckpt.model} / ${ckpt.name} @ ${formatInteger(ckpt.step)}`;
}

// ── Health tab ───────────────────────────────────────────────────────────────

function healthBadgeClass(ok: boolean): string {
  return ok ? 'tag done' : 'tag failed';
}

function HealthTab({ checkpoint }: { checkpoint: Checkpoint }): JSX.Element {
  const [health, setHealth] = useState<CheckpointHealth | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setHealth(null);
    setError(null);
    setLoading(true);
    void (async (): Promise<void> => {
      try {
        const resp = await getCheckpointHealth(checkpoint.model, String(checkpoint.step));
        if (!active) return;
        if (isApiError(resp)) setError(resp.error);
        else setHealth(resp);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, [checkpoint.model, checkpoint.step]);

  if (loading) return <div className="muted">loading…</div>;
  if (error) return <div className="err">{error}</div>;
  if (!health) return <div className="muted">no health data</div>;

  return (
    <div className="ck-section">
      <div className="row">
        <span className={healthBadgeClass(health.ok)}>{health.ok ? 'healthy' : 'no weights'}</span>
        <span className="v muted">{health.config_stem ?? 'unknown config'}</span>
      </div>

      <div className="stat-tiles">
        <div className="stat-tile">
          <span className="stat-tile-label">size</span>
          <span className="stat-tile-value mono">{formatBytes(health.size_mb * 1024 * 1024)}</span>
        </div>
        <div className="stat-tile">
          <span className="stat-tile-label">tensors</span>
          <span className="stat-tile-value mono">{formatInteger(health.num_tensors)}</span>
        </div>
        <div className="stat-tile">
          <span className="stat-tile-label">files</span>
          <span className="stat-tile-value mono">{formatInteger(health.files.length)}</span>
        </div>
        <div className="stat-tile">
          <span className="stat-tile-label">loss</span>
          <span className="stat-tile-value mono">{formatFloat(health.loss, 4)}</span>
        </div>
      </div>

      <div className="row">
        <span className="k">model</span>
        <span className="v mono">{health.model}</span>
      </div>
      <div className="row">
        <span className="k">step</span>
        <span className="v mono">{formatInteger(health.step)}</span>
      </div>

      <div className="card-title">files</div>
      {health.files.length === 0 ? (
        <div className="muted">none</div>
      ) : (
        <div className="scroll">
          {health.files.map((f) => (
            <div key={f} className="mono">
              {f}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Compare tab ───────────────────────────────────────────────────────────────

type DeltaTone = 'up' | 'down' | 'flat';

function deltaTone(value: number): DeltaTone {
  if (value > 0) return 'up';
  if (value < 0) return 'down';
  return 'flat';
}

function signedParams(n: number): string {
  const sign = n > 0 ? '+' : n < 0 ? '-' : '';
  return `${sign}${formatParams(Math.abs(n))}`;
}

function signedInteger(n: number): string {
  const sign = n > 0 ? '+' : '';
  return `${sign}${formatInteger(n)}`;
}

function moeTag(isMoe: boolean): JSX.Element {
  return <span className={`tag ${isMoe ? 'done' : 'failed'}`}>{isMoe ? 'moe' : 'dense'}</span>;
}

function CompareResultView({ result }: { result: CompareResult }): JSX.Element {
  const { a, b, diff } = result;
  return (
    <div className="cmp-result">
      <div className="cmp-grid">
        <div className="cmp-head cmp-metric" />
        <div className="cmp-head">
          <span className="cmp-head-title">A (this)</span>
          <span className="cmp-head-path muted mono" title={a.path}>
            {a.path}
          </span>
        </div>
        <div className="cmp-head">
          <span className="cmp-head-title">B (other)</span>
          <span className="cmp-head-path muted mono" title={b.path}>
            {b.path}
          </span>
        </div>

        <div className="cmp-row">
          <span className="cmp-metric muted">params</span>
          <span className="cmp-cell mono">{formatParams(a.num_params)}</span>
          <span className="cmp-cell mono">{formatParams(b.num_params)}</span>
        </div>
        <div className="cmp-row">
          <span className="cmp-metric muted">tensors</span>
          <span className="cmp-cell mono">{formatInteger(a.tensor_count)}</span>
          <span className="cmp-cell mono">{formatInteger(b.tensor_count)}</span>
        </div>
        <div className="cmp-row">
          <span className="cmp-metric muted">type</span>
          <span className="cmp-cell">{moeTag(a.is_moe)}</span>
          <span className="cmp-cell">{moeTag(b.is_moe)}</span>
        </div>
      </div>

      <div className="cmp-deltas">
        <div className="cmp-delta">
          <span className="cmp-delta-label muted">Δ params</span>
          <span className={`cmp-delta-value mono tone-${deltaTone(diff.num_params_delta)}`}>
            {signedParams(diff.num_params_delta)}
          </span>
        </div>
        <div className="cmp-delta">
          <span className="cmp-delta-label muted">Δ tensors</span>
          <span className={`cmp-delta-value mono tone-${deltaTone(diff.tensor_count_delta)}`}>
            {signedInteger(diff.tensor_count_delta)}
          </span>
        </div>
        <div className="cmp-delta">
          <span className="cmp-delta-label muted">moe</span>
          <span className="cmp-delta-value">
            <span className={`tag ${diff.is_moe_changed ? 'running' : 'done'}`}>
              {diff.is_moe_changed ? 'changed' : 'same'}
            </span>
          </span>
        </div>
      </div>
    </div>
  );
}

function CompareTab({
  checkpoint,
  others,
}: {
  checkpoint: Checkpoint;
  others: Checkpoint[];
}): JSX.Element {
  const [otherPath, setOtherPath] = useState<string>(() => others[0]?.path ?? '');
  const [result, setResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Reset stale comparisons whenever the primary checkpoint changes.
  useEffect(() => {
    setResult(null);
    setError(null);
    setOtherPath(others[0]?.path ?? '');
  }, [checkpoint.path, others]);

  const canCompare = otherPath !== '' && otherPath !== checkpoint.path;

  const onCompare = useCallback(async (): Promise<void> => {
    if (!canCompare) return;
    setResult(null);
    setError(null);
    setLoading(true);
    try {
      const resp = await getCheckpointCompare(checkpoint.path, otherPath);
      if (isApiError(resp)) setError(resp.error);
      else setResult(resp);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [canCompare, checkpoint.path, otherPath]);

  if (others.length === 0) {
    return <div className="muted">need another checkpoint to compare against</div>;
  }

  return (
    <div className="ck-section">
      <div className="md-toolbar">
        <label className="cmp-select">
          <span className="cmp-select-tag tag">vs</span>
          <select
            className="select"
            value={otherPath}
            onChange={(e) => setOtherPath(e.target.value)}
          >
            {others.map((ckpt) => (
              <option key={ckpt.path} value={ckpt.path}>
                {compareLabel(ckpt)}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => void onCompare()}
          disabled={!canCompare || loading}
        >
          {loading ? '…comparing' : 'Compare'}
        </button>
      </div>

      {error && <div className="err">{error}</div>}
      {result && !loading && <CompareResultView result={result} />}
    </div>
  );
}

// ── Export tab ────────────────────────────────────────────────────────────────

function ExportTab({ checkpoint }: { checkpoint: Checkpoint }): JSX.Element {
  const [format, setFormat] = useState<ExportFormat>('gguf-f16');
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [launching, setLaunching] = useState<boolean>(false);

  // Clear the launched-job notice when switching checkpoints.
  useEffect(() => {
    setJob(null);
    setError(null);
  }, [checkpoint.path]);

  const onExport = useCallback(async (): Promise<void> => {
    setJob(null);
    setError(null);
    setLaunching(true);
    try {
      const resp = await runAction('export_model', [
        '--checkpoint',
        checkpoint.path,
        '--config',
        `configs/${checkpoint.model}.yaml`,
        '--action',
        format,
      ]);
      setJob(resp);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLaunching(false);
    }
  }, [checkpoint.path, checkpoint.model, format]);

  return (
    <div className="ck-section">
      <div className="md-toolbar">
        <label className="cmp-select">
          <span className="cmp-select-tag tag">format</span>
          <select
            className="select"
            value={format}
            onChange={(e) => setFormat(e.target.value as ExportFormat)}
          >
            {EXPORT_FORMATS.map((f) => (
              <option key={f.value} value={f.value}>
                {f.label}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => void onExport()}
          disabled={launching}
        >
          {launching ? '…launching' : 'Export'}
        </button>
      </div>

      <div className="row">
        <span className="k">config</span>
        <span className="v mono">configs/{checkpoint.model}.yaml</span>
      </div>

      {error && <div className="err">{error}</div>}
      {job && (
        <div className="row">
          <span className="k">launched job</span>
          <span className="v mono">
            {job.id} <span className={`tag ${job.status}`}>{job.status}</span>
          </span>
        </div>
      )}
    </div>
  );
}

// ── Detail pane ───────────────────────────────────────────────────────────────

function CheckpointDetailPane({
  checkpoint,
  others,
}: {
  checkpoint: Checkpoint;
  others: Checkpoint[];
}): JSX.Element {
  const [tab, setTab] = useState<DetailTab>('health');

  // Default back to the health tab whenever a different checkpoint is selected.
  useEffect(() => {
    setTab('health');
  }, [checkpoint.path]);

  const tabs: readonly { id: DetailTab; label: string }[] = [
    { id: 'health', label: 'Health' },
    { id: 'compare', label: 'Compare' },
    { id: 'export', label: 'Export' },
  ];

  return (
    <div className="card card-wide">
      <div className="md-detail-head">
        <div>
          <h2 className="md-detail-title">
            {checkpoint.model} @ step {formatInteger(checkpoint.step)}
          </h2>
          <div className="muted mono ck-subline">
            loss {formatFloat(checkpoint.loss, 4)} · {checkpoint.path}
          </div>
        </div>
        <div className="md-toolbar">
          {tabs.map((t) => (
            <button
              key={t.id}
              type="button"
              className={`btn${tab === t.id ? ' btn-primary' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {tab === 'health' && <HealthTab checkpoint={checkpoint} />}
      {tab === 'compare' && <CompareTab checkpoint={checkpoint} others={others} />}
      {tab === 'export' && <ExportTab checkpoint={checkpoint} />}
    </div>
  );
}

// ── Screen ────────────────────────────────────────────────────────────────────

export default function CheckpointsScreen({
  checkpoints,
}: {
  checkpoints: Checkpoint[];
}): JSX.Element {
  const sorted = useMemo(() => sortByNewest(checkpoints), [checkpoints]);
  const latestByModel = useMemo(() => latestPathByModel(checkpoints), [checkpoints]);

  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Default-select the newest checkpoint; keep selection valid as data updates.
  useEffect(() => {
    setSelectedId((prev) => {
      if (prev !== null && sorted.some((c) => checkpointId(c) === prev)) return prev;
      return sorted[0] ? checkpointId(sorted[0]) : null;
    });
  }, [sorted]);

  const items: MasterItem[] = useMemo(
    () =>
      sorted.map((ckpt) => ({
        id: checkpointId(ckpt),
        title: `step ${formatInteger(ckpt.step)}`,
        subtitle: ckpt.model,
        meta: formatRelativeTime(ckpt.mtime),
        badge:
          latestByModel.get(ckpt.model) === ckpt.path ? (
            <span className="tag done">latest</span>
          ) : undefined,
      })),
    [sorted, latestByModel],
  );

  const selected = useMemo(
    () => sorted.find((c) => checkpointId(c) === selectedId) ?? null,
    [sorted, selectedId],
  );

  const others = useMemo(
    () => (selected ? sorted.filter((c) => checkpointId(c) !== selected.path) : []),
    [sorted, selected],
  );

  return (
    <MasterDetail
      items={items}
      selectedId={selectedId}
      onSelect={setSelectedId}
      listLabel={`${items.length} checkpoint${items.length === 1 ? '' : 's'}`}
      emptyList="No checkpoints yet — train a model first"
      emptyDetail="Select a checkpoint to see its health, compare it, or export it"
      detail={
        selected ? <CheckpointDetailPane checkpoint={selected} others={others} /> : null
      }
    />
  );
}
