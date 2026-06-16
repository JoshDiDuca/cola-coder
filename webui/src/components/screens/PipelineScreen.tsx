import { useCallback, useEffect, useMemo, useState } from 'react';
import type {
  ConfigFile,
  Job,
  PipelineRun,
  PipelineRunDetail,
  PipelineStageState,
} from '../../types';
import { isApiError } from '../../types';
import {
  createPipelineRun,
  deletePipelineRun,
  getConfigs,
  getPipelineDetail,
  getPipelineRuns,
  resetPipelineRun,
  setPipelineOverride,
} from '../../api';
import { formatRelativeTime } from '../../format';
import MasterDetail, { type MasterItem } from '../MasterDetail';
import PipelineLauncher from '../PipelineLauncher';
import EmptyState from '../EmptyState';

// ── Master-detail "Configs & Pipeline" screen (R2) ────────────────────────────
// Top: the one-click full-pipeline launcher. Below: a master-detail view of every
// named pipeline run — the run list on the left, the selected run's full stage
// timeline + lifecycle actions (reset / override / delete) on the right. All
// lifecycle calls are pure state ops (they never execute a stage); each refreshes
// the list + the selected detail on success.

// The generated `status` fields are open `string`s. We normalize the known
// lifecycle statuses to a typed union for color mapping, with a typed fallback
// bucket for anything the backend introduces later.
type StageStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped' | 'other';

const KNOWN_STAGE_STATUSES: readonly StageStatus[] = [
  'pending',
  'running',
  'completed',
  'failed',
  'skipped',
];

function toStageStatus(raw: string): StageStatus {
  const lower = raw.toLowerCase();
  return (KNOWN_STAGE_STATUSES as readonly string[]).includes(lower)
    ? (lower as StageStatus)
    : 'other';
}

// Map a normalized stage status to a `.tag` tone class. Exhaustive over the union.
function stageStatusClass(status: StageStatus): string {
  switch (status) {
    case 'running':
      return 'tag running';
    case 'completed':
      return 'tag done';
    case 'failed':
      return 'tag failed';
    case 'skipped':
      return 'tag warn';
    case 'pending':
      return 'tag';
    case 'other':
      return 'tag';
    default: {
      const _exhaustive: never = status;
      return _exhaustive;
    }
  }
}

// Overall-run badge reuses the same tone mapping (the run status is the same
// vocabulary as a stage status).
function runStatusBadge(status: string | null): JSX.Element {
  const normalized = toStageStatus(status ?? 'pending');
  return <span className={stageStatusClass(normalized)}>{status ?? 'pending'}</span>;
}

function progressMeta(run: PipelineRun): string {
  const total = run.num_stages ?? 0;
  const done = run.completed ?? 0;
  return `${done}/${total} done`;
}

// ── Detail toolbar action forms ────────────────────────────────────────────────

interface StageActionsProps {
  detail: PipelineRunDetail;
  busy: boolean;
  onReset: (stageNum: number) => void;
  onOverride: (stageNum: number, path: string) => void;
  onDelete: () => void;
}

function StageActions({
  detail,
  busy,
  onReset,
  onOverride,
  onDelete,
}: StageActionsProps): JSX.Element {
  const stages = detail.stages;
  const firstStageNum = stages[0]?.num ?? 1;

  const [resetStage, setResetStage] = useState<number>(firstStageNum);
  const [overrideStage, setOverrideStage] = useState<number>(firstStageNum);
  const [overridePath, setOverridePath] = useState<string>('');

  // Keep the selected stages valid as the run (and its stage set) changes.
  useEffect(() => {
    setResetStage((prev) => (stages.some((s) => s.num === prev) ? prev : firstStageNum));
    setOverrideStage((prev) => (stages.some((s) => s.num === prev) ? prev : firstStageNum));
    setOverridePath('');
  }, [detail.name, stages, firstStageNum]);

  const canOverride = overridePath.trim() !== '';

  return (
    <div className="pscreen-actions">
      <div className="pscreen-action">
        <label className="pscreen-field">
          <span className="pscreen-field-tag tag">reset from</span>
          <select
            className="select"
            value={resetStage}
            onChange={(e) => setResetStage(Number(e.target.value))}
            disabled={busy}
          >
            {stages.map((s) => (
              <option key={s.num} value={s.num}>
                {s.num}. {s.name}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="btn"
          onClick={() => onReset(resetStage)}
          disabled={busy}
        >
          Reset
        </button>
      </div>

      <div className="pscreen-action">
        <label className="pscreen-field">
          <span className="pscreen-field-tag tag">override</span>
          <select
            className="select"
            value={overrideStage}
            onChange={(e) => setOverrideStage(Number(e.target.value))}
            disabled={busy}
          >
            {stages.map((s) => (
              <option key={s.num} value={s.num}>
                {s.num}. {s.name}
              </option>
            ))}
          </select>
        </label>
        <input
          className="input pscreen-override-path"
          type="text"
          placeholder="artifact path…"
          value={overridePath}
          onChange={(e) => setOverridePath(e.target.value)}
          disabled={busy}
        />
        <button
          type="button"
          className="btn"
          onClick={() => onOverride(overrideStage, overridePath.trim())}
          disabled={busy || !canOverride}
        >
          Set
        </button>
      </div>

      <button
        type="button"
        className="btn btn-danger"
        onClick={onDelete}
        disabled={busy}
      >
        Delete run
      </button>
    </div>
  );
}

// ── Stage timeline ──────────────────────────────────────────────────────────────

function StageRow({ stage }: { stage: PipelineStageState }): JSX.Element {
  const status = toStageStatus(stage.status);
  return (
    <div className="pscreen-stage">
      <span className="pscreen-stage-num mono">{stage.num}</span>
      <div className="pscreen-stage-body">
        <div className="pscreen-stage-head">
          <span className="pscreen-stage-name">{stage.name}</span>
          {stage.optional ? <span className="tag pscreen-optional">optional</span> : null}
          <span className={stageStatusClass(status)}>{stage.status}</span>
        </div>
        {stage.description ? (
          <div className="pscreen-stage-desc muted">{stage.description}</div>
        ) : null}
        {stage.artifact ? (
          <div className="pscreen-stage-line">
            <span className="pscreen-stage-key muted">artifact</span>
            <span className="mono pscreen-stage-val">{stage.artifact}</span>
          </div>
        ) : null}
        {stage.override ? (
          <div className="pscreen-stage-line">
            <span className="pscreen-stage-key muted">override</span>
            <span className="mono pscreen-stage-val">{stage.override}</span>
          </div>
        ) : null}
        {stage.error ? (
          <div className="pscreen-stage-line">
            <span className="pscreen-stage-key muted">error</span>
            <span className="err pscreen-stage-val">{stage.error}</span>
          </div>
        ) : null}
      </div>
    </div>
  );
}

// ── Detail pane ─────────────────────────────────────────────────────────────────

interface RunDetailPaneProps {
  detail: PipelineRunDetail;
  busy: boolean;
  actionError: string | null;
  onReset: (stageNum: number) => void;
  onOverride: (stageNum: number, path: string) => void;
  onDelete: () => void;
}

function RunDetailPane({
  detail,
  busy,
  actionError,
  onReset,
  onOverride,
  onDelete,
}: RunDetailPaneProps): JSX.Element {
  return (
    <div className="card card-wide">
      <div className="md-detail-head">
        <div>
          <h2 className="md-detail-title">{detail.name}</h2>
          <div className="muted mono pscreen-subline">
            {runStatusBadge(detail.status)} · {detail.config_path} ·{' '}
            {detail.completed}/{detail.num_stages} done
          </div>
        </div>
        <div className="md-toolbar">
          <StageActions
            detail={detail}
            busy={busy}
            onReset={onReset}
            onOverride={onOverride}
            onDelete={onDelete}
          />
        </div>
      </div>

      {actionError ? <div className="err pscreen-chip">{actionError}</div> : null}

      <div className="pscreen-timeline">
        {detail.stages.map((stage) => (
          <StageRow key={stage.num} stage={stage} />
        ))}
      </div>
    </div>
  );
}

// ── "New run" form (list aside) ───────────────────────────────────────────────

interface NewRunFormProps {
  configs: ConfigFile[];
  onCreate: (name: string, configPath: string) => void;
  busy: boolean;
}

function NewRunForm({ configs, onCreate, busy }: NewRunFormProps): JSX.Element {
  const [name, setName] = useState<string>('');
  const [configPath, setConfigPath] = useState<string>(() => configs[0]?.path ?? '');

  // Default the config select to the first available config once configs load.
  useEffect(() => {
    setConfigPath((prev) => (configs.some((c) => c.path === prev) ? prev : configs[0]?.path ?? ''));
  }, [configs]);

  const canCreate = name.trim() !== '' && configPath !== '' && !busy;

  return (
    <form
      className="pscreen-newrun"
      onSubmit={(e) => {
        e.preventDefault();
        if (canCreate) onCreate(name.trim(), configPath);
      }}
    >
      <input
        className="input pscreen-newrun-name"
        type="text"
        placeholder="new run name…"
        value={name}
        onChange={(e) => setName(e.target.value)}
        disabled={busy}
      />
      <select
        className="select"
        value={configPath}
        onChange={(e) => setConfigPath(e.target.value)}
        disabled={busy || configs.length === 0}
      >
        {configs.map((c) => (
          <option key={c.path} value={c.path}>
            {c.rel}
          </option>
        ))}
      </select>
      <button type="submit" className="btn btn-primary" disabled={!canCreate}>
        Create
      </button>
    </form>
  );
}

// ── Screen ────────────────────────────────────────────────────────────────────

interface PipelineScreenProps {
  trainingAlive: boolean;
}

export default function PipelineScreen(props: PipelineScreenProps): JSX.Element {
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [configs, setConfigs] = useState<ConfigFile[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<PipelineRunDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState<boolean>(false);

  const refresh = useCallback(async (): Promise<void> => {
    try {
      const list = await getPipelineRuns();
      setRuns(list);
    } catch (e) {
      setDetailError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  // Initial load: runs + configs (for the "new run" form).
  useEffect(() => {
    void refresh();
    void (async (): Promise<void> => {
      try {
        const list = await getConfigs();
        setConfigs(list);
      } catch {
        // Non-fatal: the create form simply shows no config options.
      }
    })();
  }, [refresh]);

  // Keep the selection valid as the run list changes; default to the first run.
  useEffect(() => {
    setSelectedId((prev) => {
      if (prev !== null && runs.some((r) => r.name === prev)) return prev;
      return runs[0]?.name ?? null;
    });
  }, [runs]);

  // Load the selected run's detail whenever the selection changes.
  useEffect(() => {
    if (selectedId === null) {
      setDetail(null);
      return;
    }
    let active = true;
    setDetail(null);
    setDetailError(null);
    setActionError(null);
    void (async (): Promise<void> => {
      try {
        const resp = await getPipelineDetail(selectedId);
        if (!active) return;
        if (isApiError(resp)) setDetailError(resp.error);
        else setDetail(resp);
      } catch (e) {
        if (active) setDetailError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, [selectedId]);

  const onCreate = useCallback(
    async (name: string, configPath: string): Promise<void> => {
      setBusy(true);
      setActionError(null);
      try {
        const resp = await createPipelineRun(name, configPath);
        if (isApiError(resp)) {
          setActionError(resp.error);
          return;
        }
        await refresh();
        setSelectedId(resp.name);
        setDetail(resp);
      } catch (e) {
        setActionError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [refresh],
  );

  const onReset = useCallback(
    async (stageNum: number): Promise<void> => {
      if (selectedId === null) return;
      setBusy(true);
      setActionError(null);
      try {
        const resp = await resetPipelineRun(selectedId, stageNum);
        if (isApiError(resp)) setActionError(resp.error);
        else setDetail(resp);
        await refresh();
      } catch (e) {
        setActionError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [selectedId, refresh],
  );

  const onOverride = useCallback(
    async (stageNum: number, path: string): Promise<void> => {
      if (selectedId === null) return;
      setBusy(true);
      setActionError(null);
      try {
        const resp = await setPipelineOverride(selectedId, stageNum, path);
        if (isApiError(resp)) setActionError(resp.error);
        else setDetail(resp);
        await refresh();
      } catch (e) {
        setActionError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [selectedId, refresh],
  );

  const onDelete = useCallback(async (): Promise<void> => {
    if (selectedId === null) return;
    if (!window.confirm(`Delete pipeline run "${selectedId}"? This cannot be undone.`)) return;
    setBusy(true);
    setActionError(null);
    try {
      const resp = await deletePipelineRun(selectedId);
      if (isApiError(resp)) {
        setActionError(resp.error);
        return;
      }
      setSelectedId(null);
      setDetail(null);
      await refresh();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [selectedId, refresh]);

  const onLaunched = useCallback(
    (_job: Job): void => {
      void refresh();
    },
    [refresh],
  );

  const items: MasterItem[] = useMemo(
    () =>
      runs.map((run) => ({
        id: run.name,
        title: run.name,
        subtitle: run.path,
        meta: `${progressMeta(run)} · ${formatRelativeTime(run.mtime)}`,
        badge: runStatusBadge(run.status),
      })),
    [runs],
  );

  const detailNode = useMemo<JSX.Element | null>(() => {
    if (detailError && !detail) return <div className="card card-wide err">{detailError}</div>;
    if (!detail) return <div className="card card-wide muted">loading…</div>;
    return (
      <RunDetailPane
        detail={detail}
        busy={busy}
        actionError={actionError}
        onReset={(n) => void onReset(n)}
        onOverride={(n, p) => void onOverride(n, p)}
        onDelete={() => void onDelete()}
      />
    );
  }, [detail, detailError, busy, actionError, onReset, onOverride, onDelete]);

  return (
    <div className="pscreen">
      <section className="pscreen-launch">
        <h2 className="run-section-title">Launch pipeline</h2>
        <PipelineLauncher trainingAlive={props.trainingAlive} onLaunched={onLaunched} />
      </section>

      <section className="pscreen-runs">
        <h2 className="run-section-title">Pipeline runs</h2>
        <MasterDetail
          items={items}
          selectedId={selectedId}
          onSelect={setSelectedId}
          listLabel={`${items.length} run${items.length === 1 ? '' : 's'}`}
          listAside={<NewRunForm configs={configs} onCreate={(n, p) => void onCreate(n, p)} busy={busy} />}
          emptyList={<EmptyState title="No pipeline runs yet" hint="Create a run or launch the full pipeline above to get started." icon="⛓" />}
          emptyDetail={<EmptyState title="No run selected" hint="Pick a run to see its stage timeline." icon="⛓" />}
          detail={detailNode}
        />
      </section>
    </div>
  );
}
