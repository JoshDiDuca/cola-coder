import { useCallback, useEffect, useState } from 'react';
import type {
  PipelineRun,
  PipelineRunDetail,
  PipelineStageState,
  ConfigFile,
} from '../types';
import {
  getPipelineRuns,
  getPipelineDetail,
  createPipelineRun,
  resetPipelineRun,
  setPipelineOverride,
  deletePipelineRun,
  getConfigs,
} from '../api';
import { isApiError } from '../types';
import { formatDuration } from '../format';

// The two optional stages the create form can skip (stage 4 extend-context,
// stage 7 upcycle-moe). Kept as a typed constant so the checkboxes and the
// submitted skip list never drift apart.
const OPTIONAL_STAGES: ReadonlyArray<{ num: number; label: string }> = [
  { num: 4, label: 'Stage 4 — extend context' },
  { num: 7, label: 'Stage 7 — upcycle MoE' },
];

const NAME_PLACEHOLDER = 'run name (letters, digits, -, _)';

// Map a pipeline stage status string to a `.tag` modifier class. Statuses are
// 'pending' | 'running' | 'completed' | 'failed' | 'skipped'; anything else
// falls through to the neutral `tag` class.
function stageBadgeClass(status: string): string {
  switch (status) {
    case 'running':
      return 'tag running';
    case 'completed':
      return 'tag done';
    case 'failed':
      return 'tag failed';
    case 'pending':
    case 'skipped':
      return 'tag';
    default:
      return 'tag';
  }
}

export default function PipelineManagerPanel() {
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [configs, setConfigs] = useState<ConfigFile[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Create-run form state.
  const [newName, setNewName] = useState<string>('');
  const [newConfigPath, setNewConfigPath] = useState<string>('');
  const [skipStages, setSkipStages] = useState<number[]>([]);
  const [createError, setCreateError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  // Selected run + its detail.
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [detail, setDetail] = useState<PipelineRunDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  // Pending per-stage override edits, keyed by stage number.
  const [overrideDrafts, setOverrideDrafts] = useState<Record<number, string>>({});
  const [busyStage, setBusyStage] = useState<number | null>(null);

  const refreshRuns = useCallback(async () => {
    try {
      const next = await getPipelineRuns();
      setRuns(next);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const [runList, cfgList] = await Promise.all([getPipelineRuns(), getConfigs()]);
        if (!active) return;
        setRuns(runList);
        setConfigs(cfgList);
        if (cfgList.length > 0) setNewConfigPath(cfgList[0].path);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const loadDetail = useCallback(async (name: string) => {
    setSelectedName(name);
    setDetailLoading(true);
    setDetailError(null);
    setDetail(null);
    setOverrideDrafts({});
    try {
      const result = await getPipelineDetail(name);
      if (isApiError(result)) {
        setDetailError(result.error);
      } else {
        setDetail(result);
      }
    } catch (e) {
      setDetailError(e instanceof Error ? e.message : String(e));
    } finally {
      setDetailLoading(false);
    }
  }, []);

  const onToggleSkip = useCallback((stageNum: number, checked: boolean) => {
    setSkipStages((prev) =>
      checked ? [...prev, stageNum] : prev.filter((n) => n !== stageNum),
    );
  }, []);

  const onCreate = useCallback(async () => {
    const name = newName.trim();
    setCreateError(null);
    if (name === '') {
      setCreateError('a run name is required');
      return;
    }
    if (newConfigPath === '') {
      setCreateError('a config is required');
      return;
    }
    setCreating(true);
    try {
      const result = await createPipelineRun(name, newConfigPath, skipStages);
      if (isApiError(result)) {
        setCreateError(result.error);
        return;
      }
      setNewName('');
      setSkipStages([]);
      await refreshRuns();
      setSelectedName(result.name);
      setDetail(result);
      setDetailError(null);
      setOverrideDrafts({});
    } catch (e) {
      setCreateError(e instanceof Error ? e.message : String(e));
    } finally {
      setCreating(false);
    }
  }, [newName, newConfigPath, skipStages, refreshRuns]);

  // Apply a returned detail-or-error directly: the endpoints return the fresh
  // state, so we never need a follow-up fetch.
  const applyResult = useCallback((result: PipelineRunDetail | { error: string }) => {
    if (isApiError(result)) {
      setDetailError(result.error);
    } else {
      setDetail(result);
      setDetailError(null);
      setOverrideDrafts({});
    }
  }, []);

  const onReset = useCallback(
    async (name: string, stageNum: number) => {
      setBusyStage(stageNum);
      setDetailError(null);
      try {
        const result = await resetPipelineRun(name, stageNum);
        applyResult(result);
      } catch (e) {
        setDetailError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusyStage(null);
      }
    },
    [applyResult],
  );

  const onSetOverride = useCallback(
    async (name: string, stageNum: number) => {
      const path = (overrideDrafts[stageNum] ?? '').trim();
      if (path === '') {
        setDetailError('an override path is required');
        return;
      }
      setBusyStage(stageNum);
      setDetailError(null);
      try {
        const result = await setPipelineOverride(name, stageNum, path);
        applyResult(result);
      } catch (e) {
        setDetailError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusyStage(null);
      }
    },
    [overrideDrafts, applyResult],
  );

  const onDelete = useCallback(
    async (name: string) => {
      if (!window.confirm(`Delete pipeline run "${name}"? This cannot be undone.`)) {
        return;
      }
      setDetailError(null);
      try {
        const result = await deletePipelineRun(name);
        if (isApiError(result)) {
          setDetailError(result.error);
          return;
        }
        setSelectedName(null);
        setDetail(null);
        await refreshRuns();
      } catch (e) {
        setDetailError(e instanceof Error ? e.message : String(e));
      }
    },
    [refreshRuns],
  );

  return (
    <div className="card card-wide">
      <div className="card-title">Pipeline run manager</div>

      {error && <div className="err">{error}</div>}

      {/* Create-run form */}
      <div className="card-title" style={{ marginTop: 4 }}>
        Create a run
      </div>
      {createError && <div className="err">{createError}</div>}
      <table className="tbl">
        <tbody>
          <tr>
            <td>name</td>
            <td>
              <input
                className="input"
                style={{ width: '100%' }}
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder={NAME_PLACEHOLDER}
                spellCheck={false}
              />
            </td>
          </tr>
          <tr>
            <td>config</td>
            <td>
              {configs.length === 0 ? (
                <span className="muted">no configs found</span>
              ) : (
                <select
                  className="input"
                  style={{ width: '100%' }}
                  value={newConfigPath}
                  onChange={(e) => setNewConfigPath(e.target.value)}
                >
                  {configs.map((cfg) => (
                    <option key={cfg.path} value={cfg.path}>
                      {cfg.rel}
                    </option>
                  ))}
                </select>
              )}
            </td>
          </tr>
          <tr>
            <td>skip optional</td>
            <td>
              {OPTIONAL_STAGES.map((stage) => (
                <label key={stage.num} style={{ marginRight: 16 }}>
                  <input
                    type="checkbox"
                    checked={skipStages.includes(stage.num)}
                    onChange={(e) => onToggleSkip(stage.num, e.target.checked)}
                  />{' '}
                  {stage.label}
                </label>
              ))}
            </td>
          </tr>
        </tbody>
      </table>
      <div style={{ marginTop: 8 }}>
        <button
          className="btn btn-primary"
          onClick={() => void onCreate()}
          disabled={creating || configs.length === 0}
        >
          {creating ? '…creating' : '＋ Create run'}
        </button>
      </div>

      {/* Run list */}
      <div className="card-title" style={{ marginTop: 16 }}>
        Runs
      </div>
      {runs.length === 0 && !error ? (
        <div className="muted">no pipeline runs</div>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>name</th>
              <th>status</th>
              <th className="right">stages</th>
              <th className="right">select</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.path}>
                <td className="mono">{run.name}</td>
                <td>
                  <span className={stageBadgeClass(run.status ?? '')}>
                    {run.status ?? '—'}
                  </span>
                </td>
                <td className="right mono">
                  {run.completed == null || run.num_stages == null
                    ? '—'
                    : `${run.completed} / ${run.num_stages}`}
                </td>
                <td className="right">
                  <button
                    className={selectedName === run.name ? 'btn btn-primary' : 'btn'}
                    onClick={() => void loadDetail(run.name)}
                  >
                    {selectedName === run.name ? 'selected' : 'select'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {/* Selected run detail */}
      {selectedName !== null && (
        <>
          <div
            className="card-title"
            style={{ marginTop: 16, display: 'flex', justifyContent: 'space-between' }}
          >
            <span>Run: {selectedName}</span>
            <button className="btn" onClick={() => void onDelete(selectedName)}>
              🗑 Delete run
            </button>
          </div>

          {detailError && <div className="err">{detailError}</div>}

          {detailLoading ? (
            <div className="muted">loading…</div>
          ) : detail === null ? (
            !detailError && <div className="muted">no detail</div>
          ) : (
            <>
              <div className="muted mono">{detail.config_path}</div>
              <table className="tbl">
                <thead>
                  <tr>
                    <th className="right">#</th>
                    <th>stage</th>
                    <th>status</th>
                    <th className="right">duration</th>
                    <th>override</th>
                    <th className="right">actions</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.stages.map((stage: PipelineStageState) => (
                    <tr key={stage.num}>
                      <td className="right mono">{stage.num}</td>
                      <td>
                        <div>
                          {stage.name}
                          {stage.optional && (
                            <span className="tag" style={{ marginLeft: 8 }}>
                              optional
                            </span>
                          )}
                        </div>
                        <div className="muted">{stage.description}</div>
                        {stage.artifact !== '' && (
                          <div className="muted mono">→ {stage.artifact}</div>
                        )}
                        {stage.error !== '' && <div className="err">{stage.error}</div>}
                      </td>
                      <td>
                        <span className={stageBadgeClass(stage.status)}>
                          {stage.status}
                        </span>
                      </td>
                      <td className="right mono">
                        {stage.duration_secs > 0
                          ? formatDuration(stage.duration_secs)
                          : '—'}
                      </td>
                      <td>
                        {stage.override !== '' && (
                          <div className="muted mono" style={{ marginBottom: 4 }}>
                            {stage.override}
                          </div>
                        )}
                        <div style={{ display: 'flex', gap: 4 }}>
                          <input
                            className="input"
                            style={{ width: '100%' }}
                            value={overrideDrafts[stage.num] ?? ''}
                            onChange={(e) =>
                              setOverrideDrafts((prev) => ({
                                ...prev,
                                [stage.num]: e.target.value,
                              }))
                            }
                            placeholder="override path"
                            spellCheck={false}
                          />
                          <button
                            className="btn"
                            onClick={() =>
                              void onSetOverride(selectedName, stage.num)
                            }
                            disabled={busyStage !== null}
                          >
                            Set
                          </button>
                        </div>
                      </td>
                      <td className="right">
                        <button
                          className="btn"
                          onClick={() => void onReset(selectedName, stage.num)}
                          disabled={busyStage !== null}
                        >
                          {busyStage === stage.num ? '…' : 'Reset to here'}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </>
      )}
    </div>
  );
}
