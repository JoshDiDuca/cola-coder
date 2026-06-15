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
import { formatDuration, formatInteger } from '../format';

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

// Modifier class for the timeline node dot, so the stepper reads at a glance.
function stageNodeClass(status: string): string {
  switch (status) {
    case 'running':
      return 'stage-num stage-num-running';
    case 'completed':
      return 'stage-num stage-num-done';
    case 'failed':
      return 'stage-num stage-num-failed';
    case 'skipped':
      return 'stage-num stage-num-skipped';
    case 'pending':
    default:
      return 'stage-num';
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

  const overallFraction =
    detail !== null && detail.num_stages > 0
      ? Math.min(1, Math.max(0, detail.completed / detail.num_stages))
      : 0;

  return (
    <div className="card card-wide">
      <div className="card-title">Pipeline run manager</div>

      {error && <div className="err">{error}</div>}

      {/* Create-run form */}
      <div className="pipe-create">
        <div className="card-title">Create a run</div>
        {createError && <div className="err">{createError}</div>}
        <div className="pipe-create-grid">
          <label className="pipe-field">
            <span className="pipe-field-label">name</span>
            <input
              className="input"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder={NAME_PLACEHOLDER}
              spellCheck={false}
            />
          </label>
          <label className="pipe-field">
            <span className="pipe-field-label">config</span>
            {configs.length === 0 ? (
              <span className="muted">no configs found</span>
            ) : (
              <select
                className="select"
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
          </label>
        </div>
        <div className="pipe-skip">
          <span className="pipe-field-label">skip optional</span>
          <div className="pipe-skip-opts">
            {OPTIONAL_STAGES.map((stage) => (
              <label key={stage.num} className="pipe-skip-opt">
                <input
                  type="checkbox"
                  checked={skipStages.includes(stage.num)}
                  onChange={(e) => onToggleSkip(stage.num, e.target.checked)}
                />{' '}
                {stage.label}
              </label>
            ))}
          </div>
        </div>
        <div>
          <button
            className="btn btn-primary"
            onClick={() => void onCreate()}
            disabled={creating || configs.length === 0}
          >
            {creating ? '…creating' : '＋ Create run'}
          </button>
        </div>
      </div>

      {/* Run list */}
      <div className="card-title" style={{ marginTop: 16 }}>
        Runs
      </div>
      {runs.length === 0 && !error ? (
        <div className="muted">no pipeline runs</div>
      ) : (
        <div className="pipe-list">
          {runs.map((run) => {
            const isSelected = selectedName === run.name;
            return (
              <button
                key={run.path}
                className={isSelected ? 'pipe-run pipe-run-active' : 'pipe-run'}
                onClick={() => void loadDetail(run.name)}
              >
                <div className="pipe-run-main">
                  <span className="pipe-run-name mono">{run.name}</span>
                  <span className={stageBadgeClass(run.status ?? '')}>
                    {run.status ?? '—'}
                  </span>
                </div>
                <div className="pipe-run-foot">
                  <span className="pipe-run-count mono">
                    {run.completed == null || run.num_stages == null
                      ? '—'
                      : `${formatInteger(run.completed)} / ${formatInteger(run.num_stages)} stages`}
                  </span>
                  <span className="muted">{isSelected ? 'selected' : 'select'}</span>
                </div>
              </button>
            );
          })}
        </div>
      )}

      {/* Selected run detail — stage timeline */}
      {selectedName !== null && (
        <div className="pipe-detail">
          <div className="pipe-detail-head">
            <div className="pipe-detail-titles">
              <div className="card-title">Run</div>
              <span className="pipe-detail-name mono">{selectedName}</span>
              {detail !== null && (
                <span className="muted mono">{detail.config_path}</span>
              )}
            </div>
            <button className="btn btn-danger" onClick={() => void onDelete(selectedName)}>
              Delete run
            </button>
          </div>

          {detailError && <div className="err">{detailError}</div>}

          {detailLoading ? (
            <div className="muted">loading…</div>
          ) : detail === null ? (
            !detailError && <div className="muted">no detail</div>
          ) : (
            <>
              <div className="pipe-overall">
                <div className="pipe-overall-head">
                  <span className={stageBadgeClass(detail.status)}>{detail.status}</span>
                  <span className="pipe-run-count mono">
                    {formatInteger(detail.completed)} / {formatInteger(detail.num_stages)} stages
                  </span>
                </div>
                <div className="bar">
                  <div className="fill" style={{ width: `${overallFraction * 100}%` }} />
                </div>
              </div>

              <div className="stage-timeline">
                {detail.stages.map((stage: PipelineStageState) => (
                  <div
                    key={stage.num}
                    className={
                      stage.status === 'failed' ? 'stage-row stage-row-failed' : 'stage-row'
                    }
                  >
                    <div className="stage-rail">
                      <div className={stageNodeClass(stage.status)}>{stage.num}</div>
                    </div>

                    <div className="stage-body">
                      <div className="stage-head">
                        <span className="stage-name">{stage.name}</span>
                        {stage.optional && <span className="tag">optional</span>}
                        <span className={stageBadgeClass(stage.status)}>{stage.status}</span>
                        <span className="stage-dur mono muted">
                          {stage.duration_secs > 0 ? formatDuration(stage.duration_secs) : ''}
                        </span>
                      </div>

                      {stage.description !== '' && (
                        <div className="muted stage-desc">{stage.description}</div>
                      )}
                      {stage.artifact !== '' && (
                        <div className="muted mono stage-artifact">→ {stage.artifact}</div>
                      )}
                      {stage.override !== '' && (
                        <div className="muted mono stage-artifact">override: {stage.override}</div>
                      )}
                      {stage.error !== '' && <div className="err">{stage.error}</div>}

                      <div className="stage-actions">
                        <input
                          className="input stage-override-input"
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
                          onClick={() => void onSetOverride(selectedName, stage.num)}
                          disabled={busyStage !== null}
                        >
                          Set
                        </button>
                        <button
                          className="btn"
                          onClick={() => void onReset(selectedName, stage.num)}
                          disabled={busyStage !== null}
                        >
                          {busyStage === stage.num ? '…' : 'Reset to here'}
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
