import { useCallback, useEffect, useMemo, useState } from 'react';
import type { ActionDef, ActionParam, Job } from '../types';
import { isApiError } from '../types';
import { getActions, runAction } from '../api';
import ActionForm from './ActionForm';

// ── Full one-click training-pipeline launcher (requirement R4) ────────────────
// Exposes the 10-stage `full_pipeline` runner and the hardware-auto
// `auto_pipeline` runner. Every option is a typed form control — never a raw
// flag string. The 10-stage selection is a first-class checkbox grid; the rest
// of each action's options come from its ActionParam set via <ActionForm>.

interface PipelineLauncherProps {
  // From the live snapshot. When true, trainer-class launches (which a full
  // pipeline always is, because it pretrains) will be refused by the backend
  // with HTTP 409 to protect the live run. We warn but keep the button enabled.
  trainingAlive: boolean;
  // Called after a successful launch (parent may refresh its job list / nav).
  onLaunched?: (job: Job) => void;
}

// ── The 10 pipeline stages ────────────────────────────────────────────────────
interface PipelineStage {
  num: number;
  label: string;
  optional: boolean;
}

const STAGES: readonly PipelineStage[] = [
  { num: 1, label: 'Collect data', optional: false },
  { num: 2, label: 'Prepare data', optional: false },
  { num: 3, label: 'Pretrain', optional: false },
  { num: 4, label: 'Extend context', optional: true },
  { num: 5, label: 'Generate instructions', optional: false },
  { num: 6, label: 'Instruction-tune', optional: false },
  { num: 7, label: 'Upcycle MoE', optional: true },
  { num: 8, label: 'Train router', optional: false },
  { num: 9, label: 'Train reasoning', optional: false },
  { num: 10, label: 'Evaluate', optional: false },
];

const ALL_STAGE_NUMS: readonly number[] = STAGES.map((s) => s.num);

// Launch mode — exhaustively handled everywhere it is consumed.
type LaunchMode = 'full' | 'auto';

// The script name shown in the command preview, keyed by mode. The backend
// action keys map 1:1 to these script entry points (verified in app.py).
const MODE_SCRIPT: Record<LaunchMode, string> = {
  full: 'scripts/full_pipeline.py',
  auto: 'scripts/auto_pipeline.py',
};

// Per-launch transient state. `kind` discriminates the union exhaustively,
// mirroring ActionsPanel's RunState idiom (idle / running / launched / error).
type LaunchState =
  | { kind: 'idle' }
  | { kind: 'launching' }
  | { kind: 'launched'; job: Job }
  | { kind: 'error'; message: string };

// ── Stage selection helpers (typed, no flag-string surgery) ───────────────────

/**
 * Compile selected stage numbers into the `--stages a,b,c` arg pair.
 *
 * When every stage is selected we omit `--stages` entirely — the runner defaults
 * to all 10 stages, so the empty arg vector is the canonical "run everything"
 * form (keeps the preview clean and matches the runner's own default).
 */
function compileStageArgs(selected: ReadonlySet<number>): string[] {
  const ordered = ALL_STAGE_NUMS.filter((n) => selected.has(n));
  if (ordered.length === 0) return [];
  if (ordered.length === ALL_STAGE_NUMS.length) return [];
  return ['--stages', ordered.join(',')];
}

// ── Result chip (matches ActionsPanel's chip vocabulary) ──────────────────────

function ResultChip({ state }: { state: LaunchState }): JSX.Element | null {
  switch (state.kind) {
    case 'idle':
      return null;
    case 'launching':
      return <span className="tag running">launching…</span>;
    case 'launched':
      return (
        <span className="plaunch-chip mono" title={state.job.id}>
          <span className="tag done">started job {state.job.id}</span>
          <span className="muted">
            {state.job.name} · {state.job.status}
          </span>
        </span>
      );
    case 'error':
      return <span className="tag failed plaunch-error-chip">{state.message}</span>;
    default: {
      const _exhaustive: never = state;
      return _exhaustive;
    }
  }
}

// ── Stage selector (the centerpiece) ──────────────────────────────────────────

interface StageSelectorProps {
  selected: ReadonlySet<number>;
  onToggle: (num: number) => void;
  onAll: () => void;
  onNone: () => void;
}

function StageSelector({ selected, onToggle, onAll, onNone }: StageSelectorProps): JSX.Element {
  return (
    <div className="plaunch-stages">
      <div className="plaunch-stages-head">
        <span className="arg-field-label">Pipeline stages</span>
        <div className="plaunch-stage-quick">
          <button type="button" className="btn plaunch-quick-btn" onClick={onAll}>
            All
          </button>
          <button type="button" className="btn plaunch-quick-btn" onClick={onNone}>
            None
          </button>
        </div>
      </div>
      <div className="plaunch-stage-grid">
        {STAGES.map((stage) => {
          const id = `plaunch-stage-${stage.num}`;
          const checked = selected.has(stage.num);
          return (
            <label key={stage.num} className="plaunch-stage" htmlFor={id}>
              <input
                id={id}
                type="checkbox"
                className="arg-checkbox"
                checked={checked}
                onChange={() => onToggle(stage.num)}
              />
              <span className="plaunch-stage-num mono">{stage.num}</span>
              <span className="plaunch-stage-label">{stage.label}</span>
              {stage.optional && <span className="muted plaunch-stage-opt">(optional)</span>}
            </label>
          );
        })}
      </div>
    </div>
  );
}

// ── Command preview (read-only, shows exactly what will run) ──────────────────

function CommandPreview({ script, args }: { script: string; args: string[] }): JSX.Element {
  const cmd = ['python', script, ...args].join(' ');
  return (
    <div className="plaunch-preview">
      <span className="arg-field-label">Command preview</span>
      <code className="plaunch-preview-cmd mono">{cmd}</code>
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function PipelineLauncher(props: PipelineLauncherProps): JSX.Element {
  const { trainingAlive, onLaunched } = props;

  const [fullAction, setFullAction] = useState<ActionDef | null>(null);
  const [autoAction, setAutoAction] = useState<ActionDef | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [mode, setMode] = useState<LaunchMode>('full');

  // Stage selection — all checked by default.
  const [selectedStages, setSelectedStages] = useState<Set<number>>(
    () => new Set<number>(ALL_STAGE_NUMS),
  );

  // Arg vectors emitted by each <ActionForm> (config + remaining options for
  // full mode; the full auto option set for auto mode).
  const [fullFormArgs, setFullFormArgs] = useState<string[]>([]);
  const [autoFormArgs, setAutoFormArgs] = useState<string[]>([]);

  const [launchState, setLaunchState] = useState<LaunchState>({ kind: 'idle' });

  // Resolve the two pipeline ActionDefs on mount.
  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const defs = await getActions();
        if (!active) return;
        setFullAction(defs.find((d) => d.key === 'full_pipeline') ?? null);
        setAutoAction(defs.find((d) => d.key === 'auto_pipeline') ?? null);
      } catch (e) {
        if (active) setLoadError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  // Full-pipeline ActionForm gets every param EXCEPT `stages` — that one is
  // rendered by the dedicated StageSelector and compiled separately.
  const fullFormParams: ActionParam[] = useMemo(
    () => (fullAction?.params ?? []).filter((p) => p.name !== 'stages'),
    [fullAction],
  );
  const autoFormParams: ActionParam[] = useMemo(() => autoAction?.params ?? [], [autoAction]);

  const onToggleStage = useCallback((num: number): void => {
    setSelectedStages((prev) => {
      const next = new Set(prev);
      if (next.has(num)) next.delete(num);
      else next.add(num);
      return next;
    });
  }, []);

  const onAllStages = useCallback((): void => {
    setSelectedStages(new Set<number>(ALL_STAGE_NUMS));
  }, []);

  const onNoneStages = useCallback((): void => {
    setSelectedStages(new Set<number>());
  }, []);

  // The deterministic arg vector for the active mode. For full mode we merge the
  // ActionForm args (config, start_from, …) with the compiled --stages arg.
  const compiledArgs: string[] = useMemo(() => {
    if (mode === 'full') {
      return [...fullFormArgs, ...compileStageArgs(selectedStages)];
    }
    return autoFormArgs;
  }, [mode, fullFormArgs, autoFormArgs, selectedStages]);

  const activeScript = MODE_SCRIPT[mode];
  const activeAction: ActionDef | null = mode === 'full' ? fullAction : autoAction;

  const onLaunch = useCallback(async (): Promise<void> => {
    const actionKey: string = mode === 'full' ? 'full_pipeline' : 'auto_pipeline';
    setLaunchState({ kind: 'launching' });
    try {
      const result = await runAction(actionKey, compiledArgs);
      // The backend may return an ApiError (HTTP 409: "training already
      // running — refusing to start a second trainer"). Detect it explicitly;
      // never treat it as a started job.
      if (isApiError(result)) {
        setLaunchState({ kind: 'error', message: result.error });
        return;
      }
      setLaunchState({ kind: 'launched', job: result });
      onLaunched?.(result);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setLaunchState({
        kind: 'error',
        message: msg.includes('409')
          ? 'Refused: a training run is already live (409).'
          : msg,
      });
    }
  }, [mode, compiledArgs, onLaunched]);

  const isLaunching = launchState.kind === 'launching';

  return (
    <div className="card card-wide plaunch">
      <div className="card-title">Full training pipeline</div>

      {loadError && (
        <div className="muted plaunch-load-error">
          Could not load pipeline actions: {loadError}
        </div>
      )}

      {/* ── Mode toggle ── */}
      <div className="plaunch-mode" role="tablist" aria-label="Pipeline mode">
        <button
          type="button"
          role="tab"
          aria-selected={mode === 'full'}
          className={`btn plaunch-mode-btn${mode === 'full' ? ' plaunch-mode-active' : ''}`}
          onClick={() => setMode('full')}
        >
          Full pipeline
          <span className="muted plaunch-mode-sub">explicit 10-stage runner</span>
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === 'auto'}
          className={`btn plaunch-mode-btn${mode === 'auto' ? ' plaunch-mode-active' : ''}`}
          onClick={() => setMode('auto')}
        >
          Auto pipeline
          <span className="muted plaunch-mode-sub">detect hardware → best config</span>
        </button>
      </div>

      {/* ── Mode body ── */}
      {mode === 'full' ? (
        fullAction ? (
          <div className="plaunch-body">
            <ActionForm params={fullFormParams} onArgs={setFullFormArgs} />
            <StageSelector
              selected={selectedStages}
              onToggle={onToggleStage}
              onAll={onAllStages}
              onNone={onNoneStages}
            />
          </div>
        ) : (
          !loadError && <div className="muted">Loading full-pipeline options…</div>
        )
      ) : autoAction ? (
        <div className="plaunch-body">
          <ActionForm params={autoFormParams} onArgs={setAutoFormArgs} />
        </div>
      ) : (
        !loadError && <div className="muted">Loading auto-pipeline options…</div>
      )}

      {/* ── Trainer-guard warning ── */}
      {trainingAlive && (
        <div className="plaunch-warn">
          A training run is live — launching a pipeline that pretrains will be refused to
          protect it. Use a non-overlapping stage selection, or wait for it to finish.
        </div>
      )}

      {/* ── Command preview ── */}
      <CommandPreview script={activeScript} args={compiledArgs} />

      {/* ── Launch ── */}
      <div className="plaunch-foot">
        <ResultChip state={launchState} />
        <button
          type="button"
          className="btn btn-primary plaunch-run-btn"
          onClick={() => void onLaunch()}
          disabled={isLaunching || activeAction === null}
          title={activeAction === null ? 'Pipeline action unavailable' : undefined}
        >
          {isLaunching ? 'Launching…' : '▶ Run pipeline'}
        </button>
      </div>
    </div>
  );
}
