import { useCallback, useEffect, useState } from 'react';
import type { ActionDef, Job } from '../types';
import { getActions, runAction } from '../api';

interface ActionsPanelProps {
  onRan: () => void;
  trainingAlive: boolean;
}

// Client-side classification of an action into one of five calm, human
// categories. The backend ships a flat list of ActionDef; the UI groups them
// for a navigable gallery. Keep this exhaustive over `Category`.
type Category = 'Data' | 'Evaluation' | 'Training' | 'Export' | 'Tools';

const CATEGORY_ORDER: readonly Category[] = [
  'Training',
  'Data',
  'Evaluation',
  'Export',
  'Tools',
];

// Short, plain-language blurb shown under each category heading.
const CATEGORY_BLURB: Record<Category, string> = {
  Training: 'Train, fine-tune, and scale models.',
  Data: 'Collect, prepare, score, and combine datasets.',
  Evaluation: 'Benchmark quality, safety, and regressions.',
  Export: 'Package and average checkpoints for release.',
  Tools: 'Utilities, inspection, and housekeeping.',
};

function classify(haystack: string): Category | null {
  // Data
  if (
    /\b(prepare_data|collect_data|score_data|combine_datasets|scrape|prepare_fim|prepare_docs|prepare_repo|generate_(instructions|sft|router)_data|generate_sft|generate_instructions|score_repos)\b/.test(
      haystack,
    )
  ) {
    return 'Data';
  }
  // Evaluation
  if (
    /\b(evaluate|benchmark|safety|regression|quality|completion|ts_benchmark|nano_benchmark|inference_benchmark|smoke_test|compare_models|run_eval)\b/.test(
      haystack,
    )
  ) {
    return 'Evaluation';
  }
  // Training
  if (/\b(train|upcycle|find_lr|full_pipeline|run_pipeline|auto_pipeline|background_train)\b/.test(haystack)) {
    return 'Training';
  }
  // Export
  if (/\b(export_model|export|average_checkpoints|model_card)\b/.test(haystack)) {
    return 'Export';
  }
  return null;
}

function categoryOf(a: ActionDef): Category {
  const haystack = `${a.key} ${a.script} ${a.label}`.toLowerCase();
  return classify(haystack) ?? 'Tools';
}

// Per-card transient run state. `kind` discriminates the union exhaustively.
type RunState =
  | { kind: 'idle' }
  | { kind: 'running' }
  | { kind: 'launched'; job: Job }
  | { kind: 'error'; message: string };

interface ActionCardProps {
  action: ActionDef;
  trainingAlive: boolean;
  argsValue: string;
  state: RunState;
  expanded: boolean;
  onToggleOptions: (key: string) => void;
  onArgsChange: (key: string, value: string) => void;
  onRun: (action: ActionDef) => void;
}

function ResultChip({ state }: { state: RunState }): JSX.Element | null {
  switch (state.kind) {
    case 'idle':
      return null;
    case 'running':
      return <span className="tag running">launching…</span>;
    case 'launched':
      return (
        <span className="action-chip mono" title={state.job.id}>
          <span className="tag done">started</span>
          <span className="muted">
            {state.job.name} · {state.job.status}
          </span>
        </span>
      );
    case 'error':
      return <span className="action-chip err mono">{state.message}</span>;
    default: {
      const _exhaustive: never = state;
      return _exhaustive;
    }
  }
}

function ActionCard(props: ActionCardProps): JSX.Element {
  const { action, trainingAlive, argsValue, state, expanded } = props;
  const blockedByTrainer = action.trainer === true && trainingAlive;
  const isRunning = state.kind === 'running';

  return (
    <div className="action-card">
      <div className="action-card-head">
        <div className="action-card-titles">
          <div className="action-card-title">{action.label}</div>
          <div className="action-card-script mono muted">{action.script}</div>
        </div>
        <div className="action-badges">
          {action.trainer === true && <span className="tag running">trainer</span>}
          {action.gpu === true && <span className="tag warn">GPU</span>}
        </div>
      </div>

      <div className="action-card-foot">
        <button
          type="button"
          className="btn action-options-toggle"
          aria-expanded={expanded}
          onClick={() => props.onToggleOptions(action.key)}
        >
          {expanded ? '▾ Options' : '▸ Options'}
        </button>

        <div className="action-card-actions">
          {blockedByTrainer ? (
            <span className="action-note err">training running</span>
          ) : (
            <ResultChip state={state} />
          )}
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => props.onRun(action)}
            disabled={isRunning || blockedByTrainer}
            title={blockedByTrainer ? 'Disabled while training is live' : undefined}
          >
            {isRunning ? '…running' : '▶ Run'}
          </button>
        </div>
      </div>

      {expanded && (
        <div className="action-options">
          <label className="action-options-label muted mono" htmlFor={`args-${action.key}`}>
            arguments
          </label>
          <input
            id={`args-${action.key}`}
            className="input action-args"
            value={argsValue}
            onChange={(e) => props.onArgsChange(action.key, e.target.value)}
            placeholder="args (space-separated)"
            spellCheck={false}
          />
        </div>
      )}
    </div>
  );
}

export default function ActionsPanel({ onRan, trainingAlive }: ActionsPanelProps): JSX.Element {
  const [actions, setActions] = useState<ActionDef[]>([]);
  const [args, setArgs] = useState<Record<string, string>>({});
  const [states, setStates] = useState<Record<string, RunState>>({});
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const defs = await getActions();
        if (!active) return;
        setActions(defs);
        const initial: Record<string, string> = {};
        for (const d of defs) initial[d.key] = d.args.join(' ');
        setArgs(initial);
      } catch (e) {
        if (active) setLoadError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const onArgsChange = useCallback((key: string, value: string) => {
    setArgs((prev) => ({ ...prev, [key]: value }));
  }, []);

  const onToggleOptions = useCallback((key: string) => {
    setExpanded((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const setState = useCallback((key: string, next: RunState) => {
    setStates((prev) => ({ ...prev, [key]: next }));
  }, []);

  const onRun = useCallback(
    async (action: ActionDef) => {
      const key = action.key;
      // Mirror the backend 409 guard: never launch a trainer action while a
      // live training run holds the GPU.
      if (action.trainer === true && trainingAlive) {
        setState(key, { kind: 'error', message: 'refused: training already running' });
        return;
      }
      setState(key, { kind: 'running' });
      try {
        const raw = (args[key] ?? '').trim();
        const job = await runAction(key, raw ? raw.split(/\s+/) : []);
        setState(key, { kind: 'launched', job });
        onRan();
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setState(key, {
          kind: 'error',
          message: msg.includes('409') ? 'refused: training already running' : msg,
        });
      }
    },
    [args, onRan, trainingAlive, setState],
  );

  // Bucket actions into ordered categories, preserving backend order within each.
  const grouped = CATEGORY_ORDER.map((cat) => ({
    category: cat,
    items: actions.filter((a) => categoryOf(a) === cat),
  })).filter((g) => g.items.length > 0);

  return (
    <div className="card card-wide">
      <div className="card-title">Actions</div>

      {loadError && <div className="err">{loadError}</div>}

      {actions.length === 0 && !loadError ? (
        <div className="muted">no actions available</div>
      ) : (
        <div className="action-gallery">
          {grouped.map((group) => (
            <section className="action-cat" key={group.category}>
              <div className="action-cat-head">
                <h3 className="action-cat-title">{group.category}</h3>
                <span className="action-cat-count muted mono">{group.items.length}</span>
              </div>
              <p className="action-cat-blurb muted">{CATEGORY_BLURB[group.category]}</p>
              <div className="action-grid">
                {group.items.map((a) => (
                  <ActionCard
                    key={a.key}
                    action={a}
                    trainingAlive={trainingAlive}
                    argsValue={args[a.key] ?? ''}
                    state={states[a.key] ?? { kind: 'idle' }}
                    expanded={expanded[a.key] === true}
                    onToggleOptions={onToggleOptions}
                    onArgsChange={onArgsChange}
                    onRun={(act) => void onRun(act)}
                  />
                ))}
              </div>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}
