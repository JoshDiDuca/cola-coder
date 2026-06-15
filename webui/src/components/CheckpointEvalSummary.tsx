import { useEffect, useState } from 'react';
import type {
  EvalResult,
  EvalHistoryView,
  EvalSnapshot,
  RegressionHistory,
  RegressionRun,
  RegressionMetric,
  ApiError,
  JsonValue,
} from '../types';
import { isApiError } from '../types';
import { getEvals, getEvalHistory, getRegressionHistory } from '../api';
import { formatFloat, formatInteger, formatRelativeTime } from '../format';

// Each source loads independently: `null` = still loading, an `ApiError`/error
// string = a real failure for that source only, a resolved value = done. A
// successfully-loaded-but-empty source is represented by the value itself
// (e.g. an empty array / zero-length snapshots), not an error.
interface SourceState<T> {
  data: T | null;
  error: string | null;
  loaded: boolean;
}

function pending<T>(): SourceState<T> {
  return { data: null, error: null, loaded: false };
}

/** Newest-by-mtime comparator (descending). Typed — no implicit any in sort. */
function byMtimeDesc(a: { mtime: number }, b: { mtime: number }): number {
  return b.mtime - a.mtime;
}

/**
 * A numeric metric pulled out of a snapshot's open `Record<string, JsonValue>`
 * metrics map. We surface a pass@k score when present, falling back to the
 * first finite numeric metric so an unusual eval still shows something.
 */
interface NumericMetric {
  key: string;
  value: number;
}

function pickNumericMetric(metrics: Record<string, JsonValue>): NumericMetric | null {
  const entries = Object.entries(metrics);
  // Prefer a pass@k metric (the headline eval number), lowest k first.
  const passEntries = entries
    .filter(([key]) => key.toLowerCase().startsWith('pass@'))
    .sort(([a], [b]) => a.localeCompare(b));
  for (const [key, value] of passEntries) {
    if (typeof value === 'number' && Number.isFinite(value)) return { key, value };
  }
  for (const [key, value] of entries) {
    if (typeof value === 'number' && Number.isFinite(value)) return { key, value };
  }
  return null;
}

/** Regression verdict reuses the existing `.tag` colour tiers. */
type Verdict = 'done' | 'failed';

interface RegressionSummary {
  run: RegressionRun;
  verdict: Verdict;
  verdictLabel: string;
  headline: RegressionMetric | null;
}

function summarizeRegression(history: RegressionHistory): RegressionSummary | null {
  if (history.runs.length === 0) return null;
  const run = [...history.runs].sort(byMtimeDesc)[0];
  const regressed = run.metrics.find((m) => m.regressed);
  // Headline: the regressed metric if any, else the first metric with a delta.
  const headline = regressed ?? run.metrics.find((m) => m.delta !== null) ?? run.metrics[0] ?? null;
  return {
    run,
    verdict: run.passed ? 'done' : 'failed',
    verdictLabel: run.passed ? 'no regression' : 'regressed',
    headline,
  };
}

function newestEval(results: EvalResult[]): EvalResult | null {
  if (results.length === 0) return null;
  return [...results].sort(byMtimeDesc)[0];
}

function newestSnapshot(view: EvalHistoryView): EvalSnapshot | null {
  if (view.snapshots.length === 0) return null;
  return [...view.snapshots].sort(byMtimeDesc)[0];
}

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : 'request failed';
}

interface SourceRowProps<T> {
  label: string;
  state: SourceState<T>;
  /** Rendered only when the source loaded successfully with data. */
  children: JSX.Element;
  /** True when the source loaded but had nothing to show. */
  empty: boolean;
}

function SourceRow<T>({ label, state, children, empty }: SourceRowProps<T>): JSX.Element {
  return (
    <div className="evalsum-row">
      <div className="evalsum-label">{label}</div>
      <div className="evalsum-body">
        {!state.loaded && <span className="muted">loading…</span>}
        {state.loaded && state.error !== null && (
          <span className="evalsum-err">{state.error}</span>
        )}
        {state.loaded && state.error === null && empty && (
          <span className="muted">no eval results yet</span>
        )}
        {state.loaded && state.error === null && !empty && children}
      </div>
    </div>
  );
}

export default function CheckpointEvalSummary(): JSX.Element {
  const [evals, setEvals] = useState<SourceState<EvalResult[]>>(pending);
  const [hist, setHist] = useState<SourceState<EvalHistoryView>>(pending);
  const [regr, setRegr] = useState<SourceState<RegressionHistory>>(pending);

  useEffect(() => {
    let active = true;

    void (async () => {
      try {
        const resp: EvalResult[] = await getEvals();
        if (active) setEvals({ data: resp, error: null, loaded: true });
      } catch (e) {
        if (active) setEvals({ data: null, error: errorMessage(e), loaded: true });
      }
    })();

    void (async () => {
      try {
        const resp: EvalHistoryView | ApiError = await getEvalHistory();
        if (!active) return;
        if (isApiError(resp)) setHist({ data: null, error: resp.error, loaded: true });
        else setHist({ data: resp, error: null, loaded: true });
      } catch (e) {
        if (active) setHist({ data: null, error: errorMessage(e), loaded: true });
      }
    })();

    void (async () => {
      try {
        const resp: RegressionHistory | ApiError = await getRegressionHistory();
        if (!active) return;
        if (isApiError(resp)) setRegr({ data: null, error: resp.error, loaded: true });
        else setRegr({ data: resp, error: null, loaded: true });
      } catch (e) {
        if (active) setRegr({ data: null, error: errorMessage(e), loaded: true });
      }
    })();

    return () => {
      active = false;
    };
  }, []);

  const latestEval = evals.data === null ? null : newestEval(evals.data);
  const latestSnapshot = hist.data === null ? null : newestSnapshot(hist.data);
  const snapshotMetric =
    latestSnapshot === null ? null : pickNumericMetric(latestSnapshot.metrics);
  const regression = regr.data === null ? null : summarizeRegression(regr.data);

  return (
    <div className="card">
      <div className="row">
        <div className="card-title">Latest evaluation</div>
      </div>

      <div className="evalsum-list">
        <SourceRow label="eval artifact" state={evals} empty={latestEval === null}>
          <div className="evalsum-line">
            <span className="evalsum-name mono">{latestEval?.name}</span>
            <span className="tag">{latestEval?.kind}</span>
            <span className="muted evalsum-when">
              {formatRelativeTime(latestEval?.mtime ?? null)}
            </span>
            <div className="evalsum-summary muted">{latestEval?.summary}</div>
          </div>
        </SourceRow>

        <SourceRow
          label="auto-eval"
          state={hist}
          empty={latestSnapshot === null || snapshotMetric === null}
        >
          <div className="evalsum-line">
            <span className="evalsum-metric mono">{snapshotMetric?.key}</span>
            <span className="evalsum-value mono">
              {formatFloat(snapshotMetric?.value ?? null, 3)}
            </span>
            <span className="muted evalsum-when">
              step {formatInteger(latestSnapshot?.step ?? null)} ·{' '}
              {formatRelativeTime(latestSnapshot?.mtime ?? null)}
            </span>
          </div>
        </SourceRow>

        <SourceRow label="regression" state={regr} empty={regression === null}>
          <div className="evalsum-line">
            <span className={`tag ${regression?.verdict ?? 'done'}`}>
              {regression?.verdictLabel}
            </span>
            {regression?.headline != null && (
              <>
                <span className="evalsum-metric mono">{regression.headline.name}</span>
                <span className="evalsum-value mono">
                  {formatFloat(regression.headline.value, 3)}
                </span>
                {regression.headline.delta !== null && (
                  <span
                    className={`evalsum-delta mono ${
                      regression.headline.regressed ? 'evalsum-delta-bad' : 'evalsum-delta-ok'
                    }`}
                  >
                    {regression.headline.delta >= 0 ? '+' : ''}
                    {formatFloat(regression.headline.delta, 3)}
                  </span>
                )}
              </>
            )}
            <span className="muted evalsum-when">
              {formatRelativeTime(regression?.run.mtime ?? null)}
            </span>
          </div>
        </SourceRow>
      </div>
    </div>
  );
}
