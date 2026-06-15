import { useCallback, useEffect, useMemo, useState } from 'react';
import type {
  EvalResult,
  EvalDetail,
  EvalHistoryView,
  EvalSnapshot,
  BenchmarkResults,
  BenchmarkRun,
  SafetyEvalResults,
  SafetyEvalRun,
  SafetyProbe,
  RegressionHistory,
  RegressionRun,
  RegressionMetric,
  JsonValue,
} from '../../types';
import { isApiError } from '../../types';
import {
  getEvals,
  getEval,
  getEvalHistory,
  getBenchmarkResults,
  getSafetyEvalResults,
  getRegressionHistory,
} from '../../api';
import { formatFloat, formatInteger, formatRelativeTime } from '../../format';
import MasterDetail, { type MasterItem } from '../MasterDetail';
import Sparkline from '../Sparkline';

// ── Category (pinned) item identity ──────────────────────────────────────────
// Pinned category items live at the top of the unified list. Their ids are
// namespaced so they never collide with a concrete eval-artifact path id.
type CategoryId = 'cat:history' | 'cat:benchmarks' | 'cat:safety' | 'cat:regression';

const CATEGORY_IDS: readonly CategoryId[] = [
  'cat:history',
  'cat:benchmarks',
  'cat:safety',
  'cat:regression',
];

function isCategoryId(id: string): id is CategoryId {
  return (CATEGORY_IDS as readonly string[]).includes(id);
}

// A concrete eval artifact is keyed by its file path, prefixed to disambiguate
// from category items.
function artifactId(ev: EvalResult): string {
  return `art:${ev.path}`;
}

// ── JSON metric extraction (exhaustive over JsonValue, no any/unknown) ────────
interface MetricTile {
  key: string;
  value: string;
}

/** Render a single scalar JsonValue for a tile (objects/arrays excluded upstream). */
function scalarText(value: string | number | boolean): string {
  switch (typeof value) {
    case 'string':
      return value;
    case 'number':
      return String(value);
    case 'boolean':
      return value ? 'yes' : 'no';
    default: {
      const _exhaustive: never = value;
      return _exhaustive;
    }
  }
}

/**
 * Pull top-level scalar fields (string/number/boolean) out of a parsed eval
 * object so they render as metric tiles. Nested objects/arrays stay in the raw
 * `.pre` view. Exhaustive over the `JsonValue` union.
 */
function extractMetricTiles(parsed: JsonValue): MetricTile[] {
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return [];
  }
  const tiles: MetricTile[] = [];
  for (const [key, value] of Object.entries(parsed)) {
    if (value === null) continue;
    switch (typeof value) {
      case 'string':
      case 'number':
      case 'boolean':
        tiles.push({ key, value: scalarText(value) });
        break;
      case 'object':
        // Nested objects/arrays stay in the raw view.
        break;
      default:
        break;
    }
  }
  return tiles;
}

/** The ONE sanctioned JSON → string renderer, inlined for the raw `.pre` body. */
function jsonText(value: JsonValue): string {
  if (value === null) return '—';
  switch (typeof value) {
    case 'string':
      return value;
    case 'number':
      return String(value);
    case 'boolean':
      return value ? 'yes' : 'no';
    case 'object':
      return JSON.stringify(value, null, 2);
    default: {
      const _exhaustive: never = value;
      return _exhaustive;
    }
  }
}

function rawBody(d: EvalDetail): string {
  if (d.parsed !== null) {
    const text = jsonText(d.parsed);
    return d.truncated ? `${text}\n…(truncated)` : text;
  }
  const body = d.content ?? '';
  return d.truncated ? `${body}\n…(truncated)` : body;
}

function asNumber(value: JsonValue): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  return null;
}

// ── Detail sub-views ──────────────────────────────────────────────────────────

/** Concrete eval artifact: scalar metric tiles + raw content fallback. */
function ArtifactDetail({
  ev,
  detail,
  loading,
  error,
}: {
  ev: EvalResult;
  detail: EvalDetail | null;
  loading: boolean;
  error: string | null;
}): JSX.Element {
  const tiles = useMemo(() => (detail ? extractMetricTiles(detail.parsed) : []), [detail]);
  return (
    <>
      <div className="md-detail-head">
        <div>
          <h2 className="md-detail-title mono">{ev.name}</h2>
          <div className="muted mono">{formatRelativeTime(ev.mtime)}</div>
        </div>
        <span className="tag">{ev.kind}</span>
      </div>

      {ev.summary && <div className="muted">{ev.summary}</div>}
      {loading && <div className="muted">loading…</div>}
      {error && <div className="err">{error}</div>}

      {!loading && !error && detail && (
        <>
          {tiles.length > 0 && (
            <div className="stat-tiles">
              {tiles.map((t) => (
                <div className="stat-tile" key={t.key}>
                  <div className="stat-tile-label">{t.key}</div>
                  <div className="stat-tile-value mono">{t.value}</div>
                </div>
              ))}
            </div>
          )}
          <pre className="pre scroll">{rawBody(detail)}</pre>
        </>
      )}
    </>
  );
}

interface HistoryRow {
  snap: EvalSnapshot;
  value: number | null;
  delta: number | null;
}

function deltaParts(delta: number | null): { cls: string; text: string } {
  if (delta === null || delta === 0) return { cls: 'es-delta', text: '—' };
  const sign = delta > 0 ? '▲' : '▼';
  const cls = delta > 0 ? 'es-delta es-delta--up' : 'es-delta es-delta--down';
  return { cls, text: `${sign} ${formatFloat(Math.abs(delta), 4)}` };
}

/** Eval-over-training history: trend sparkline + delta table. */
function HistoryDetail({
  view,
  error,
}: {
  view: EvalHistoryView | null;
  error: string | null;
}): JSX.Element {
  const [metric, setMetric] = useState<string>('');
  const snapshots = view?.snapshots ?? [];

  useEffect(() => {
    const keys = view?.metric_keys ?? [];
    if (keys.length > 0 && !keys.includes(metric)) setMetric(keys[0]);
  }, [view, metric]);

  const rows = useMemo(() => {
    if (!metric) return [];
    const out: HistoryRow[] = [];
    let prev: number | null = null;
    for (const snap of snapshots) {
      const value = asNumber(snap.metrics[metric]);
      const delta = value !== null && prev !== null ? value - prev : null;
      out.push({ snap, value, delta });
      if (value !== null) prev = value;
    }
    return out;
  }, [snapshots, metric]);

  const series = useMemo(() => {
    const out: number[] = [];
    for (const r of rows) if (r.value !== null) out.push(r.value);
    return out;
  }, [rows]);

  const range = useMemo(() => {
    if (series.length === 0) return null;
    let min = series[0];
    let max = series[0];
    const last = series[series.length - 1];
    for (const v of series) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    return { min, max, last };
  }, [series]);

  return (
    <>
      <div className="md-detail-head">
        <h2 className="md-detail-title">Eval History</h2>
        <span className="tag">{formatInteger(view?.count ?? snapshots.length)} snapshots</span>
      </div>

      {error && <div className="err">{error}</div>}

      {view && snapshots.length === 0 && !error && (
        <div className="md-detail-empty">
          No eval-over-training snapshots yet. Snapshots accumulate as evaluation runs are saved
          across training steps.
        </div>
      )}

      {snapshots.length > 0 && (
        <>
          <div className="md-toolbar">
            <select
              className="select"
              value={metric}
              onChange={(e) => setMetric(e.target.value)}
              disabled={(view?.metric_keys.length ?? 0) === 0}
            >
              {(view?.metric_keys ?? []).length === 0 ? (
                <option value="">no numeric metrics</option>
              ) : (
                view?.metric_keys.map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))
              )}
            </select>
          </div>

          {metric && (
            <>
              {range && (
                <div className="stat-tiles">
                  <div className="stat-tile">
                    <div className="stat-tile-label">latest</div>
                    <div className="stat-tile-value mono">{formatFloat(range.last, 4)}</div>
                  </div>
                  <div className="stat-tile">
                    <div className="stat-tile-label">min</div>
                    <div className="stat-tile-value mono">{formatFloat(range.min, 4)}</div>
                  </div>
                  <div className="stat-tile">
                    <div className="stat-tile-label">max</div>
                    <div className="stat-tile-value mono">{formatFloat(range.max, 4)}</div>
                  </div>
                </div>
              )}

              <div className="es-chart">
                <div className="muted mono">{metric}</div>
                <Sparkline points={series} stroke="var(--accent)" width={640} height={96} />
              </div>

              <table className="tbl">
                <thead>
                  <tr>
                    <th className="right">step</th>
                    <th className="right">{metric}</th>
                    <th className="right">Δ</th>
                    <th className="right">when</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => {
                    const d = deltaParts(r.delta);
                    return (
                      <tr key={r.snap.path}>
                        <td className="right mono">{r.snap.step ?? '—'}</td>
                        <td className="right mono">{formatFloat(r.value, 4)}</td>
                        <td className="right mono">
                          <span className={d.cls}>{d.text}</span>
                        </td>
                        <td className="right mono muted">{formatRelativeTime(r.snap.mtime)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </>
          )}
        </>
      )}
    </>
  );
}

function benchKindClass(kind: string): string {
  switch (kind) {
    case 'throughput':
      return 'tag done';
    case 'latency':
      return 'tag running';
    case 'nano':
      return 'tag';
    default:
      return 'tag failed';
  }
}

function BenchmarkRow({ run }: { run: BenchmarkRun }): JSX.Element {
  return (
    <tr>
      <td className="mono">{run.name}</td>
      <td>
        <span className={benchKindClass(run.kind)}>{run.kind}</span>
      </td>
      <td className="right mono">
        {run.tokens_per_s === null ? '—' : `${formatFloat(run.tokens_per_s, 1)} tok/s`}
      </td>
      <td className="right mono">
        {run.latency_ms === null ? '—' : `${formatFloat(run.latency_ms, 1)} ms`}
      </td>
      <td className="mono muted">{run.checkpoint ?? '—'}</td>
    </tr>
  );
}

function BenchmarkDetail({
  view,
  error,
}: {
  view: BenchmarkResults | null;
  error: string | null;
}): JSX.Element {
  const runs = view?.runs ?? [];
  return (
    <>
      <div className="md-detail-head">
        <h2 className="md-detail-title">Benchmarks</h2>
        <span className="tag">{formatInteger(view?.count ?? runs.length)} reports</span>
      </div>
      {error && <div className="err">{error}</div>}
      {view && runs.length === 0 && !error && (
        <div className="md-detail-empty">
          No saved benchmark reports found. Run <span className="mono">scripts/benchmark.py</span>,{' '}
          <span className="mono">inference_benchmark.py</span>, or{' '}
          <span className="mono">nano_benchmark.py</span> to populate throughput / latency results.
        </div>
      )}
      {runs.length > 0 && (
        <table className="tbl">
          <thead>
            <tr>
              <th>report</th>
              <th>kind</th>
              <th className="right">throughput</th>
              <th className="right">first token</th>
              <th>checkpoint</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <BenchmarkRow key={run.path} run={run} />
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}

function SafetyProbeRow({ probe }: { probe: SafetyProbe }): JSX.Element {
  return (
    <div className="row">
      <span className="k">
        <span className={probe.passed ? 'tag done' : 'tag failed'}>
          {probe.passed ? 'PASS' : 'FAIL'}
        </span>{' '}
        <span className="tag">{probe.suite}</span> {probe.name}
      </span>
      <span className="v muted mono">{probe.detail ?? ''}</span>
    </div>
  );
}

function SafetyRunCard({ run }: { run: SafetyEvalRun }): JSX.Element {
  return (
    <div className="card">
      <div className="row">
        <span className="k">
          <span className={run.failed === 0 ? 'tag done' : 'tag failed'}>{run.suite}</span>{' '}
          <span className="mono">{run.name}</span>
        </span>
        <span className="v muted">
          {formatInteger(run.passed)}/{formatInteger(run.total)} passed
        </span>
      </div>
      <div className="row">
        <span className="k">checkpoint</span>
        <span className="v mono">{run.checkpoint ?? '—'}</span>
      </div>
      {run.probes.length === 0 ? (
        <div className="muted">no per-probe detail</div>
      ) : (
        run.probes.map((probe, i) => (
          <SafetyProbeRow key={`${probe.suite}:${probe.name}:${i}`} probe={probe} />
        ))
      )}
    </div>
  );
}

function SafetyDetail({
  view,
  error,
}: {
  view: SafetyEvalResults | null;
  error: string | null;
}): JSX.Element {
  const runs = view?.runs ?? [];
  return (
    <>
      <div className="md-detail-head">
        <h2 className="md-detail-title">Safety</h2>
        <span className="tag">{formatInteger(view?.count ?? runs.length)} runs</span>
      </div>
      {error && <div className="err">{error}</div>}
      {view && runs.length === 0 && !error && (
        <div className="md-detail-empty">
          No safety-eval results found. Run{' '}
          <span className="mono">scripts/safety_eval.py --suite extended</span> to probe generated
          code for secrets, dangerous patterns, and hallucinations.
        </div>
      )}
      {runs.length > 0 && runs.map((run) => <SafetyRunCard key={run.path} run={run} />)}
    </>
  );
}

function regressionMetricValue(metric: RegressionMetric): string {
  if (metric.value === null) return metric.regressed ? 'regressed' : 'ok';
  return formatFloat(metric.value, 3);
}

function regressionMetricDelta(metric: RegressionMetric): string {
  if (metric.delta === null) return '—';
  const sign = metric.delta >= 0 ? '+' : '';
  return `${sign}${formatFloat(metric.delta, 3)}`;
}

function RegressionRunCard({ run }: { run: RegressionRun }): JSX.Element {
  const regressedCount = run.metrics.filter((m) => m.regressed).length;
  return (
    <div className="card">
      <div className="row">
        <span className="mono">{run.name}</span>
        <span className={run.passed ? 'tag done' : 'tag failed'}>
          {run.passed ? 'passed' : 'regressed'}
        </span>
      </div>
      <div className="row">
        <span className="muted mono">{run.checkpoint ?? 'unknown checkpoint'}</span>
        <span className="muted mono">
          {formatInteger(regressedCount)} regressed / {formatInteger(run.metrics.length)} metrics
        </span>
      </div>
      <table className="tbl">
        <thead>
          <tr>
            <th>metric</th>
            <th className="right">value</th>
            <th className="right">baseline</th>
            <th className="right">delta</th>
            <th>status</th>
          </tr>
        </thead>
        <tbody>
          {run.metrics.map((metric) => (
            <tr key={metric.name}>
              <td className="mono">{metric.name}</td>
              <td className="right mono">{regressionMetricValue(metric)}</td>
              <td className="right mono muted">
                {metric.baseline === null ? '—' : formatFloat(metric.baseline, 3)}
              </td>
              <td className="right mono">{regressionMetricDelta(metric)}</td>
              <td>
                <span className={metric.regressed ? 'tag failed' : 'tag done'}>
                  {metric.regressed ? 'regressed' : 'ok'}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RegressionDetail({
  view,
  error,
}: {
  view: RegressionHistory | null;
  error: string | null;
}): JSX.Element {
  const runs = view?.runs ?? [];
  return (
    <>
      <div className="md-detail-head">
        <h2 className="md-detail-title">Regression</h2>
        <span className="tag">{formatInteger(view?.count ?? runs.length)} reports</span>
      </div>
      {error && <div className="err">{error}</div>}
      {view && runs.length === 0 && !error && (
        <div className="md-detail-empty">
          No saved regression reports found. Run{' '}
          <span className="mono">scripts/regression_test.py --save</span> to track quality
          regressions against a baseline checkpoint.
        </div>
      )}
      {runs.length > 0 && runs.map((run) => <RegressionRunCard key={run.path} run={run} />)}
    </>
  );
}

// ── Screen ────────────────────────────────────────────────────────────────────

export default function EvalScreen(): JSX.Element {
  const [evals, setEvals] = useState<EvalResult[]>([]);
  const [listError, setListError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Concrete eval artifact detail (lazily fetched per selection).
  const [artDetail, setArtDetail] = useState<EvalDetail | null>(null);
  const [artLoading, setArtLoading] = useState(false);
  const [artError, setArtError] = useState<string | null>(null);

  // Category datasets (fetched once on mount).
  const [history, setHistory] = useState<EvalHistoryView | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [benchmarks, setBenchmarks] = useState<BenchmarkResults | null>(null);
  const [benchmarksError, setBenchmarksError] = useState<string | null>(null);
  const [safety, setSafety] = useState<SafetyEvalResults | null>(null);
  const [safetyError, setSafetyError] = useState<string | null>(null);
  const [regression, setRegression] = useState<RegressionHistory | null>(null);
  const [regressionError, setRegressionError] = useState<string | null>(null);

  const loadEvals = useCallback(async (): Promise<EvalResult[]> => {
    setListError(null);
    try {
      const next = await getEvals();
      setEvals(next);
      return next;
    } catch (e) {
      setListError(e instanceof Error ? e.message : String(e));
      return [];
    }
  }, []);

  // Initial load: artifacts list + all category datasets, each isolated so one
  // failing endpoint never blocks the others. Each loader resolves its own state.
  useEffect(() => {
    let active = true;

    void (async () => {
      try {
        const next = await getEvals();
        if (!active) return;
        setEvals(next);
        // Default-select the first real artifact, else the History category.
        setSelectedId(next[0] ? artifactId(next[0]) : 'cat:history');
      } catch (e) {
        if (!active) return;
        setListError(e instanceof Error ? e.message : String(e));
        setSelectedId('cat:history');
      }
    })();

    void (async () => {
      try {
        const resp = await getEvalHistory();
        if (!active) return;
        if (isApiError(resp)) setHistoryError(resp.error);
        else setHistory(resp);
      } catch (e) {
        if (active) setHistoryError(e instanceof Error ? e.message : String(e));
      }
    })();

    void (async () => {
      try {
        const resp = await getBenchmarkResults();
        if (!active) return;
        if (isApiError(resp)) setBenchmarksError(resp.error);
        else setBenchmarks(resp);
      } catch (e) {
        if (active) setBenchmarksError(e instanceof Error ? e.message : String(e));
      }
    })();

    void (async () => {
      try {
        const resp = await getSafetyEvalResults();
        if (!active) return;
        if (isApiError(resp)) setSafetyError(resp.error);
        else setSafety(resp);
      } catch (e) {
        if (active) setSafetyError(e instanceof Error ? e.message : String(e));
      }
    })();

    void (async () => {
      try {
        const resp = await getRegressionHistory();
        if (!active) return;
        if (isApiError(resp)) setRegressionError(resp.error);
        else setRegression(resp);
      } catch (e) {
        if (active) setRegressionError(e instanceof Error ? e.message : String(e));
      }
    })();

    return () => {
      active = false;
    };
  }, []);

  // Fetch concrete-artifact detail whenever a non-category item is selected.
  useEffect(() => {
    if (selectedId === null || isCategoryId(selectedId)) {
      setArtDetail(null);
      setArtError(null);
      return;
    }
    const path = selectedId.slice('art:'.length);
    let active = true;
    setArtLoading(true);
    setArtDetail(null);
    setArtError(null);
    void (async () => {
      try {
        const d = await getEval(path);
        if (!active) return;
        if (isApiError(d)) setArtError(d.error);
        else setArtDetail(d);
      } catch (e) {
        if (active) setArtError(e instanceof Error ? e.message : String(e));
      } finally {
        if (active) setArtLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, [selectedId]);

  // Unified list: pinned categories first, then concrete artifacts.
  const items = useMemo<MasterItem[]>(() => {
    const categoryItems: MasterItem[] = [
      {
        id: 'cat:history',
        title: 'Eval History',
        subtitle: 'metric trend over training',
        meta: `${formatInteger(history?.count ?? 0)} snapshots`,
        badge: <span className="tag">trend</span>,
      },
      {
        id: 'cat:benchmarks',
        title: 'Benchmarks',
        subtitle: 'throughput & latency',
        meta: `${formatInteger(benchmarks?.count ?? 0)} reports`,
        badge: <span className="tag">speed</span>,
      },
      {
        id: 'cat:safety',
        title: 'Safety',
        subtitle: 'secrets / dangerous patterns',
        meta: `${formatInteger(safety?.count ?? 0)} runs`,
        badge: <span className="tag">safety</span>,
      },
      {
        id: 'cat:regression',
        title: 'Regression',
        subtitle: 'quality vs baseline',
        meta: `${formatInteger(regression?.count ?? 0)} reports`,
        badge: <span className="tag">regression</span>,
      },
    ];

    const artifactItems: MasterItem[] = evals.map((ev) => ({
      id: artifactId(ev),
      title: ev.name,
      subtitle: ev.summary || undefined,
      meta: formatRelativeTime(ev.mtime),
      badge: <span className="tag">{ev.kind}</span>,
    }));

    return [...categoryItems, ...artifactItems];
  }, [evals, history, benchmarks, safety, regression]);

  const selectedEval = useMemo<EvalResult | null>(() => {
    if (selectedId === null || isCategoryId(selectedId)) return null;
    const path = selectedId.slice('art:'.length);
    return evals.find((ev) => ev.path === path) ?? null;
  }, [selectedId, evals]);

  const detail = useMemo<JSX.Element | null>(() => {
    if (selectedId === null) return null;
    switch (selectedId) {
      case 'cat:history':
        return <HistoryDetail view={history} error={historyError} />;
      case 'cat:benchmarks':
        return <BenchmarkDetail view={benchmarks} error={benchmarksError} />;
      case 'cat:safety':
        return <SafetyDetail view={safety} error={safetyError} />;
      case 'cat:regression':
        return <RegressionDetail view={regression} error={regressionError} />;
      default:
        if (selectedEval === null) return null;
        return (
          <ArtifactDetail
            ev={selectedEval}
            detail={artDetail}
            loading={artLoading}
            error={artError}
          />
        );
    }
  }, [
    selectedId,
    selectedEval,
    history,
    historyError,
    benchmarks,
    benchmarksError,
    safety,
    safetyError,
    regression,
    regressionError,
    artDetail,
    artLoading,
    artError,
  ]);

  return (
    <div className="es-screen">
      <div className="md-toolbar es-head">
        <h1 className="md-detail-title">Evaluation</h1>
        <button className="btn" onClick={() => void loadEvals()}>
          refresh
        </button>
      </div>

      {listError && <div className="err">{listError}</div>}

      <MasterDetail
        items={items}
        selectedId={selectedId}
        onSelect={setSelectedId}
        detail={detail}
        listLabel="Evaluations"
        emptyList="No eval artifacts found"
        emptyDetail="Select an evaluation to see details"
      />
    </div>
  );
}
