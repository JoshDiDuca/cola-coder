import { useCallback, useEffect, useState } from 'react';
import type { RegressionHistory, RegressionRun, RegressionMetric } from '../types';
import { isApiError } from '../types';
import { getRegressionHistory } from '../api';
import { formatInteger, formatFloat } from '../format';

// Pass/fail badge for a whole run.
function runBadgeClass(passed: boolean): string {
  return passed ? 'tag done' : 'tag failed';
}

// Render one metric's value. The persisted regression artifact only carries
// numeric values for the run-level rows (pass_rate/passed/total); per-baseline
// rows have a null value and are summarized by their regressed flag instead.
function metricValue(metric: RegressionMetric): string {
  if (metric.value === null) {
    return metric.regressed ? 'regressed' : 'ok';
  }
  return formatFloat(metric.value, 3);
}

function metricDelta(metric: RegressionMetric): string {
  if (metric.delta === null) return '—';
  const sign = metric.delta >= 0 ? '+' : '';
  return `${sign}${formatFloat(metric.delta, 3)}`;
}

function MetricRow({ metric }: { metric: RegressionMetric }) {
  return (
    <tr>
      <td className="mono">{metric.name}</td>
      <td className="right mono">{metricValue(metric)}</td>
      <td className="right mono muted">
        {metric.baseline === null ? '—' : formatFloat(metric.baseline, 3)}
      </td>
      <td className="right mono">{metricDelta(metric)}</td>
      <td>
        <span className={metric.regressed ? 'tag failed' : 'tag done'}>
          {metric.regressed ? 'regressed' : 'ok'}
        </span>
      </td>
    </tr>
  );
}

function RunCard({ run }: { run: RegressionRun }) {
  const regressedCount = run.metrics.filter((m) => m.regressed).length;
  return (
    <div className="card">
      <div className="row">
        <span className="mono">{run.name}</span>
        <span className={runBadgeClass(run.passed)}>{run.passed ? 'passed' : 'regressed'}</span>
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
            <MetricRow key={metric.name} metric={metric} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function RegressionHistoryPanel() {
  const [view, setView] = useState<RegressionHistory | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (): Promise<void> => {
    setError(null);
    setLoading(true);
    try {
      const resp = await getRegressionHistory();
      if (isApiError(resp)) {
        setError(resp.error);
        setView(null);
      } else {
        setView(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setView(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let active = true;
    void (async () => {
      setError(null);
      setLoading(true);
      try {
        const resp = await getRegressionHistory();
        if (!active) return;
        if (isApiError(resp)) {
          setError(resp.error);
          setView(null);
        } else {
          setView(resp);
        }
      } catch (e) {
        if (!active) return;
        setError(e instanceof Error ? e.message : String(e));
        setView(null);
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const runs = view?.runs ?? [];

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Regression History</div>
        <button className="btn" onClick={() => void load()} disabled={loading}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}
      {loading && !view && <div className="muted">loading…</div>}

      {view && runs.length === 0 && !error && (
        <div className="muted">
          no saved regression reports found (run scripts/regression_test.py --save)
        </div>
      )}

      {runs.length > 0 && (
        <>
          <div className="row">
            <span className="muted mono">{formatInteger(view?.count ?? runs.length)} reports</span>
          </div>
          {runs.map((run) => (
            <RunCard key={run.path} run={run} />
          ))}
        </>
      )}
    </div>
  );
}
