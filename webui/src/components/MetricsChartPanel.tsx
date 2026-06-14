import { useCallback, useEffect, useState } from 'react';
import type { MetricsHistory, MetricPoint } from '../types';
import { isApiError } from '../types';
import { getMetricsHistory } from '../api';
import Sparkline from './Sparkline';

function extent(values: number[]): { min: number; max: number } | null {
  if (values.length === 0) return null;
  let min = values[0];
  let max = values[0];
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return { min, max };
}

function nums(points: MetricPoint[], key: 'loss' | 'tok_s'): number[] {
  const out: number[] = [];
  for (const p of points) {
    const v = p[key];
    if (typeof v === 'number' && Number.isFinite(v)) out.push(v);
  }
  return out;
}

export default function MetricsChartPanel() {
  const [view, setView] = useState<MetricsHistory | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const resp = await getMetricsHistory();
      if (isApiError(resp)) {
        setError(resp.error);
        setView(null);
      } else {
        setView(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const resp = await getMetricsHistory();
        if (!active) return;
        if (isApiError(resp)) setError(resp.error);
        else setView(resp);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const points = view?.points ?? [];
  const lossVals = nums(points, 'loss');
  const tokVals = nums(points, 'tok_s');
  const lossExt = extent(lossVals);
  const firstStep = points.length > 0 ? points[0].step : null;
  const lastStep = points.length > 0 ? points[points.length - 1].step : null;

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Training Metrics</div>
        <button className="btn" onClick={() => void load()}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {view && points.length === 0 && !error && (
        <div className="muted">no training log yet</div>
      )}

      {view && points.length > 0 && (
        <>
          <div className="muted mono">{view.count} points</div>

          <div className="card-title">loss</div>
          <Sparkline points={lossVals} stroke="#f85149" />
          <div className="row">
            <span className="k">
              step {firstStep ?? '?'} → {lastStep ?? '?'}
            </span>
            <span className="v">
              {lossExt
                ? `min ${lossExt.min.toFixed(3)} · max ${lossExt.max.toFixed(3)}`
                : 'no loss'}
            </span>
          </div>

          <div className="card-title">tok/s</div>
          <Sparkline points={tokVals} stroke="#3fb950" />
          <div className="row">
            <span className="k">throughput</span>
            <span className="v">
              {tokVals.length > 0
                ? `${tokVals[tokVals.length - 1].toFixed(0)} tok/s`
                : 'no throughput'}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
