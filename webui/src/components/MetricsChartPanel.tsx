import { useCallback, useEffect, useState } from 'react';
import type { MetricsHistory, MetricPoint } from '../types';
import { isApiError } from '../types';
import { getMetricsHistory } from '../api';
import { formatFloat, formatInteger } from '../format';

type Series = 'loss' | 'tok_s';

interface SeriesStyle {
  label: string;
  stroke: string;
  unit: string;
  /** How to render the value callouts for this series. */
  format: (value: number) => string;
}

const SERIES: Record<Series, SeriesStyle> = {
  loss: {
    label: 'loss',
    stroke: 'var(--bad)',
    unit: '',
    format: (v) => formatFloat(v, 3),
  },
  tok_s: {
    label: 'tok/s',
    stroke: 'var(--good)',
    unit: ' tok/s',
    format: (v) => formatInteger(v),
  },
};

// Chart viewBox geometry. The SVG scales responsively via CSS, but all path
// math is computed in this fixed coordinate space.
const VIEW_W = 720;
const VIEW_H = 260;
const PAD_L = 52;
const PAD_R = 16;
const PAD_T = 16;
const PAD_B = 30;
const PLOT_W = VIEW_W - PAD_L - PAD_R;
const PLOT_H = VIEW_H - PAD_T - PAD_B;
const GRID_ROWS = 4;

interface SampledPoint {
  step: number;
  value: number;
}

interface Bounds {
  min: number;
  max: number;
}

/** Extract finite (step, value) pairs for a series, dropping nulls/NaN. */
function samples(points: MetricPoint[], key: Series): SampledPoint[] {
  const out: SampledPoint[] = [];
  for (const p of points) {
    const v = p[key];
    if (typeof v === 'number' && Number.isFinite(v)) out.push({ step: p.step, value: v });
  }
  return out;
}

function valueBounds(samps: SampledPoint[]): Bounds | null {
  if (samps.length === 0) return null;
  let min = samps[0].value;
  let max = samps[0].value;
  for (const s of samps) {
    if (s.value < min) min = s.value;
    if (s.value > max) max = s.value;
  }
  // Pad a flat series so it renders as a centred line, not a clipped edge.
  if (min === max) {
    const delta = Math.abs(min) > 0 ? Math.abs(min) * 0.1 : 1;
    return { min: min - delta, max: max + delta };
  }
  return { min, max };
}

function xFor(step: number, stepMin: number, stepSpan: number): number {
  if (stepSpan === 0) return PAD_L + PLOT_W / 2;
  return PAD_L + ((step - stepMin) / stepSpan) * PLOT_W;
}

function yFor(value: number, bounds: Bounds): number {
  const span = bounds.max - bounds.min;
  if (span === 0) return PAD_T + PLOT_H / 2;
  return PAD_T + (1 - (value - bounds.min) / span) * PLOT_H;
}

/** Catmull-Rom → cubic-bezier smoothing for a clean, library-free curve. */
function smoothPath(coords: { x: number; y: number }[]): string {
  if (coords.length === 0) return '';
  if (coords.length === 1) {
    const p = coords[0];
    return `M ${p.x.toFixed(2)} ${p.y.toFixed(2)} L ${(p.x + 1).toFixed(2)} ${p.y.toFixed(2)}`;
  }
  let d = `M ${coords[0].x.toFixed(2)} ${coords[0].y.toFixed(2)}`;
  for (let i = 0; i < coords.length - 1; i += 1) {
    const p0 = coords[i === 0 ? 0 : i - 1];
    const p1 = coords[i];
    const p2 = coords[i + 1];
    const p3 = coords[i + 2 < coords.length ? i + 2 : i + 1];
    const cp1x = p1.x + (p2.x - p0.x) / 6;
    const cp1y = p1.y + (p2.y - p0.y) / 6;
    const cp2x = p2.x - (p3.x - p1.x) / 6;
    const cp2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C ${cp1x.toFixed(2)} ${cp1y.toFixed(2)}, ${cp2x.toFixed(2)} ${cp2y.toFixed(2)}, ${p2.x.toFixed(2)} ${p2.y.toFixed(2)}`;
  }
  return d;
}

interface ChartProps {
  samps: SampledPoint[];
  style: SeriesStyle;
  gradientId: string;
}

function Chart({ samps, style, gradientId }: ChartProps): JSX.Element {
  const bounds = valueBounds(samps);
  if (bounds === null) {
    return <div className="chart-empty muted">no data for this series</div>;
  }

  const stepMin = samps[0].step;
  const stepMax = samps[samps.length - 1].step;
  const stepSpan = stepMax - stepMin;

  const coords = samps.map((s) => ({
    x: xFor(s.step, stepMin, stepSpan),
    y: yFor(s.value, bounds),
  }));

  const line = smoothPath(coords);
  const baseline = PAD_T + PLOT_H;
  const last = coords[coords.length - 1];
  const area =
    coords.length > 0
      ? `${line} L ${last.x.toFixed(2)} ${baseline} L ${coords[0].x.toFixed(2)} ${baseline} Z`
      : '';

  // Horizontal gridlines + y-axis ticks, evenly spaced across the value range.
  const rows = Array.from({ length: GRID_ROWS + 1 }, (_, i) => {
    const t = i / GRID_ROWS;
    const y = PAD_T + t * PLOT_H;
    const value = bounds.max - t * (bounds.max - bounds.min);
    return { y, value };
  });

  return (
    <svg
      className="chart-svg"
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={`${style.label} over training steps`}
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={style.stroke} stopOpacity={0.28} />
          <stop offset="100%" stopColor={style.stroke} stopOpacity={0} />
        </linearGradient>
      </defs>

      {rows.map((r) => (
        <g key={r.y}>
          <line
            className="chart-grid"
            x1={PAD_L}
            y1={r.y.toFixed(2)}
            x2={PAD_L + PLOT_W}
            y2={r.y.toFixed(2)}
          />
          <text className="chart-axis-label" x={PAD_L - 8} y={r.y + 3.5} textAnchor="end">
            {style.format(r.value)}
          </text>
        </g>
      ))}

      <text className="chart-axis-label" x={PAD_L} y={VIEW_H - 8} textAnchor="start">
        {formatInteger(stepMin)}
      </text>
      <text className="chart-axis-label" x={PAD_L + PLOT_W} y={VIEW_H - 8} textAnchor="end">
        {formatInteger(stepMax)}
      </text>

      {area && <path className="chart-area" d={area} fill={`url(#${gradientId})`} />}
      {line && <path className="chart-line" d={line} stroke={style.stroke} />}

      {coords.length > 0 && (
        <circle className="chart-dot" cx={last.x} cy={last.y} r={3.5} fill={style.stroke} />
      )}
    </svg>
  );
}

export default function MetricsChartPanel() {
  const [view, setView] = useState<MetricsHistory | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [series, setSeries] = useState<Series>('loss');

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
  const lossSamps = samples(points, 'loss');
  const tokSamps = samples(points, 'tok_s');
  const activeSamps = series === 'loss' ? lossSamps : tokSamps;
  const activeStyle = SERIES[series];

  const lossBounds = valueBounds(lossSamps);
  const lastTok = tokSamps.length > 0 ? tokSamps[tokSamps.length - 1].value : null;
  const lastLoss = lossSamps.length > 0 ? lossSamps[lossSamps.length - 1].value : null;
  const lastActive = activeSamps.length > 0 ? activeSamps[activeSamps.length - 1].value : null;

  const firstStep = points.length > 0 ? points[0].step : null;
  const lastStep = points.length > 0 ? points[points.length - 1].step : null;

  const hasData = view !== null && points.length > 0;

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Training Metrics</div>
        <button className="btn" onClick={() => void load()}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {!error && view === null && <div className="muted">loading…</div>}

      {!error && view !== null && points.length === 0 && (
        <div className="muted">no training log yet</div>
      )}

      {hasData && (
        <>
          <div className="chart">
            <div className="chart-toolbar">
              <div className="chart-legend">
                {(Object.keys(SERIES) as Series[]).map((key) => {
                  const s = SERIES[key];
                  const selected = key === series;
                  return (
                    <button
                      key={key}
                      type="button"
                      className={`chart-legend-item${selected ? ' active' : ''}`}
                      aria-pressed={selected}
                      onClick={() => setSeries(key)}
                    >
                      <span className="chart-legend-swatch" style={{ background: s.stroke }} />
                      {s.label}
                    </button>
                  );
                })}
              </div>
              <div className="chart-callout">
                <span className="chart-callout-label">last {activeStyle.label}</span>
                <span className="chart-callout-value mono" style={{ color: activeStyle.stroke }}>
                  {lastActive === null ? '—' : `${activeStyle.format(lastActive)}${activeStyle.unit}`}
                </span>
              </div>
            </div>

            <Chart samps={activeSamps} style={activeStyle} gradientId={`chart-grad-${series}`} />
          </div>

          <div className="stat-tiles">
            <div className="stat-tile">
              <div className="stat-tile-label">steps</div>
              <div className="stat-tile-value mono">
                {formatInteger(firstStep)} → {formatInteger(lastStep)}
              </div>
            </div>
            <div className="stat-tile">
              <div className="stat-tile-label">points</div>
              <div className="stat-tile-value mono">{formatInteger(view.count)}</div>
            </div>
            <div className="stat-tile">
              <div className="stat-tile-label">loss range</div>
              <div className="stat-tile-value mono">
                {lossBounds === null
                  ? '—'
                  : `${formatFloat(lossBounds.min, 3)} – ${formatFloat(lossBounds.max, 3)}`}
              </div>
            </div>
            <div className="stat-tile">
              <div className="stat-tile-label">last tok/s</div>
              <div className="stat-tile-value mono">{formatInteger(lastTok)}</div>
            </div>
          </div>

          <div className="muted mono chart-footnote">
            latest loss {lastLoss === null ? '—' : formatFloat(lastLoss, 3)} · latest throughput{' '}
            {lastTok === null ? '—' : `${formatInteger(lastTok)} tok/s`}
          </div>
        </>
      )}
    </div>
  );
}
