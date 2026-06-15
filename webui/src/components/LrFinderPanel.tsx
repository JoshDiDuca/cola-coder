import { useCallback, useEffect, useState } from 'react';
import type { LrFinderResults, LrFinderRun, LrPoint } from '../types';
import { isApiError } from '../types';
import { getLrFinderResults } from '../api';
import { formatFloat, formatInteger } from '../format';

const CURVE_WIDTH = 260;
const CURVE_HEIGHT = 64;
const CURVE_PAD = 4;

// Format an LR in scientific notation, e.g. 3.0e-4. null → "—".
function formatLr(lr: number | null): string {
  if (lr === null) return '—';
  return lr.toExponential(1);
}

// Build an SVG polyline path for loss-vs-LR on a log-x axis. Points with a
// non-positive lr are skipped (log undefined). Returns null when fewer than two
// plottable points exist.
function buildCurvePath(points: LrPoint[]): string | null {
  const usable = points.filter((p) => p.lr > 0 && Number.isFinite(p.loss));
  if (usable.length < 2) return null;

  const logXs = usable.map((p) => Math.log10(p.lr));
  const losses = usable.map((p) => p.loss);

  let minX = logXs[0];
  let maxX = logXs[0];
  for (const x of logXs) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
  }
  let minY = losses[0];
  let maxY = losses[0];
  for (const y of losses) {
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  const xSpan = maxX - minX || 1;
  const ySpan = maxY - minY || 1;
  const innerW = CURVE_WIDTH - 2 * CURVE_PAD;
  const innerH = CURVE_HEIGHT - 2 * CURVE_PAD;

  const coords: string[] = [];
  for (let i = 0; i < usable.length; i += 1) {
    const px = CURVE_PAD + ((logXs[i] - minX) / xSpan) * innerW;
    // Loss low at the bottom: invert the y axis.
    const py = CURVE_PAD + (1 - (losses[i] - minY) / ySpan) * innerH;
    coords.push(`${px.toFixed(1)},${py.toFixed(1)}`);
  }
  return coords.join(' ');
}

function LossCurve({ points }: { points: LrPoint[] }) {
  const path = buildCurvePath(points);
  if (path === null) {
    return <div className="muted">not enough points to plot</div>;
  }
  return (
    <svg
      width={CURVE_WIDTH}
      height={CURVE_HEIGHT}
      viewBox={`0 0 ${CURVE_WIDTH} ${CURVE_HEIGHT}`}
      role="img"
      aria-label="loss vs learning rate (log scale)"
    >
      <polyline
        points={path}
        fill="none"
        stroke="#58a6ff"
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

function RunRow({ run }: { run: LrFinderRun }) {
  return (
    <div className="card">
      <div className="row">
        <span className="card-title mono">{run.name}</span>
        {run.suggested_lr !== null && (
          <span className="tag done">suggested {formatLr(run.suggested_lr)}</span>
        )}
      </div>
      <div className="row">
        <span className="k">loss vs LR (log-x)</span>
        <span className="v">{formatInteger(run.num_points)} pts</span>
      </div>
      <LossCurve points={run.points} />
      <div className="row">
        <span className="k">min loss</span>
        <span className="v mono">{formatFloat(run.min_loss, 3)}</span>
      </div>
      {run.config !== null && (
        <div className="row">
          <span className="k">config</span>
          <span className="v mono muted">{run.config}</span>
        </div>
      )}
    </div>
  );
}

export default function LrFinderPanel() {
  const [view, setView] = useState<LrFinderResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (): Promise<void> => {
    setError(null);
    setLoading(true);
    try {
      const resp = await getLrFinderResults();
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
        const resp = await getLrFinderResults();
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
        <div className="card-title">LR Finder Results</div>
        <button className="btn" onClick={() => void load()} disabled={loading}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}
      {loading && !view && <div className="muted">loading…</div>}

      {view && runs.length === 0 && !error && (
        <div className="muted">
          no saved LR-finder results found — find_lr.py persists only a plot image
        </div>
      )}

      {runs.length > 0 && (
        <>
          <div className="row">
            <span className="muted mono">{formatInteger(view?.count ?? runs.length)} runs</span>
          </div>
          {runs.map((run) => (
            <RunRow key={run.path} run={run} />
          ))}
        </>
      )}
    </div>
  );
}
