import { useCallback, useEffect, useMemo, useState } from 'react';
import type { EvalResult, EvalDetail, JsonValue } from '../types';
import { isApiError } from '../types';
import { getEvals, getEval } from '../api';
import { formatJsonValue, formatRelativeTime } from '../format';

/** A flat, displayable metric extracted from a parsed eval object. */
interface MetricTile {
  key: string;
  value: string;
}

/**
 * Pull top-level scalar fields (string/number/boolean) out of a parsed eval
 * object so they can render as metric tiles. Nested objects/arrays are left to
 * the raw `.pre` view. Exhaustive over the `JsonValue` union — no any/unknown.
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
        tiles.push({ key, value: formatJsonValue(value) });
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

/** Raw text body for the `.pre` fallback view. */
function rawBody(d: EvalDetail): string {
  if (d.parsed !== null) {
    const text = formatJsonValue(d.parsed);
    return d.truncated ? `${text}\n…(truncated)` : text;
  }
  const body = d.content ?? '';
  return d.truncated ? `${body}\n…(truncated)` : body;
}

export default function EvalsPanel() {
  const [evals, setEvals] = useState<EvalResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [selected, setSelected] = useState<EvalResult | null>(null);
  const [detail, setDetail] = useState<EvalDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const load = useCallback(async () => {
    setError(null);
    try {
      setEvals(await getEvals());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const next = await getEvals();
        if (active) setEvals(next);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const onView = useCallback(async (ev: EvalResult) => {
    setSelected(ev);
    setDetailLoading(true);
    setDetail(null);
    setDetailError(null);

    try {
      const d = await getEval(ev.path);
      if (isApiError(d)) {
        setDetailError(d.error);
      } else {
        setDetail(d);
      }
    } catch (e) {
      setDetailError(e instanceof Error ? e.message : String(e));
    } finally {
      setDetailLoading(false);
    }
  }, []);

  const tiles = useMemo(
    () => (detail ? extractMetricTiles(detail.parsed) : []),
    [detail],
  );

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Evals</div>
        <button className="btn" onClick={() => void load()}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {evals.length === 0 && !error ? (
        <div className="muted">no eval artifacts found</div>
      ) : (
        <ul className="eval-list">
          {evals.map((ev) => {
            const active = selected?.path === ev.path;
            return (
              <li key={ev.path}>
                <button
                  type="button"
                  className={active ? 'eval-item eval-item--active' : 'eval-item'}
                  onClick={() => void onView(ev)}
                >
                  <div className="eval-item-head">
                    <span className="eval-item-name mono">{ev.name}</span>
                    <span className="tag">{ev.kind}</span>
                  </div>
                  {ev.summary && <div className="eval-item-summary">{ev.summary}</div>}
                  <div className="eval-item-meta muted mono">
                    {formatRelativeTime(ev.mtime)}
                  </div>
                </button>
              </li>
            );
          })}
        </ul>
      )}

      {selected !== null && (
        <div className="eval-detail">
          <div className="row">
            <div className="card-title mono">{selected.name}</div>
            <span className="tag">{selected.kind}</span>
          </div>

          {detailLoading && <div className="muted">loading…</div>}
          {detailError && <div className="err">{detailError}</div>}

          {!detailLoading && !detailError && detail && (
            <>
              {tiles.length > 0 && (
                <div className="stat-tiles eval-tiles">
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
        </div>
      )}
    </div>
  );
}
