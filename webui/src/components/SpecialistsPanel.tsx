import { useCallback, useEffect, useState } from 'react';
import type { SpecialistEntry, SpecialistsView } from '../types';
import { isApiError } from '../types';
import { getSpecialists } from '../api';
import { formatPercent } from '../format';

function SpecialistRow({ entry }: { entry: SpecialistEntry }): JSX.Element {
  return (
    <div className="spec-row">
      <div className="spec-head">
        <span className="tag">{entry.domain}</span>
        {entry.confidence_threshold != null && (
          <span className="spec-threshold muted">
            threshold {formatPercent(entry.confidence_threshold)}
          </span>
        )}
      </div>

      <div className="spec-kv">
        <span className="k">checkpoint</span>
        <span className="mono">{entry.checkpoint}</span>
      </div>

      {entry.config != null && (
        <div className="spec-kv">
          <span className="k">config</span>
          <span className="mono">{entry.config}</span>
        </div>
      )}

      {(entry.keywords?.length ?? 0) > 0 && (
        <div className="spec-chips">
          {(entry.keywords ?? []).map((kw, i) => (
            <span key={`${i}-${kw}`} className="spec-chip">
              {kw}
            </span>
          ))}
        </div>
      )}

      {entry.description !== null && <div className="muted spec-desc">{entry.description}</div>}
    </div>
  );
}

export default function SpecialistsPanel(): JSX.Element {
  const [view, setView] = useState<SpecialistsView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const load = useCallback(async (): Promise<void> => {
    setError(null);
    setLoading(true);
    try {
      const resp = await getSpecialists();
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
      try {
        const resp = await getSpecialists();
        if (!active) return;
        if (isApiError(resp)) {
          setError(resp.error);
          setView(null);
        } else {
          setView(resp);
        }
      } catch (e) {
        if (active) {
          setError(e instanceof Error ? e.message : String(e));
          setView(null);
        }
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="card">
      <div className="row">
        <div className="card-title">Domain specialists</div>
        <button className="btn" onClick={() => void load()}>
          refresh
        </button>
      </div>

      <div className="muted spec-help">
        Router&rarr;specialist registry from <span className="mono">configs/specialists.yaml</span>:
        per-domain checkpoints the 125M router can dispatch requests to.
      </div>

      {loading && <div className="muted">loading&hellip;</div>}

      {!loading && error !== null && <div className="err">{error}</div>}

      {!loading && error === null && view !== null && (
        <>
          {!view.exists && (
            <div className="muted spec-empty">
              <span className="mono">specialists.yaml</span> not found at{' '}
              <span className="mono">{view.path}</span>.
            </div>
          )}

          {view.exists && view.count === 0 && (
            <div className="muted spec-empty">
              No specialists registered yet. Add entries to{' '}
              <span className="mono">configs/specialists.yaml</span> (domain &rarr; checkpoint,
              keywords, threshold) to route requests to per-domain models.
              <div className="spec-empty-path">
                <span className="k">path</span> <span className="mono">{view.path}</span>
              </div>
            </div>
          )}

          {view.exists && view.count > 0 && (
            <div className="spec-list">
              {view.specialists.map((entry) => (
                <SpecialistRow key={entry.domain} entry={entry} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
