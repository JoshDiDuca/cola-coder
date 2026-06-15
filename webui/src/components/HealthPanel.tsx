import { useCallback, useEffect, useState } from 'react';
import type { HealthSummary, HealthCheck } from '../types';
import { isApiError } from '../types';
import { getHealth } from '../api';

type ScoreTier = 'done' | 'running' | 'failed';

/** Score → status tier reusing the existing tag colour classes. */
function scoreTag(score: number): ScoreTier {
  if (score >= 80) return 'done';
  if (score >= 50) return 'running';
  return 'failed';
}

function CheckRow({ check }: { check: HealthCheck }) {
  return (
    <div className={`check-row ${check.ok ? 'check-ok' : 'check-bad'}`}>
      <span className={`dot ${check.ok ? 'live' : 'dead'}`} aria-hidden="true" />
      <span className="check-name mono">{check.name}</span>
      <span className="check-detail muted">{check.detail}</span>
    </div>
  );
}

export default function HealthPanel() {
  const [view, setView] = useState<HealthSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const resp = await getHealth();
      if (isApiError(resp)) {
        setError(resp.error);
        setView(null);
      } else {
        setView(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setView(null);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const passed = view?.checks.filter((c) => c.ok).length ?? 0;
  const total = view?.checks.length ?? 0;

  return (
    <div className="card health-card">
      <div className="row">
        <div className="card-title">Health</div>
        <button className="btn" onClick={() => void load()}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {view && (
        <>
          <div className="health-head">
            <span className={`health-score stat-big mono tag ${scoreTag(view.score)}`}>
              {view.score}
            </span>
            <div className="health-summary">
              <div className="health-summary-text">{view.summary}</div>
              <div className="health-count muted mono">
                {passed} / {total} checks passing
              </div>
            </div>
          </div>

          {view.checks.length === 0 ? (
            <div className="muted">no checks</div>
          ) : (
            <div className="check-list">
              {view.checks.map((check) => (
                <CheckRow key={check.name} check={check} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
