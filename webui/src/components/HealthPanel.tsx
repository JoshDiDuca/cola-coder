import { useCallback, useEffect, useState } from 'react';
import type { HealthSummary, HealthCheck } from '../types';
import { isApiError } from '../types';
import { getHealth } from '../api';

// Score → status tier reusing the existing tag colour classes.
function scoreTag(score: number): 'done' | 'running' | 'failed' {
  if (score >= 80) return 'done';
  if (score >= 50) return 'running';
  return 'failed';
}

function CheckRow({ check }: { check: HealthCheck }) {
  return (
    <div className="row">
      <span className="mono">
        <span className={`dot ${check.ok ? 'live' : 'dead'}`} /> {check.name}
      </span>
      <span className="v muted">{check.detail}</span>
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

  return (
    <div className="card">
      <div className="row">
        <div className="card-title">Health</div>
        <button className="btn" onClick={() => void load()}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {view && (
        <>
          <div className="row">
            <span className={`stat-big tag ${scoreTag(view.score)}`}>{view.score}</span>
            <span className="stat-sub">{view.summary}</span>
          </div>

          {view.checks.length === 0 ? (
            <div className="muted">no checks</div>
          ) : (
            view.checks.map((check) => <CheckRow key={check.name} check={check} />)
          )}
        </>
      )}
    </div>
  );
}
