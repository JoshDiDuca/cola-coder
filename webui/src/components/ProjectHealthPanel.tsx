import { useCallback, useEffect, useState } from 'react';
import type { ProjectHealthReport, HealthDimension } from '../types';
import { isApiError } from '../types';
import { getProjectHealth } from '../api';
import { formatPercent } from '../format';

// Grade letter drives the overall badge colour, reusing existing tag classes.
function gradeBadgeClass(grade: string): string {
  const g = grade.trim().toUpperCase();
  if (g === 'A' || g === 'B') return 'tag done';
  if (g === 'C' || g === 'D') return 'tag running';
  return 'tag failed';
}

function DimensionRow({ dim }: { dim: HealthDimension }) {
  const pct = Math.round(dim.score * 100);
  return (
    <div className="row">
      <span className="k">{dim.name}</span>
      <span className="v">
        <span className="bar">
          <span className="bar-fill" style={{ width: `${pct}%` }} />
        </span>
        <span className="mono"> {formatPercent(dim.score)}</span>
        <span className="muted"> — {dim.detail}</span>
      </span>
    </div>
  );
}

export default function ProjectHealthPanel() {
  const [report, setReport] = useState<ProjectHealthReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (): Promise<void> => {
    setError(null);
    setLoading(true);
    try {
      const resp = await getProjectHealth();
      if (isApiError(resp)) {
        setError(resp.error);
        setReport(null);
      } else {
        setReport(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setReport(null);
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
        const resp = await getProjectHealth();
        if (!active) return;
        if (isApiError(resp)) {
          setError(resp.error);
          setReport(null);
        } else {
          setReport(resp);
        }
      } catch (e) {
        if (!active) return;
        setError(e instanceof Error ? e.message : String(e));
        setReport(null);
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const overallPct = report ? Math.round(report.overall_score * 100) : 0;

  return (
    <div className="card">
      <div className="row">
        <div className="card-title">Project Health</div>
        <button className="btn" onClick={() => void load()} disabled={loading}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}
      {loading && !report && <div className="muted">loading…</div>}

      {report && (
        <>
          <div className="row">
            <span className={gradeBadgeClass(report.grade)}>grade {report.grade}</span>
            <span className="v mono">{formatPercent(report.overall_score)} overall</span>
          </div>

          <div className="row">
            <span className="k">score</span>
            <span className="v">
              <span className="bar">
                <span className="bar-fill" style={{ width: `${overallPct}%` }} />
              </span>
            </span>
          </div>

          <div className="tbl">
            {report.dimensions.length === 0 ? (
              <div className="muted">no dimensions</div>
            ) : (
              report.dimensions.map((dim) => <DimensionRow key={dim.name} dim={dim} />)
            )}
          </div>

          <div className="row">
            <span className="v muted">{report.summary}</span>
          </div>
        </>
      )}
    </div>
  );
}
