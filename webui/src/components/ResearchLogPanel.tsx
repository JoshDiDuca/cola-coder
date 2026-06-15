import { useCallback, useEffect, useState } from 'react';
import { isApiError, type ResearchLog, type ResearchEntry } from '../types';
import { getResearchLog } from '../api';
import { formatInteger } from '../format';

export default function ResearchLogPanel() {
  const [log, setLog] = useState<ResearchLog | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const resp = await getResearchLog();
      if (isApiError(resp)) {
        setLog(null);
        setError(resp.error);
      } else {
        setLog(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const entries: ResearchEntry[] = log?.entries ?? [];

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Research Log</div>
        <button className="btn" onClick={() => void load()}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {log && entries.length === 0 && !error && (
        <div className="muted">no research-log entries yet</div>
      )}

      {entries.length > 0 && (
        <>
          <div className="row">
            <span className="muted mono">
              {formatInteger(log?.count ?? entries.length)} entries · newest first
            </span>
          </div>

          <table className="tbl">
            <thead>
              <tr>
                <th>date</th>
                <th>technique</th>
                <th className="right">sources</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => (
                <tr key={`${entry.date}:${entry.title}`}>
                  <td className="mono">{entry.date}</td>
                  <td>
                    <div>
                      {entry.title}{' '}
                      {entry.area !== null && <span className="tag">{entry.area}</span>}{' '}
                      {entry.has_original_idea && (
                        <span className="tag done">original idea</span>
                      )}
                    </div>
                    {entry.summary && <div className="muted">{entry.summary}</div>}
                  </td>
                  <td className="right mono">{formatInteger(entry.source_count)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {!log && !error && <div className="muted">loading research log…</div>}
    </div>
  );
}
