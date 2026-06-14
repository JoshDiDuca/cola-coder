import { useCallback, useEffect, useRef, useState } from 'react';
import type { Job } from '../types';
import { getJobLog, stopJob } from '../api';

const LOG_LINES = 200;

export default function JobsPanel({ jobs }: { jobs: Job[] }) {
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [log, setLog] = useState<string>('');
  const [logError, setLogError] = useState<string | null>(null);
  const [logLoading, setLogLoading] = useState(false);
  const logRef = useRef<HTMLDivElement | null>(null);

  const loadLog = useCallback(async (id: string) => {
    setSelectedId(id);
    setLogLoading(true);
    setLogError(null);
    try {
      const res = await getJobLog(id, LOG_LINES);
      setLog(res.log);
    } catch (e) {
      setLog('');
      setLogError(e instanceof Error ? e.message : String(e));
    } finally {
      setLogLoading(false);
    }
  }, []);

  const onStop = useCallback(async (id: string) => {
    try {
      // The event stream reflects the new status within ~1s — no manual refresh.
      await stopJob(id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  // Auto-scroll the log view to the bottom whenever its content changes.
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [log]);

  return (
    <div className="card card-wide">
      <div className="card-title">Background jobs</div>

      {error && <div className="err">{error}</div>}

      {jobs.length === 0 ? (
        <div className="muted">no jobs yet</div>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>name</th>
              <th>status</th>
              <th>pid</th>
              <th className="right">actions</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr key={job.id}>
                <td>{job.name}</td>
                <td>
                  <span className={`tag ${job.status}`}>{job.status}</span>
                </td>
                <td className="mono">{job.pid}</td>
                <td className="right">
                  <button className="btn" onClick={() => void loadLog(job.id)}>
                    log
                  </button>
                  {job.status === 'running' && (
                    <button className="btn btn-danger" onClick={() => void onStop(job.id)}>
                      stop
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {selectedId !== null && (
        <>
          {logError && <div className="err">{logError}</div>}
          <div className="pre scroll" ref={logRef}>
            {logLoading ? 'loading…' : log || <span className="muted">no log output</span>}
          </div>
        </>
      )}
    </div>
  );
}
