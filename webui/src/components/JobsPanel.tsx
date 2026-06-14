import { useCallback, useEffect, useRef, useState } from 'react';
import type { Job, JobLogChunk } from '../types';
import { jobLogStreamUrl, stopJob } from '../api';

const MAX_LOG_LINES = 5000;

type StreamState = 'streaming' | 'ended' | 'error';

interface JobsPanelProps {
  jobs: Job[];
}

// Keep only the last `max` lines of `text` to bound memory growth.
function clampLines(text: string, max: number): string {
  const lines = text.split('\n');
  if (lines.length <= max) return text;
  return lines.slice(lines.length - max).join('\n');
}

export default function JobsPanel({ jobs }: JobsPanelProps) {
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [log, setLog] = useState<string>('');
  const [streamState, setStreamState] = useState<StreamState>('streaming');
  const logRef = useRef<HTMLDivElement | null>(null);
  // Tracks whether the stream has finished (done frame) so a post-done
  // onerror does not flip us into a "reconnecting" state.
  const doneRef = useRef<boolean>(false);

  const onStop = useCallback(async (id: string) => {
    try {
      // The event stream reflects the new status within ~1s — no manual refresh.
      await stopJob(id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  // Own the EventSource lifecycle: create when a job is selected, close on
  // cleanup (job switch or unmount). Switching jobs runs cleanup first, so the
  // previous EventSource is always closed before a new one opens — no leaks.
  useEffect(() => {
    if (selectedId === null) return;

    doneRef.current = false;
    let firstFrame = true;
    setLog('');
    setStreamState('streaming');

    const source = new EventSource(jobLogStreamUrl(selectedId));

    source.onmessage = (e: MessageEvent<string>) => {
      const chunk = JSON.parse(e.data) as JobLogChunk;
      setLog((prev) => {
        const next = firstFrame ? chunk.text : prev + chunk.text;
        return clampLines(next, MAX_LOG_LINES);
      });
      firstFrame = false;
      if (chunk.done) {
        doneRef.current = true;
        setStreamState('ended');
        source.close();
      }
    };

    source.onerror = () => {
      // EventSource auto-reconnects on transient errors. Once we have seen the
      // done frame we have closed it ourselves, so don't thrash the indicator.
      if (doneRef.current) return;
      setStreamState('error');
    };

    return () => {
      doneRef.current = true;
      source.close();
    };
  }, [selectedId]);

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
                  <button className="btn" onClick={() => setSelectedId(job.id)}>
                    follow
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
          {streamState === 'ended' && <div className="muted">stream ended</div>}
          {streamState === 'error' && <div className="muted">reconnecting…</div>}
          <div className="pre scroll" ref={logRef}>
            {log || <span className="muted">no log output</span>}
          </div>
        </>
      )}
    </div>
  );
}
