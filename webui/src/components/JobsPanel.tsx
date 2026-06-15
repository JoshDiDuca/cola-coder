import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Job, JobLogChunk } from '../types';
import { jobLogStreamUrl, stopJob } from '../api';
import { formatRelativeTime } from '../format';

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

// Human label for the stream indicator. Exhaustive over StreamState.
function streamLabel(state: StreamState): string {
  switch (state) {
    case 'streaming':
      return 'live';
    case 'ended':
      return 'stream ended';
    case 'error':
      return 'reconnecting…';
    default: {
      const _exhaustive: never = state;
      return _exhaustive;
    }
  }
}

interface JobRowProps {
  job: Job;
  selected: boolean;
  onSelect: (id: string) => void;
  onStop: (id: string) => void;
}

function JobRow({ job, selected, onSelect, onStop }: JobRowProps): JSX.Element {
  const cmd = job.cmd.join(' ');
  const isRunning = job.status === 'running';

  return (
    <div className={`job-row${selected ? ' job-row-active' : ''}`}>
      <div className="job-row-main">
        <span className={`tag ${job.status}`}>{job.status}</span>
        <span className="job-name">{job.name}</span>
        <span className="job-meta muted mono">
          <span title={new Date(job.started * 1000).toLocaleString()}>
            {formatRelativeTime(job.started)}
          </span>
          <span className="job-sep">·</span>
          <span>pid {job.pid}</span>
          {job.returncode !== null && (
            <>
              <span className="job-sep">·</span>
              <span className={job.returncode === 0 ? '' : 'err'}>
                exit {job.returncode}
              </span>
            </>
          )}
        </span>
      </div>

      <div className="job-cmd muted mono" title={cmd}>
        {cmd}
      </div>

      <div className="job-row-actions">
        <button
          type="button"
          className="btn"
          onClick={() => onSelect(job.id)}
          aria-pressed={selected}
        >
          {selected ? '▾ following' : '▸ follow'}
        </button>
        {isRunning && (
          <button type="button" className="btn btn-danger" onClick={() => onStop(job.id)}>
            stop
          </button>
        )}
      </div>
    </div>
  );
}

export default function JobsPanel({ jobs }: JobsPanelProps): JSX.Element {
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [log, setLog] = useState<string>('');
  const [streamState, setStreamState] = useState<StreamState>('streaming');
  const logRef = useRef<HTMLDivElement | null>(null);
  // Tracks whether the stream has finished (done frame) so a post-done
  // onerror does not flip us into a "reconnecting" state.
  const doneRef = useRef<boolean>(false);

  // Newest first — jobs are ordered oldest→newest by start time upstream.
  const orderedJobs = useMemo(
    () => [...jobs].sort((a, b) => b.started - a.started),
    [jobs],
  );

  const selectedJob = useMemo(
    () => orderedJobs.find((j) => j.id === selectedId) ?? null,
    [orderedJobs, selectedId],
  );

  const onStop = useCallback(async (id: string) => {
    try {
      // The event stream reflects the new status within ~1s — no manual refresh.
      await stopJob(id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const onSelect = useCallback((id: string) => {
    setSelectedId((prev) => (prev === id ? null : id));
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

      {orderedJobs.length === 0 ? (
        <div className="muted">no jobs yet</div>
      ) : (
        <div className="job-list">
          {orderedJobs.map((job) => (
            <JobRow
              key={job.id}
              job={job}
              selected={job.id === selectedId}
              onSelect={onSelect}
              onStop={(id) => void onStop(id)}
            />
          ))}
        </div>
      )}

      {selectedJob !== null && (
        <div className="job-log">
          <div className="job-log-head">
            <span className="card-title job-log-title">{selectedJob.name}</span>
            <span className={`tag ${streamState === 'error' ? 'failed' : streamState === 'ended' ? 'done' : 'running'}`}>
              {streamLabel(streamState)}
            </span>
          </div>
          <div className="pre scroll" ref={logRef}>
            {log || <span className="muted">no log output</span>}
          </div>
        </div>
      )}
    </div>
  );
}
