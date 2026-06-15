import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Job, JobLogChunk } from '../../types';
import { jobLogStreamUrl, stopJob } from '../../api';
import { formatRelativeTime } from '../../format';
import ActionsPanel from '../ActionsPanel';
import MasterDetail, { type MasterItem } from '../MasterDetail';

// ── "Run & Jobs" screen ──────────────────────────────────────────────────────
// Top: the action gallery for LAUNCHING work. Below: a master-detail jobs view —
// the job list on the left, the selected job's live streaming log on the right.
// The streaming logic mirrors JobsPanel exactly (jobLogStreamUrl + EventSource +
// JobLogChunk parsing + stopJob), reused here in the detail pane.

const MAX_LOG_LINES = 5000;

type StreamState = 'streaming' | 'ended' | 'error';

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

// Map the stream state to a status-tag tone (reuses .tag.running/.done/.failed).
function streamTagClass(state: StreamState): string {
  switch (state) {
    case 'streaming':
      return 'tag running';
    case 'ended':
      return 'tag done';
    case 'error':
      return 'tag failed';
    default: {
      const _exhaustive: never = state;
      return _exhaustive;
    }
  }
}

// ── Job detail pane: header + live streaming log + stop button ────────────────

interface JobDetailProps {
  job: Job;
  onStop: (id: string) => void;
}

function JobDetail({ job, onStop }: JobDetailProps): JSX.Element {
  const [log, setLog] = useState<string>('');
  const [streamState, setStreamState] = useState<StreamState>('streaming');
  const logRef = useRef<HTMLDivElement | null>(null);
  // Tracks whether the stream has finished (done frame) so a post-done
  // onerror does not flip us into a "reconnecting" state.
  const doneRef = useRef<boolean>(false);

  const cmd = job.cmd.join(' ');
  const isRunning = job.status === 'running';

  // Own the EventSource lifecycle: create when the job changes, close on cleanup
  // (job switch or unmount). Switching jobs runs cleanup first, so the previous
  // EventSource is always closed before a new one opens — no leaks.
  useEffect(() => {
    doneRef.current = false;
    let firstFrame = true;
    setLog('');
    setStreamState('streaming');

    const source = new EventSource(jobLogStreamUrl(job.id));

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
  }, [job.id]);

  // Auto-scroll the log view to the bottom whenever its content changes.
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [log]);

  return (
    <div className="card card-wide">
      <div className="job-detail-head">
        <div className="job-detail-titles">
          <h2 className="md-detail-title">{job.name}</h2>
          <div className="muted mono job-detail-sub">
            <span className={`tag ${job.status}`}>{job.status}</span>
            <span>pid {job.pid}</span>
            {job.returncode !== null && (
              <span className={job.returncode === 0 ? '' : 'err'}>exit {job.returncode}</span>
            )}
            <span title={new Date(job.started * 1000).toLocaleString()}>
              {formatRelativeTime(job.started)}
            </span>
          </div>
          <div className="muted mono job-detail-cmd" title={cmd}>
            {cmd}
          </div>
        </div>
        <div className="job-detail-actions">
          <span className={streamTagClass(streamState)}>{streamLabel(streamState)}</span>
          {isRunning && (
            <button type="button" className="btn btn-danger" onClick={() => onStop(job.id)}>
              stop
            </button>
          )}
        </div>
      </div>

      <div className="pre scroll" ref={logRef}>
        {log || <span className="muted">no log output</span>}
      </div>
    </div>
  );
}

// ── Screen ────────────────────────────────────────────────────────────────────

interface RunScreenProps {
  jobs: Job[];
  trainingAlive: boolean;
}

export default function RunScreen({ jobs, trainingAlive }: RunScreenProps): JSX.Element {
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Newest first — jobs are ordered oldest→newest by start time upstream.
  const orderedJobs = useMemo(
    () => [...jobs].sort((a, b) => b.started - a.started),
    [jobs],
  );

  // Default-select the newest running job, else the newest job. Keep the
  // selection valid (and follow newer arrivals) as the event stream updates.
  useEffect(() => {
    setSelectedId((prev) => {
      if (prev !== null && orderedJobs.some((j) => j.id === prev)) return prev;
      const newestRunning = orderedJobs.find((j) => j.status === 'running');
      return (newestRunning ?? orderedJobs[0])?.id ?? null;
    });
  }, [orderedJobs]);

  const selectedJob = useMemo(
    () => orderedJobs.find((j) => j.id === selectedId) ?? null,
    [orderedJobs, selectedId],
  );

  const onStop = useCallback(async (id: string): Promise<void> => {
    try {
      // The event stream reflects the new status within ~1s — no manual refresh.
      await stopJob(id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const items: MasterItem[] = useMemo(
    () =>
      orderedJobs.map((job) => ({
        id: job.id,
        title: job.name,
        subtitle: undefined,
        meta: formatRelativeTime(job.started),
        badge: <span className={`tag ${job.status}`}>{job.status}</span>,
      })),
    [orderedJobs],
  );

  return (
    <div className="run-screen">
      <section className="run-launch">
        <h2 className="run-section-title">Launch an action</h2>
        <ActionsPanel
          onRan={() => {
            /* jobs arrive via the event stream — no manual refresh needed */
          }}
          trainingAlive={trainingAlive}
        />
      </section>

      <section className="run-jobs">
        <h2 className="run-section-title">Jobs</h2>
        {error && <div className="err">{error}</div>}
        <MasterDetail
          items={items}
          selectedId={selectedId}
          onSelect={setSelectedId}
          listLabel={`${items.length} job${items.length === 1 ? '' : 's'}`}
          emptyList="No jobs yet — launch an action above"
          emptyDetail="Select a job to follow its live log"
          detail={
            selectedJob ? <JobDetail job={selectedJob} onStop={(id) => void onStop(id)} /> : null
          }
        />
      </section>
    </div>
  );
}
