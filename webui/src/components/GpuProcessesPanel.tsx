import { useCallback, useEffect, useRef, useState } from 'react';
import type { GpuProcess, GpuProcesses } from '../types';
import { getGpuProcesses } from '../api';
import { formatInteger } from '../format';

const REFRESH_MS = 5000;
const RESTRICTED_NAME = '[Insufficient Permissions]';

function ProcessRow({ proc, index }: { proc: GpuProcess; index: number }): JSX.Element {
  const restricted = proc.name === RESTRICTED_NAME;
  return (
    <div className="gpuproc-row" key={`${proc.pid}-${index}`}>
      <span className="gpuproc-pid mono">{proc.pid}</span>
      {restricted ? (
        <span className="gpuproc-name muted" title={proc.name}>
          restricted (elevated process)
        </span>
      ) : (
        <span className="gpuproc-name mono">{proc.name}</span>
      )}
      <span className="gpuproc-mem mono">
        {proc.used_memory_mb === null ? '—' : `${formatInteger(proc.used_memory_mb)} MB`}
      </span>
    </div>
  );
}

export default function GpuProcessesPanel(): JSX.Element {
  const [view, setView] = useState<GpuProcesses | null>(null);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const load = useCallback(async (): Promise<void> => {
    try {
      const resp = await getGpuProcesses();
      setView(resp);
      setError(null);
    } catch (e) {
      // getGpuProcesses does not return ApiError — only a network/fetch failure
      // reaches here. Keep prior data so a transient blip doesn't blank the panel.
      setError(e instanceof Error ? e.message : 'couldn’t query GPU');
    }
  }, []);

  useEffect(() => {
    void load();
    intervalRef.current = setInterval(() => {
      void load();
    }, REFRESH_MS);
    return () => {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [load]);

  const loadingFirst = view === null && error === null;

  return (
    <div className="card gpuproc-card">
      <div className="row">
        <div className="card-title">GPU processes</div>
        <span className="gpuproc-live muted mono" aria-label="auto-refreshing">
          <span className="dot live" aria-hidden="true" /> live
        </span>
      </div>

      {error !== null && view === null && <div className="muted">couldn’t query GPU</div>}
      {loadingFirst && <div className="muted">loading…</div>}

      {view !== null && (
        <>
          {!view.available ? (
            <div className="muted">nvidia-smi not available.</div>
          ) : view.count === 0 ? (
            <div className="muted">No processes are using the GPU.</div>
          ) : (
            <>
              <div className="gpuproc-row gpuproc-head">
                <span className="gpuproc-pid">PID</span>
                <span className="gpuproc-name">process</span>
                <span className="gpuproc-mem">memory</span>
              </div>
              <div className="gpuproc-list">
                {view.processes.map((proc, i) => (
                  <ProcessRow proc={proc} index={i} key={`${proc.pid}-${i}`} />
                ))}
              </div>
              {view.restricted && (
                <div className="gpuproc-note muted">
                  Some process names are hidden by OS integrity — the elevated training run
                  shows as restricted.
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
