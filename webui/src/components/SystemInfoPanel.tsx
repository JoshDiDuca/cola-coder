import { useCallback, useEffect, useState } from 'react';
import type { SystemInfo } from '../types';
import { isApiError } from '../types';
import { getSystemInfo } from '../api';
import { formatBytes, formatInteger, formatPercentValue } from '../format';

export default function SystemInfoPanel() {
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    setLoading(true);
    try {
      const resp = await getSystemInfo();
      if (isApiError(resp)) {
        setError(resp.error);
        setInfo(null);
      } else {
        setInfo(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const diskTotal = info?.disk.total_bytes ?? null;
  const diskUsed = info?.disk.used_bytes ?? null;
  const diskFill =
    diskTotal != null && diskTotal > 0 && diskUsed != null
      ? Math.max(0, Math.min(100, (diskUsed / diskTotal) * 100))
      : 0;

  return (
    <div className="card">
      <div className="card-title">System Info</div>

      <div className="row" style={{ borderBottom: 'none' }}>
        <button className="btn" onClick={() => void load()} disabled={loading}>
          Refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}
      {loading && <div className="muted">loading…</div>}

      {info && (
        <>
          <div className="row">
            <span className="k">python</span>
            <span className="v mono">{info.python_version}</span>
          </div>
          <div className="row">
            <span className="k">platform</span>
            <span className="v mono">{info.platform}</span>
          </div>

          <div className="card-title">packages</div>
          {Object.keys(info.packages).length === 0 ? (
            <div className="muted">none</div>
          ) : (
            Object.entries(info.packages).map(([name, ver]) => (
              <div className="row" key={name}>
                <span className="k mono">{name}</span>
                <span className="v mono">{ver ?? '—'}</span>
              </div>
            ))
          )}

          <div className="card-title">gpus</div>
          {info.gpus.length === 0 ? (
            <div className="muted">no GPUs</div>
          ) : (
            info.gpus.map((g, i) => (
              <div className="row" key={`${g.name}-${i}`}>
                <span className="k">{g.name}</span>
                <span className="v mono">
                  {formatInteger(g.mem_used_mb)} / {formatInteger(g.mem_total_mb)} MB
                  {g.util_pct == null ? '' : ` · ${formatPercentValue(g.util_pct, 0)}`}
                </span>
              </div>
            ))
          )}

          <div className="card-title">disk</div>
          <div className="muted mono">{info.disk.path}</div>
          <div
            className="bar"
            role="progressbar"
            aria-valuenow={Math.round(diskFill)}
            aria-valuemin={0}
            aria-valuemax={100}
          >
            <div className="fill" style={{ width: `${diskFill}%` }} />
          </div>
          <div className="row" style={{ borderBottom: 'none' }}>
            <span className="k">free / total</span>
            <span className="v mono">
              {formatBytes(info.disk.free_bytes)} / {formatBytes(diskTotal)}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
