import { useCallback, useEffect, useState } from 'react';
import type { SystemInfo, GpuInfo } from '../types';
import { isApiError } from '../types';
import { getSystemInfo } from '../api';
import { formatBytes, formatInteger, formatPercentValue } from '../format';

interface InfoTile {
  label: string;
  value: string;
}

function mbToBytes(mb: number | null): number | null {
  return mb === null ? null : mb * 1024 * 1024;
}

function gpuMemDisplay(gpu: GpuInfo): string {
  const used = mbToBytes(gpu.mem_used_mb);
  const total = mbToBytes(gpu.mem_total_mb);
  return `${formatBytes(used)} / ${formatBytes(total)}`;
}

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
  const diskFree = info?.disk.free_bytes ?? null;
  const diskFill =
    diskTotal != null && diskTotal > 0 && diskUsed != null
      ? Math.max(0, Math.min(100, (diskUsed / diskTotal) * 100))
      : 0;

  const packageEntries = info ? Object.entries(info.packages) : [];

  const envTiles: InfoTile[] = info
    ? [
        { label: 'python', value: info.python_version },
        { label: 'platform', value: info.platform },
        { label: 'gpus', value: formatInteger(info.gpus.length) },
        { label: 'disk free', value: formatBytes(diskFree) },
      ]
    : [];

  return (
    <div className="card">
      <div className="row">
        <div className="card-title">System Info</div>
        <button className="btn" onClick={() => void load()} disabled={loading}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}
      {loading && info === null && <div className="muted">loading…</div>}

      {info && (
        <>
          <div className="stat-tiles sysinfo-tiles">
            {envTiles.map((tile) => (
              <div className="stat-tile" key={tile.label}>
                <div className="stat-tile-label">{tile.label}</div>
                <div className="stat-tile-value mono sysinfo-tile-value">{tile.value}</div>
              </div>
            ))}
          </div>

          <div className="card-title">gpus</div>
          {info.gpus.length === 0 ? (
            <div className="muted">no GPUs detected</div>
          ) : (
            <div className="sysinfo-grid">
              {info.gpus.map((gpu, i) => (
                <div className="sysinfo-gpu" key={`${gpu.name}-${i}`}>
                  <div className="sysinfo-gpu-name">{gpu.name}</div>
                  <div className="sysinfo-gpu-stats mono">
                    <span>{gpuMemDisplay(gpu)}</span>
                    <span className="muted">
                      {gpu.util_pct == null ? 'util —' : `util ${formatPercentValue(gpu.util_pct, 0)}`}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="card-title">disk</div>
          <div className="row" style={{ borderBottom: 'none' }}>
            <span className="k mono">{info.disk.path}</span>
            <span className="v mono">
              {formatBytes(diskFree)} free / {formatBytes(diskTotal)}
            </span>
          </div>
          <div
            className="bar"
            role="progressbar"
            aria-label="disk usage"
            aria-valuenow={Math.round(diskFill)}
            aria-valuemin={0}
            aria-valuemax={100}
          >
            <div className="fill" style={{ width: `${diskFill}%` }} />
          </div>

          <div className="card-title">packages</div>
          {packageEntries.length === 0 ? (
            <div className="muted">none reported</div>
          ) : (
            <div className="sysinfo-pkgs">
              {packageEntries.map(([name, ver]) => (
                <div className="sysinfo-pkg" key={name}>
                  <span className="sysinfo-pkg-name mono">{name}</span>
                  <span className="sysinfo-pkg-ver mono">{ver ?? '—'}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
