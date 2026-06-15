import type { SystemStatus } from '../types';
import { formatBytes, formatFloat, formatPercentValue } from '../format';

type MeterLevel = 'good' | 'warn' | 'bad';

interface Meter {
  label: string;
  /** Fill fraction 0..1 used for the bar width. */
  fraction: number | null;
  /** Pre-formatted value shown on the right. */
  display: string;
  level: MeterLevel;
}

/** Map a 0..1 load fraction to a colour tier (green / amber / red). */
function loadLevel(fraction: number | null): MeterLevel {
  if (fraction === null) return 'good';
  if (fraction >= 0.9) return 'bad';
  if (fraction >= 0.7) return 'warn';
  return 'good';
}

function clampFraction(fraction: number | null): number {
  if (fraction === null) return 0;
  return Math.max(0, Math.min(1, fraction));
}

function MeterRow({ meter }: { meter: Meter }) {
  const widthPct = clampFraction(meter.fraction) * 100;
  return (
    <div className="meter">
      <div className="meter-head">
        <span className="meter-label">{meter.label}</span>
        <span className="meter-value mono">{meter.display}</span>
      </div>
      <div
        className="bar meter-bar"
        role="progressbar"
        aria-label={meter.label}
        aria-valuenow={Math.round(widthPct)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div className={`fill meter-fill meter-fill-${meter.level}`} style={{ width: `${widthPct}%` }} />
      </div>
    </div>
  );
}

export default function SystemPanel({ system }: { system: SystemStatus | null }) {
  const gpuName = system?.gpu_name ?? null;
  const util = system?.gpu_util_pct ?? null;
  const memUsedMb = system?.gpu_mem_used_mb ?? null;
  const memTotalMb = system?.gpu_mem_total_mb ?? null;
  const power = system?.gpu_power_w ?? null;

  const hasGpu =
    gpuName !== null ||
    util !== null ||
    memUsedMb !== null ||
    memTotalMb !== null ||
    power !== null;

  const utilFraction = util === null ? null : util / 100;
  const memFraction =
    memUsedMb === null || memTotalMb === null || memTotalMb === 0
      ? null
      : memUsedMb / memTotalMb;
  // Power has no fixed ceiling; show the bar relative to a typical 320W board cap.
  const POWER_REFERENCE_W = 320;
  const powerFraction = power === null ? null : power / POWER_REFERENCE_W;

  const memUsedBytes = memUsedMb === null ? null : memUsedMb * 1024 * 1024;
  const memTotalBytes = memTotalMb === null ? null : memTotalMb * 1024 * 1024;

  const meters: Meter[] = [
    {
      label: 'utilization',
      fraction: utilFraction,
      display: formatPercentValue(util, 0),
      level: loadLevel(utilFraction),
    },
    {
      label: 'memory',
      fraction: memFraction,
      display: `${formatBytes(memUsedBytes)} / ${formatBytes(memTotalBytes)}`,
      level: loadLevel(memFraction),
    },
    {
      label: 'power',
      fraction: powerFraction,
      display: power === null ? '—' : `${formatFloat(power, 0)} W`,
      level: loadLevel(powerFraction),
    },
  ];

  return (
    <div className="card gpu-card">
      <div className="card-title">GPU / System</div>

      {!hasGpu ? (
        <div className="gpu-empty muted">no GPU data</div>
      ) : (
        <>
          <div className="gpu-name mono">{gpuName ?? 'unknown GPU'}</div>
          <div className="gpu-meters">
            {meters.map((meter) => (
              <MeterRow key={meter.label} meter={meter} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
