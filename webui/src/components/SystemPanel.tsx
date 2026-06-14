import type { SystemStatus } from '../types';
import { formatInteger, formatFloat, formatPercentValue } from '../format';

export default function SystemPanel({ system }: { system: SystemStatus | null }) {
  const util = system?.gpu_util_pct ?? null;
  const power = system?.gpu_power_w ?? null;
  const fillPct = Math.max(0, Math.min(100, util ?? 0));

  return (
    <div className="card">
      <div className="card-title">GPU / System</div>

      <div className="stat-big">{formatPercentValue(util, 0)}</div>
      <div className="stat-sub">{system?.gpu_name ?? 'no GPU'}</div>

      <div className="bar" role="progressbar" aria-valuenow={fillPct} aria-valuemin={0} aria-valuemax={100}>
        <div className="fill" style={{ width: `${fillPct}%` }} />
      </div>

      <div className="row">
        <span className="k">VRAM</span>
        <span className="v mono">
          {formatInteger(system?.gpu_mem_used_mb ?? null)} / {formatInteger(system?.gpu_mem_total_mb ?? null)} MB
        </span>
      </div>
      <div className="row">
        <span className="k">power</span>
        <span className="v mono">{power == null ? '—' : `${formatFloat(power, 0)} W`}</span>
      </div>
    </div>
  );
}
