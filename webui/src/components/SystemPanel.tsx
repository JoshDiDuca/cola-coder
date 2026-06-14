import type { SystemStatus } from '../types';

function fmtInt(n: number | null | undefined): string {
  return n == null ? '—' : Math.round(n).toLocaleString();
}

export default function SystemPanel({ system }: { system: SystemStatus | null }) {
  const util = system?.gpu_util_pct;
  const name = system?.gpu_name;
  const memUsed = system?.gpu_mem_used_mb;
  const memTotal = system?.gpu_mem_total_mb;
  const power = system?.gpu_power_w;

  const utilText = util == null ? '—' : `${Math.round(util)}%`;
  const fillPct = Math.max(0, Math.min(100, util ?? 0));

  return (
    <div className="card">
      <div className="card-title">GPU / System</div>

      <div className="stat-big">{utilText}</div>
      <div className="stat-sub">{name ?? 'no GPU'}</div>

      <div className="bar" role="progressbar" aria-valuenow={fillPct} aria-valuemin={0} aria-valuemax={100}>
        <div className="fill" style={{ width: `${fillPct}%` }} />
      </div>

      <div className="row">
        <span className="k">VRAM</span>
        <span className="v mono">
          {fmtInt(memUsed)} / {fmtInt(memTotal)} MB
        </span>
      </div>
      <div className="row">
        <span className="k">power</span>
        <span className="v mono">{power == null ? '—' : `${Math.round(power)} W`}</span>
      </div>
    </div>
  );
}
