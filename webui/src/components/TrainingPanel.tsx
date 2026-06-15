import type { TrainingStatus } from '../types';
import { formatInteger, formatFloat, formatPercent } from '../format';

type HeroState = 'training' | 'idle' | 'offline';

interface StatTile {
  label: string;
  value: string;
}

function heroState(training: TrainingStatus | null): HeroState {
  if (training === null) return 'offline';
  return training.alive ? 'training' : 'idle';
}

function statusLabel(state: HeroState): string {
  switch (state) {
    case 'training':
      return 'Training';
    case 'idle':
      return 'Idle';
    case 'offline':
      return 'Offline';
    default: {
      const _exhaustive: never = state;
      return _exhaustive;
    }
  }
}

export default function TrainingPanel({ training }: { training: TrainingStatus | null }) {
  const state = heroState(training);
  const live = state === 'training';

  const step = training?.step ?? null;
  const totalSteps = training?.total_steps ?? null;
  const progressFraction = training?.progress_pct ?? null;
  const fillPct = Math.max(0, Math.min(100, progressFraction ?? 0));
  const lastLog = training?.last_log_line ?? null;

  const tiles: StatTile[] = [
    { label: 'loss', value: formatFloat(training?.loss ?? null, 4) },
    { label: 'perplexity', value: formatFloat(training?.ppl ?? null, 2) },
    { label: 'tok / s', value: formatInteger(training?.tok_per_s ?? null) },
    { label: 's / it', value: formatFloat(training?.s_per_it ?? null, 1) },
  ];

  return (
    <div className={`card card-wide hero${live ? '' : ' hero-dim'}`}>
      <div className="hero-head">
        <div className="card-title hero-status">
          <span
            className={live ? 'dot live' : 'dot dead'}
            role="img"
            aria-label={statusLabel(state)}
            title={statusLabel(state)}
          />
          {statusLabel(state)}
        </div>
        <div className="hero-progress-pct mono">{formatPercent(progressFraction)}</div>
      </div>

      <div className="hero-headline">
        <div className="hero-step-block">
          <div className="hero-step stat-big mono">{formatInteger(step)}</div>
          <div className="hero-step-total muted mono">/ {formatInteger(totalSteps)} steps</div>
        </div>
        {!live && <div className="hero-waiting muted">waiting for training…</div>}
      </div>

      <div
        className="bar hero-bar"
        role="progressbar"
        aria-valuenow={Math.round(fillPct)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div className="fill" style={{ width: `${fillPct}%` }} />
      </div>

      <div className="stat-tiles">
        {tiles.map((tile) => (
          <div className="stat-tile" key={tile.label}>
            <div className="stat-tile-label">{tile.label}</div>
            <div className="stat-tile-value mono">{tile.value}</div>
          </div>
        ))}
      </div>

      <div className="hero-log mono muted" title={lastLog ?? undefined}>
        {lastLog ?? 'no log output yet'}
      </div>
    </div>
  );
}
