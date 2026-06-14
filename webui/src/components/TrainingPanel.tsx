import type { TrainingStatus } from '../types';
import { formatInteger, formatFloat } from '../format';

export default function TrainingPanel({ training }: { training: TrainingStatus | null }) {
  const alive = training?.alive ?? false;
  const fillPct = Math.max(0, Math.min(100, training?.progress_pct ?? 0));
  const lossText = formatFloat(training?.loss ?? null, 4);
  const pplText = formatFloat(training?.ppl ?? null, 2);
  const sPerIt = training?.s_per_it ?? null;

  return (
    <div className="card">
      <div className="card-title">
        Training
        <span
          className={alive ? 'dot live' : 'dot dead'}
          role="img"
          aria-label={alive ? 'live' : 'idle'}
          title={alive ? 'live' : 'idle'}
        />
        <span className="muted">{training == null ? 'loading…' : alive ? 'live' : 'idle'}</span>
      </div>

      <div className="stat-big">{lossText}</div>
      <div className="stat-sub">perplexity {pplText}</div>

      <div className="bar" role="progressbar" aria-valuenow={fillPct} aria-valuemin={0} aria-valuemax={100}>
        <div className="fill" style={{ width: `${fillPct}%` }} />
      </div>

      <div className="row">
        <span className="k">step</span>
        <span className="v mono">
          {formatInteger(training?.step ?? null)} / {formatInteger(training?.total_steps ?? null)}
        </span>
      </div>
      <div className="row">
        <span className="k">ppl</span>
        <span className="v mono">{pplText}</span>
      </div>
      <div className="row">
        <span className="k">tok/s</span>
        <span className="v mono">{formatInteger(training?.tok_per_s ?? null)}</span>
      </div>
      <div className="row">
        <span className="k">s/it</span>
        <span className="v mono">{sPerIt == null ? '—' : `${formatFloat(sPerIt, 1)}s`}</span>
      </div>
    </div>
  );
}
