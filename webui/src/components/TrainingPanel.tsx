import type { TrainingStatus } from '../types';

function fmtInt(n: number | null | undefined): string {
  return n == null ? '—' : Math.round(n).toLocaleString();
}

export default function TrainingPanel({ training }: { training: TrainingStatus | null }) {
  const alive = training?.alive ?? false;
  const loss = training?.loss;
  const ppl = training?.ppl;
  const step = training?.step;
  const totalSteps = training?.total_steps;
  const tokPerS = training?.tok_per_s;
  const sPerIt = training?.s_per_it;
  const progress = training?.progress_pct;

  const lossText = loss == null ? '—' : loss.toFixed(4);
  const pplText = ppl == null ? '—' : ppl.toFixed(2);
  const fillPct = Math.max(0, Math.min(100, progress ?? 0));

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
          {fmtInt(step)} / {fmtInt(totalSteps)}
        </span>
      </div>
      <div className="row">
        <span className="k">ppl</span>
        <span className="v mono">{pplText}</span>
      </div>
      <div className="row">
        <span className="k">tok/s</span>
        <span className="v mono">{fmtInt(tokPerS)}</span>
      </div>
      <div className="row">
        <span className="k">s/it</span>
        <span className="v mono">{sPerIt == null ? '—' : `${sPerIt.toFixed(1)}s`}</span>
      </div>
    </div>
  );
}
