import { useCallback, useState } from 'react';
import type { ScorerBreakdown, SnippetScores } from '../types';
import { isApiError } from '../types';
import { scoreSnippet } from '../api';
import { formatPercent } from '../format';

// ── Tier → tag class ──────────────────────────────────────────────────────────
// `tier` is an open string from the backend (the scorer tier label), so a plain
// lookup is appropriate here — no exhaustive `never` check.
function tierClass(tier: string): string {
  const t = tier.toLowerCase();
  if (t === 'excellent' || t === 'good') return 'tag done';
  if (t === 'average') return 'tag warn';
  if (t === 'poor' || t === 'reject') return 'tag failed';
  return 'tag';
}

// ── Per-scorer row ────────────────────────────────────────────────────────────
function ScorerRow({ scorer }: { scorer: ScorerBreakdown }): JSX.Element {
  const widthPct = `${Math.round(scorer.score * 100)}%`;
  return (
    <div className="row">
      <span className="mono">{scorer.name}</span>
      <span className="bar">
        {/* The ONE sanctioned data-driven inline style: confidence bar width. */}
        <span className="fill" style={{ width: widthPct }} />
      </span>
      <span className="mono">{formatPercent(scorer.score)}</span>
      <span className={tierClass(scorer.tier)}>{scorer.tier}</span>
    </div>
  );
}

export default function ScorerPlaygroundPanel(): JSX.Element {
  const [code, setCode] = useState<string>('');
  const [result, setResult] = useState<SnippetScores | null>(null);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const onScore = useCallback(async (): Promise<void> => {
    const snippet = code.trim();
    if (snippet.length === 0) return;
    setError(null);
    setSubmitting(true);
    try {
      const resp = await scoreSnippet({ code: snippet });
      if (isApiError(resp)) {
        setError(resp.error);
        setResult(null);
      } else {
        setResult(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setResult(null);
    } finally {
      setSubmitting(false);
    }
  }, [code]);

  const disabled: boolean = submitting || code.trim().length === 0;

  return (
    <>
      <div className="card-title">Quality scorer</div>
      <div className="muted">
        Runs the pure-Python scorers (no model/Docker). Shows the per-signal quality breakdown used
        for data weighting.
      </div>

      <textarea
        className="textarea mono"
        rows={12}
        spellCheck={false}
        placeholder="paste TypeScript/Python to score…"
        value={code}
        onChange={(e) => setCode(e.target.value)}
      />

      <div className="row">
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => void onScore()}
          disabled={disabled}
        >
          {submitting ? 'Scoring…' : 'Score'}
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {result && (
        <>
          <div className="row">
            <span className="mono">mean</span>
            <span className="bar">
              <span className="fill" style={{ width: `${Math.round(result.mean_score * 100)}%` }} />
            </span>
            <span className="mono">{formatPercent(result.mean_score)}</span>
            <span className={tierClass(result.mean_tier)}>{result.mean_tier}</span>
          </div>

          {result.scorers.map((scorer) => (
            <ScorerRow key={scorer.name} scorer={scorer} />
          ))}
        </>
      )}
    </>
  );
}
