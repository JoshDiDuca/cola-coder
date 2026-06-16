import { useCallback, useState } from 'react';
import type { ChangeEvent } from 'react';
import type {
  DomainDetectRequest,
  DomainDetectResult,
  DomainScoreOut,
  RouteDecisionOut,
} from '../types';
import { isApiError } from '../types';
import { detectDomain } from '../api';
import { formatPercent } from '../format';

type RouteReason = RouteDecisionOut['reason'];

const KNOWN_REASONS = ['ok', 'low_confidence', 'low_margin', 'no_signal'] as const;
type KnownReason = (typeof KNOWN_REASONS)[number];

function isKnownReason(reason: RouteReason): reason is KnownReason {
  return (KNOWN_REASONS as readonly string[]).includes(reason);
}

/** Map a router reason code to friendly, human-readable text. */
function describeReason(reason: RouteReason): string {
  if (!isKnownReason(reason)) {
    return reason;
  }
  switch (reason) {
    case 'ok':
      return 'confident specialist match';
    case 'low_confidence':
      return 'top score below confidence threshold';
    case 'low_margin':
      return 'top two domains too close';
    case 'no_signal':
      return 'no domain signal';
    default: {
      const _exhaustive: never = reason;
      return _exhaustive;
    }
  }
}

/**
 * Domain Detection tool: paste a code snippet and see which framework/domain
 * the heuristic router would classify it as. Pairs with the Domain Specialists
 * registry — purely a regex heuristic, no model is loaded.
 */
export default function DomainDetectPanel(): JSX.Element {
  const [code, setCode] = useState<string>('');
  const [filename, setFilename] = useState<string>('');
  const [result, setResult] = useState<DomainDetectResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState<boolean>(false);

  const onCodeChange = useCallback((e: ChangeEvent<HTMLTextAreaElement>): void => {
    setCode(e.target.value);
  }, []);

  const onFilenameChange = useCallback((e: ChangeEvent<HTMLInputElement>): void => {
    setFilename(e.target.value);
  }, []);

  const onDetect = useCallback(async (): Promise<void> => {
    setSubmitting(true);
    setError(null);
    const req: DomainDetectRequest = { code };
    const trimmedFilename = filename.trim();
    if (trimmedFilename.length > 0) {
      req.filename = trimmedFilename;
    }
    try {
      const res = await detectDomain(req);
      if (isApiError(res)) {
        setError(res.error);
        setResult(null);
      } else {
        setResult(res);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setResult(null);
    } finally {
      setSubmitting(false);
    }
  }, [code, filename]);

  const canSubmit: boolean = !submitting && code.trim().length > 0;

  // Always render the top row even if its confidence is 0; otherwise hide zero rows.
  const visibleScores: DomainScoreOut[] = result
    ? result.scores.filter((s, i) => i === 0 || s.confidence > 0)
    : [];

  return (
    <div className="card card-wide">
      <div className="card-title">Domain detection</div>
      <div className="muted">
        Regex heuristic — no model is loaded. Shows which specialist the router would pick.
      </div>

      <textarea
        className="textarea mono"
        rows={12}
        spellCheck={false}
        placeholder="Paste a code snippet…"
        value={code}
        onChange={onCodeChange}
      />

      <input
        className="input"
        placeholder="Button.test.tsx — optional"
        value={filename}
        onChange={onFilenameChange}
      />

      <button className="btn btn-primary" disabled={!canSubmit} onClick={onDetect}>
        {submitting ? 'Detecting…' : 'Detect domain'}
      </button>

      {error !== null && <div className="err">{error}</div>}

      {result !== null && (
        <div>
          <div className="card-title">Routing decision</div>
          <div className="row">
            <span className="muted">Router dispatches to:</span>{' '}
            {result.routing.abstained ? (
              <span className="tag warn">{result.routing.domain}</span>
            ) : (
              <span className="tag">{result.routing.domain}</span>
            )}
            {result.routing.abstained && (
              <span className="muted">— fell back to the general model</span>
            )}
          </div>
          <div className="row">
            <span className="muted">margin:</span>{' '}
            <span className="mono">{formatPercent(result.routing.margin)}</span>{' '}
            <span className="muted">· {describeReason(result.routing.reason)}</span>
          </div>
          <div className="muted">
            Routing can differ from the top raw score — it abstains on weak or ambiguous signals.
          </div>

          <div className="card-title">raw scores</div>
          <div className="row">
            <span className="muted">Top domain:</span> <span className="tag">{result.top_domain}</span>
          </div>
          {visibleScores.map((score) => (
            <div className="row" key={score.domain}>
              <span className="mono">{score.domain}</span>
              <div className="bar">
                <div className="fill" style={{ width: `${Math.round(score.confidence * 100)}%` }} />
              </div>
              <span>{formatPercent(score.confidence)}</span>
              <span className="muted">
                imports: {score.import_matches} · keywords: {score.keyword_matches}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
