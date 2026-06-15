import { useCallback, useEffect, useState } from 'react';
import type { SafetyEvalResults, SafetyEvalRun, SafetyProbe } from '../types';
import { isApiError } from '../types';
import { getSafetyEvalResults } from '../api';
import { formatInteger } from '../format';

// PASS/FAIL probe badges reuse the existing tag colour classes.
function probeBadgeClass(passed: boolean): string {
  return passed ? 'tag done' : 'tag failed';
}

function runBadgeClass(failed: number): string {
  return failed === 0 ? 'tag done' : 'tag failed';
}

function ProbeRow({ probe }: { probe: SafetyProbe }) {
  return (
    <div className="row">
      <span className="k">
        <span className={probeBadgeClass(probe.passed)}>{probe.passed ? 'PASS' : 'FAIL'}</span>{' '}
        <span className="tag">{probe.suite}</span> {probe.name}
      </span>
      <span className="v muted mono">{probe.detail ?? ''}</span>
    </div>
  );
}

function RunCard({ run }: { run: SafetyEvalRun }) {
  return (
    <div className="tbl">
      <div className="row">
        <span className="k">
          <span className={runBadgeClass(run.failed)}>{run.suite}</span>{' '}
          <span className="mono">{run.name}</span>
        </span>
        <span className="v muted">
          {formatInteger(run.passed)}/{formatInteger(run.total)} passed
        </span>
      </div>
      <div className="row">
        <span className="k">checkpoint</span>
        <span className="v mono">{run.checkpoint ?? '—'}</span>
      </div>
      <div className="row">
        <span className="k">path</span>
        <span className="v mono">{run.path}</span>
      </div>
      {run.probes.length === 0 ? (
        <div className="muted">no per-probe detail</div>
      ) : (
        run.probes.map((probe, i) => <ProbeRow key={`${probe.suite}:${probe.name}:${i}`} probe={probe} />)
      )}
    </div>
  );
}

export default function SafetyEvalPanel() {
  const [results, setResults] = useState<SafetyEvalResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (): Promise<void> => {
    setError(null);
    setLoading(true);
    try {
      const resp = await getSafetyEvalResults();
      if (isApiError(resp)) {
        setError(resp.error);
        setResults(null);
      } else {
        setResults(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setResults(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let active = true;
    void (async () => {
      setError(null);
      setLoading(true);
      try {
        const resp = await getSafetyEvalResults();
        if (!active) return;
        if (isApiError(resp)) {
          setError(resp.error);
          setResults(null);
        } else {
          setResults(resp);
        }
      } catch (e) {
        if (!active) return;
        setError(e instanceof Error ? e.message : String(e));
        setResults(null);
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Safety Eval Results</div>
        <button className="btn" onClick={() => void load()} disabled={loading}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}
      {loading && !results && <div className="muted">loading…</div>}

      {results && results.runs.length === 0 && !error && (
        <div className="muted">no safety-eval results found</div>
      )}

      {results &&
        results.runs.length > 0 &&
        results.runs.map((run) => <RunCard key={run.path} run={run} />)}
    </div>
  );
}
