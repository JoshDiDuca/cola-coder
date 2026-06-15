import { useCallback, useEffect, useState } from 'react';
import type { BenchmarkResults, BenchmarkRun } from '../types';
import { isApiError } from '../types';
import { getBenchmarkResults } from '../api';
import { formatInteger, formatFloat } from '../format';

type BenchmarkKind = 'throughput' | 'latency' | 'nano' | 'unknown';

// Reuse the existing tag colour classes for the kind badge.
function kindBadgeClass(kind: BenchmarkKind): string {
  switch (kind) {
    case 'throughput':
      return 'tag done';
    case 'latency':
      return 'tag running';
    case 'nano':
      return 'tag';
    case 'unknown':
      return 'tag failed';
    default: {
      const _exhaustive: never = kind;
      return _exhaustive;
    }
  }
}

// The backend emits one of four kind labels; narrow the wire string to the union
// so the badge switch stays exhaustive (unrecognized values fall back to unknown).
function asKind(kind: string): BenchmarkKind {
  switch (kind) {
    case 'throughput':
    case 'latency':
    case 'nano':
    case 'unknown':
      return kind;
    default:
      return 'unknown';
  }
}

function RunRow({ run }: { run: BenchmarkRun }) {
  const kind = asKind(run.kind);
  return (
    <tr>
      <td className="mono">{run.name}</td>
      <td>
        <span className={kindBadgeClass(kind)}>{kind}</span>
      </td>
      <td className="right mono">
        {run.tokens_per_s === null ? '—' : `${formatFloat(run.tokens_per_s, 1)} tok/s`}
      </td>
      <td className="right mono">
        {run.latency_ms === null ? '—' : `${formatFloat(run.latency_ms, 1)} ms`}
      </td>
      <td className="mono muted">{run.checkpoint ?? '—'}</td>
    </tr>
  );
}

export default function BenchmarkResultsPanel() {
  const [view, setView] = useState<BenchmarkResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (): Promise<void> => {
    setError(null);
    setLoading(true);
    try {
      const resp = await getBenchmarkResults();
      if (isApiError(resp)) {
        setError(resp.error);
        setView(null);
      } else {
        setView(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setView(null);
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
        const resp = await getBenchmarkResults();
        if (!active) return;
        if (isApiError(resp)) {
          setError(resp.error);
          setView(null);
        } else {
          setView(resp);
        }
      } catch (e) {
        if (!active) return;
        setError(e instanceof Error ? e.message : String(e));
        setView(null);
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const runs = view?.runs ?? [];

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Benchmark Results</div>
        <button className="btn" onClick={() => void load()} disabled={loading}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}
      {loading && !view && <div className="muted">loading…</div>}

      {view && runs.length === 0 && !error && (
        <div className="muted">no saved benchmark reports found</div>
      )}

      {runs.length > 0 && (
        <>
          <div className="row">
            <span className="muted mono">{formatInteger(view?.count ?? runs.length)} reports</span>
          </div>
          <table className="tbl">
            <thead>
              <tr>
                <th>report</th>
                <th>kind</th>
                <th className="right">throughput</th>
                <th className="right">first token</th>
                <th>checkpoint</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <RunRow key={run.path} run={run} />
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
