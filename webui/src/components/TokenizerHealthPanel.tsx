import { useCallback, useEffect, useState } from 'react';
import type { TokenizerHealthReport, TokenizerHealthItem } from '../types';
import { isApiError } from '../types';
import { getTokenizerHealth } from '../api';
import { formatInteger } from '../format';

// PASS/FAIL rows reuse the existing tag colour classes.
function checkBadgeClass(ok: boolean): string {
  return ok ? 'tag done' : 'tag failed';
}

function CheckRow({ check }: { check: TokenizerHealthItem }) {
  return (
    <div className="row">
      <span className="k">
        <span className={checkBadgeClass(check.ok)}>{check.ok ? 'PASS' : 'FAIL'}</span> {check.name}
      </span>
      <span className="v muted mono">{check.detail}</span>
    </div>
  );
}

export default function TokenizerHealthPanel() {
  const [report, setReport] = useState<TokenizerHealthReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (): Promise<void> => {
    setError(null);
    setLoading(true);
    try {
      const resp = await getTokenizerHealth();
      if (isApiError(resp)) {
        setError(resp.error);
        setReport(null);
      } else {
        setReport(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setReport(null);
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
        const resp = await getTokenizerHealth();
        if (!active) return;
        if (isApiError(resp)) {
          setError(resp.error);
          setReport(null);
        } else {
          setReport(resp);
        }
      } catch (e) {
        if (!active) return;
        setError(e instanceof Error ? e.message : String(e));
        setReport(null);
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="card">
      <div className="row">
        <div className="card-title">Tokenizer Health</div>
        <button className="btn" onClick={() => void load()} disabled={loading}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}
      {loading && !report && <div className="muted">loading…</div>}

      {report && (
        <>
          <div className="row">
            <span className={report.ok ? 'tag done' : 'tag failed'}>
              {report.ok ? 'healthy' : 'unhealthy'}
            </span>
            <span className="v muted">
              {report.passed}/{report.checks.length} passed
            </span>
          </div>

          <div className="row">
            <span className="k">path</span>
            <span className="v mono">{report.path}</span>
          </div>
          <div className="row">
            <span className="k">vocab size</span>
            <span className="v mono">{formatInteger(report.vocab_size)}</span>
          </div>

          <div className="tbl">
            {report.checks.length === 0 ? (
              <div className="muted">no checks</div>
            ) : (
              report.checks.map((check) => <CheckRow key={check.name} check={check} />)
            )}
          </div>
        </>
      )}
    </div>
  );
}
