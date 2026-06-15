import { useCallback, useEffect, useState } from 'react';
import type { EnvCheckReport, EnvCheckItem } from '../types';
import { isApiError } from '../types';
import { getEnvCheck } from '../api';

// PASS/FAIL rows reuse the existing tag colour classes.
function checkBadgeClass(ok: boolean): string {
  return ok ? 'tag done' : 'tag failed';
}

function CheckRow({ check }: { check: EnvCheckItem }) {
  return (
    <div className="row">
      <span className="k">
        <span className={checkBadgeClass(check.ok)}>{check.ok ? 'PASS' : 'FAIL'}</span> {check.name}
      </span>
      <span className="v muted mono">
        {check.value}
        {check.detail ? ` — ${check.detail}` : ''}
      </span>
    </div>
  );
}

export default function EnvCheckPanel() {
  const [report, setReport] = useState<EnvCheckReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (): Promise<void> => {
    setError(null);
    setLoading(true);
    try {
      const resp = await getEnvCheck();
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
        const resp = await getEnvCheck();
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
        <div className="card-title">Environment Check</div>
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
              {report.ok ? 'ready' : 'issues'}
            </span>
            <span className="v muted">
              {report.passed}/{report.checks.length} passed
              {report.failed > 0 ? ` (${report.failed} failed)` : ''}
            </span>
          </div>

          <div className="row">
            <span className="k">python</span>
            <span className="v mono">{report.python_version}</span>
          </div>
          <div className="row">
            <span className="k">torch</span>
            <span className="v mono">{report.torch_version ?? 'not installed'}</span>
          </div>
          <div className="row">
            <span className="k">cuda</span>
            <span className="v mono">{report.cuda_available ? 'available' : 'unavailable'}</span>
          </div>
          <div className="row">
            <span className="k">gpu</span>
            <span className="v mono">
              {report.gpu_name ?? '—'}
              {report.vram_gb !== null ? ` (${report.vram_gb} GB)` : ''}
            </span>
          </div>
          <div className="row">
            <span className="k">HF_TOKEN</span>
            <span className="v mono">{report.hf_token_set ? 'set' : 'not set'}</span>
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
