import { useCallback, useState } from 'react';
import type { MalwareScanResult, ThreatInfo } from '../types';
import { isApiError } from '../types';
import { scanForMalware } from '../api';
import { formatInteger, formatFloat } from '../format';

type Severity = 'high' | 'medium' | 'low';

function severityBadgeClass(severity: string): string {
  const known: Severity = (['high', 'medium', 'low'] as const).includes(severity as Severity)
    ? (severity as Severity)
    : 'low';
  switch (known) {
    case 'high':
      return 'tag failed';
    case 'medium':
      return 'tag running';
    case 'low':
      return 'tag';
    default: {
      const _exhaustive: never = known;
      return _exhaustive;
    }
  }
}

function ThreatRow({ threat }: { threat: ThreatInfo }) {
  return (
    <div className="row">
      <span className="k">
        <span className={severityBadgeClass(threat.severity)}>{threat.severity}</span> {threat.name}
      </span>
      <span className="v muted mono">
        {threat.file_path}
        {threat.details ? ` — ${threat.details}` : ''} [{threat.scanner}]
      </span>
    </div>
  );
}

export default function SecurityScanPanel() {
  const [path, setPath] = useState('');
  const [result, setResult] = useState<MalwareScanResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const scan = useCallback(async (): Promise<void> => {
    const target = path.trim();
    if (target.length === 0) {
      setError('enter a path to scan');
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const resp = await scanForMalware(target);
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
      setLoading(false);
    }
  }, [path]);

  return (
    <div className="card">
      <div className="row">
        <div className="card-title">Security / Malware Scan</div>
        <button className="btn" onClick={() => void scan()} disabled={loading}>
          scan
        </button>
      </div>

      <div className="row">
        <input
          className="input mono"
          placeholder="path to scan (file or directory)"
          value={path}
          onChange={(e) => setPath(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') void scan();
          }}
        />
      </div>

      {error && <div className="err">{error}</div>}
      {loading && <div className="muted">scanning…</div>}

      {result && !loading && (
        <>
          <div className="row">
            <span className={result.is_clean ? 'tag done' : 'tag failed'}>
              {result.is_clean ? 'clean' : 'threats found'}
            </span>
            <span className="v muted">
              {formatInteger(result.files_scanned)} files · {formatFloat(result.duration_ms)} ms
            </span>
          </div>

          <div className="row">
            <span className="k">path</span>
            <span className="v mono">{result.path}</span>
          </div>

          <div className="tbl">
            {result.threats.length === 0 ? (
              <div className="muted">no threats detected</div>
            ) : (
              result.threats.map((threat, i) => (
                <ThreatRow key={`${threat.file_path}:${threat.name}:${i}`} threat={threat} />
              ))
            )}
          </div>
        </>
      )}
    </div>
  );
}
