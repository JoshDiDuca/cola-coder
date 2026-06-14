import { useCallback, useState } from 'react';
import type { Checkpoint, CheckpointDetail, CompareResult } from '../types';
import { isApiError } from '../types';
import { getCheckpointCompare } from '../api';

function humanParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

function signedParams(n: number): string {
  const sign = n > 0 ? '+' : n < 0 ? '-' : '';
  return `${sign}${humanParams(Math.abs(n))}`;
}

function signedInt(n: number): string {
  const sign = n > 0 ? '+' : '';
  return `${sign}${n.toLocaleString()}`;
}

function label(ckpt: Checkpoint): string {
  return `${ckpt.model} / ${ckpt.name} @ ${ckpt.step.toLocaleString()}`;
}

function SideColumn({ title, detail }: { title: string; detail: CheckpointDetail }) {
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div className="card-title">{title}</div>
      <div className="muted mono">{detail.path}</div>
      <div className="row">
        <span className="k">num_params</span>
        <span className="v">{humanParams(detail.num_params)}</span>
      </div>
      <div className="row">
        <span className="k">tensor_count</span>
        <span className="v">{detail.tensor_count.toLocaleString()}</span>
      </div>
      <div className="row" style={{ borderBottom: 'none' }}>
        <span className="k">is_moe</span>
        <span>
          <span className={`tag ${detail.is_moe ? 'done' : 'failed'}`}>
            {detail.is_moe ? 'moe' : 'dense'}
          </span>
        </span>
      </div>
    </div>
  );
}

export default function CheckpointComparePanel({ checkpoints }: { checkpoints: Checkpoint[] }) {
  const rows = [...checkpoints].sort((a, b) => b.step - a.step);

  const [aPath, setAPath] = useState<string>('');
  const [bPath, setBPath] = useState<string>('');
  const [result, setResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canCompare = aPath !== '' && bPath !== '' && aPath !== bPath;

  const onCompare = useCallback(async () => {
    if (!canCompare) return;
    setResult(null);
    setError(null);
    setLoading(true);
    try {
      const resp = await getCheckpointCompare(aPath, bPath);
      if (isApiError(resp)) setError(resp.error);
      else setResult(resp);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [aPath, bPath, canCompare]);

  return (
    <div className="card card-wide">
      <div className="card-title">Checkpoint Compare</div>

      {rows.length < 2 ? (
        <div className="muted">need two distinct checkpoints</div>
      ) : (
        <>
          <div className="row" style={{ borderBottom: 'none', flexWrap: 'wrap' }}>
            <select
              className="select"
              value={aPath}
              onChange={(e) => setAPath(e.target.value)}
              style={{ flex: 1, minWidth: 180 }}
            >
              <option value="">A: select checkpoint…</option>
              {rows.map((ckpt) => (
                <option key={ckpt.path} value={ckpt.path}>
                  {label(ckpt)}
                </option>
              ))}
            </select>
            <select
              className="select"
              value={bPath}
              onChange={(e) => setBPath(e.target.value)}
              style={{ flex: 1, minWidth: 180 }}
            >
              <option value="">B: select checkpoint…</option>
              {rows.map((ckpt) => (
                <option key={ckpt.path} value={ckpt.path}>
                  {label(ckpt)}
                </option>
              ))}
            </select>
            <button
              className="btn btn-primary"
              onClick={() => void onCompare()}
              disabled={!canCompare || loading}
            >
              Compare
            </button>
          </div>

          {aPath !== '' && bPath !== '' && aPath === bPath && (
            <div className="muted">pick two distinct checkpoints</div>
          )}
          {error && <div className="err">{error}</div>}
          {loading && <div className="muted">loading…</div>}

          {result && (
            <>
              <div className="card-title">diff</div>
              <div className="row">
                <span className="k">num_params_delta</span>
                <span className="v">{signedParams(result.diff.num_params_delta)}</span>
              </div>
              <div className="row">
                <span className="k">tensor_count_delta</span>
                <span className="v">{signedInt(result.diff.tensor_count_delta)}</span>
              </div>
              <div className="row">
                <span className="k">is_moe_changed</span>
                <span>
                  <span className={`tag ${result.diff.is_moe_changed ? 'running' : 'done'}`}>
                    {result.diff.is_moe_changed ? 'changed' : 'same'}
                  </span>
                </span>
              </div>

              <div className="row" style={{ flexWrap: 'wrap', gap: 6 }}>
                <span className="k">metadata_changed_keys</span>
                {result.diff.metadata_changed_keys.length === 0 ? (
                  <span className="muted">none</span>
                ) : (
                  result.diff.metadata_changed_keys.map((k) => (
                    <span key={k} className="tag mono">
                      {k}
                    </span>
                  ))
                )}
              </div>

              <div className="row" style={{ flexWrap: 'wrap', gap: 6 }}>
                <span className="k">dtypes_only_a</span>
                {result.diff.dtypes_only_a.length === 0 ? (
                  <span className="muted">none</span>
                ) : (
                  result.diff.dtypes_only_a.map((dt) => (
                    <span key={dt} className="tag mono">
                      {dt}
                    </span>
                  ))
                )}
              </div>

              <div className="row" style={{ flexWrap: 'wrap', gap: 6, borderBottom: 'none' }}>
                <span className="k">dtypes_only_b</span>
                {result.diff.dtypes_only_b.length === 0 ? (
                  <span className="muted">none</span>
                ) : (
                  result.diff.dtypes_only_b.map((dt) => (
                    <span key={dt} className="tag mono">
                      {dt}
                    </span>
                  ))
                )}
              </div>

              <div className="row" style={{ alignItems: 'flex-start', gap: 24, borderBottom: 'none' }}>
                <SideColumn title="A" detail={result.a} />
                <SideColumn title="B" detail={result.b} />
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
