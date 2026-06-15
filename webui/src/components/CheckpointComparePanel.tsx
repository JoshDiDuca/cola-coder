import { useCallback, useMemo, useState } from 'react';
import type { Checkpoint, CheckpointDetail, CompareResult } from '../types';
import { isApiError } from '../types';
import { getCheckpointCompare } from '../api';
import { formatInteger, formatParams } from '../format';

type DeltaTone = 'up' | 'down' | 'flat';

function deltaTone(value: number): DeltaTone {
  if (value > 0) return 'up';
  if (value < 0) return 'down';
  return 'flat';
}

function signedParams(n: number): string {
  const sign = n > 0 ? '+' : n < 0 ? '-' : '';
  return `${sign}${formatParams(Math.abs(n))}`;
}

function signedInteger(n: number): string {
  const sign = n > 0 ? '+' : '';
  return `${sign}${formatInteger(n)}`;
}

function label(ckpt: Checkpoint): string {
  return `${ckpt.model} / ${ckpt.name} @ ${formatInteger(ckpt.step)}`;
}

function moeTag(isMoe: boolean): JSX.Element {
  return <span className={`tag ${isMoe ? 'done' : 'failed'}`}>{isMoe ? 'moe' : 'dense'}</span>;
}

// One scalar row rendered across both sides — the heart of the side-by-side view.
function CompareRow({
  metric,
  a,
  b,
}: {
  metric: string;
  a: string;
  b: string;
}): JSX.Element {
  return (
    <div className="cmp-row">
      <span className="cmp-metric muted">{metric}</span>
      <span className="cmp-cell mono">{a}</span>
      <span className="cmp-cell mono">{b}</span>
    </div>
  );
}

function TagRow({ metric, items }: { metric: string; items: string[] }): JSX.Element {
  return (
    <div className="cmp-tagrow">
      <span className="cmp-metric muted">{metric}</span>
      <span className="cmp-tags">
        {items.length === 0 ? (
          <span className="muted">none</span>
        ) : (
          items.map((it) => (
            <span key={it} className="tag mono">
              {it}
            </span>
          ))
        )}
      </span>
    </div>
  );
}

function ResultView({ result }: { result: CompareResult }): JSX.Element {
  const { a, b, diff } = result;
  const paramsTone = deltaTone(diff.num_params_delta);
  const tensorTone = deltaTone(diff.tensor_count_delta);

  return (
    <div className="cmp-result">
      <div className="cmp-grid">
        <div className="cmp-head cmp-metric" />
        <SideHead title="A" detail={a} />
        <SideHead title="B" detail={b} />

        <CompareRow metric="params" a={formatParams(a.num_params)} b={formatParams(b.num_params)} />
        <CompareRow
          metric="tensors"
          a={formatInteger(a.tensor_count)}
          b={formatInteger(b.tensor_count)}
        />
        <div className="cmp-row">
          <span className="cmp-metric muted">type</span>
          <span className="cmp-cell">{moeTag(a.is_moe)}</span>
          <span className="cmp-cell">{moeTag(b.is_moe)}</span>
        </div>
      </div>

      <div className="cmp-deltas">
        <div className="cmp-delta">
          <span className="cmp-delta-label muted">Δ params</span>
          <span className={`cmp-delta-value mono tone-${paramsTone}`}>
            {signedParams(diff.num_params_delta)}
          </span>
        </div>
        <div className="cmp-delta">
          <span className="cmp-delta-label muted">Δ tensors</span>
          <span className={`cmp-delta-value mono tone-${tensorTone}`}>
            {signedInteger(diff.tensor_count_delta)}
          </span>
        </div>
        <div className="cmp-delta">
          <span className="cmp-delta-label muted">moe</span>
          <span className="cmp-delta-value">
            <span className={`tag ${diff.is_moe_changed ? 'running' : 'done'}`}>
              {diff.is_moe_changed ? 'changed' : 'same'}
            </span>
          </span>
        </div>
      </div>

      <div className="cmp-tagrows">
        <TagRow metric="metadata changed" items={diff.metadata_changed_keys} />
        <TagRow metric="dtypes only A" items={diff.dtypes_only_a} />
        <TagRow metric="dtypes only B" items={diff.dtypes_only_b} />
      </div>
    </div>
  );
}

function SideHead({ title, detail }: { title: string; detail: CheckpointDetail }): JSX.Element {
  return (
    <div className="cmp-head">
      <span className="cmp-head-title">{title}</span>
      <span className="cmp-head-path muted mono" title={detail.path}>
        {detail.path}
      </span>
    </div>
  );
}

export default function CheckpointComparePanel({
  checkpoints,
}: {
  checkpoints: Checkpoint[];
}): JSX.Element {
  const rows = useMemo(() => [...checkpoints].sort((a, b) => b.step - a.step), [checkpoints]);

  // Default A/B to the two newest checkpoints.
  const [aPath, setAPath] = useState<string>(() => rows[0]?.path ?? '');
  const [bPath, setBPath] = useState<string>(() => rows[1]?.path ?? '');
  const [result, setResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
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
        <div className="muted">need two distinct checkpoints to compare</div>
      ) : (
        <>
          <div className="cmp-controls">
            <label className="cmp-select">
              <span className="cmp-select-tag tag">A</span>
              <select
                className="select"
                value={aPath}
                onChange={(e) => setAPath(e.target.value)}
              >
                {rows.map((ckpt) => (
                  <option key={ckpt.path} value={ckpt.path}>
                    {label(ckpt)}
                  </option>
                ))}
              </select>
            </label>
            <label className="cmp-select">
              <span className="cmp-select-tag tag">B</span>
              <select
                className="select"
                value={bPath}
                onChange={(e) => setBPath(e.target.value)}
              >
                {rows.map((ckpt) => (
                  <option key={ckpt.path} value={ckpt.path}>
                    {label(ckpt)}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => void onCompare()}
              disabled={!canCompare || loading}
            >
              {loading ? '…comparing' : 'Compare'}
            </button>
          </div>

          {aPath !== '' && bPath !== '' && aPath === bPath && (
            <div className="muted">pick two distinct checkpoints</div>
          )}
          {error && <div className="err">{error}</div>}

          {result && !loading && <ResultView result={result} />}
        </>
      )}
    </div>
  );
}
