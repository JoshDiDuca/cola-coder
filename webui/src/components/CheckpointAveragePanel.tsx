import { useCallback, useMemo, useState } from 'react';
import type { Checkpoint, Job } from '../types';
import { runAction } from '../api';

/** Averaging method exposed by scripts/average_checkpoints.py (`--method`). */
type AverageMethod = 'uniform' | 'ema';

interface CheckpointAveragePanelProps {
  /** Available checkpoints (App passes `snap?.checkpoints ?? []`). */
  checkpoints: Checkpoint[];
}

function label(ckpt: Checkpoint): string {
  return `${ckpt.model} / ${ckpt.name} @ ${ckpt.step.toLocaleString()}`;
}

/**
 * Checkpoint Averaging (Model Soup) launcher.
 *
 * Multi-select 2+ checkpoints (ordered oldest→newest by step, which is the
 * order `average_checkpoints.py` expects), pick an averaging method, set an
 * output directory, then launch `scripts/average_checkpoints.py` as a
 * background job via the allow-listed `average_checkpoints` action.
 *
 * This is CPU weight-averaging — it reads each selected checkpoint's weights
 * and writes a single soup; no GPU trainer is involved, so there is no
 * trainingAlive guard.
 */
export default function CheckpointAveragePanel({ checkpoints }: CheckpointAveragePanelProps) {
  // Oldest first — the script treats the leading checkpoint as oldest and (for
  // EMA) weights newer ones more heavily.
  const rows = useMemo<Checkpoint[]>(
    () => [...checkpoints].sort((a, b) => a.step - b.step),
    [checkpoints],
  );

  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
  const [method, setMethod] = useState<AverageMethod>('uniform');
  const [output, setOutput] = useState<string>('checkpoints/soup');

  const [pending, setPending] = useState<boolean>(false);
  const [launched, setLaunched] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  const toggle = useCallback((path: string): void => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
    setLaunched(null);
    setError(null);
  }, []);

  // Selected paths in oldest→newest order (follow the table/display order).
  const selectedPaths = useMemo<string[]>(
    () => rows.filter((c) => selected.has(c.path)).map((c) => c.path),
    [rows, selected],
  );

  const canCreate: boolean =
    selectedPaths.length >= 2 && output.trim().length > 0 && !pending;

  const onCreate = useCallback(async (): Promise<void> => {
    if (selectedPaths.length < 2 || output.trim().length === 0) return;
    setPending(true);
    setLaunched(null);
    setError(null);
    try {
      const args: string[] = [
        '--checkpoints',
        ...selectedPaths,
        '--method',
        method,
        '--output',
        output.trim(),
      ];
      const job = await runAction('average_checkpoints', args);
      setLaunched(job);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(false);
    }
  }, [selectedPaths, method, output]);

  return (
    <div className="card card-wide">
      <div className="card-title">Checkpoint Averaging (Model Soup)</div>
      <div className="muted" style={{ marginBottom: 8 }}>
        Pick 2+ checkpoints to average into one model. Reads each checkpoint's
        weights (CPU) and writes a single soup — no GPU trainer involved.
      </div>

      {rows.length < 2 ? (
        <div className="muted">need at least two checkpoints to average</div>
      ) : (
        <>
          <table className="tbl">
            <thead>
              <tr>
                <th className="right">pick</th>
                <th>checkpoint</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((ckpt) => {
                const checked = selected.has(ckpt.path);
                return (
                  <tr key={ckpt.path}>
                    <td className="right">
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => toggle(ckpt.path)}
                        disabled={pending}
                        aria-label={`select ${label(ckpt)}`}
                      />
                    </td>
                    <td>
                      <div>{label(ckpt)}</div>
                      <div className="muted mono">{ckpt.path}</div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <div className="muted" style={{ marginTop: 8 }}>
            {selectedPaths.length} selected (averaged oldest → newest by step)
          </div>

          <div
            className="row"
            style={{ borderBottom: 'none', flexWrap: 'wrap', marginTop: 8 }}
          >
            <label className="muted" htmlFor="soup-method">
              method
            </label>
            <select
              id="soup-method"
              className="select"
              value={method}
              onChange={(e) => setMethod(e.target.value as AverageMethod)}
              disabled={pending}
              style={{ minWidth: 160 }}
            >
              <option value="uniform">uniform (simple mean)</option>
              <option value="ema">ema (newer weighted more)</option>
            </select>
          </div>

          <div
            className="row"
            style={{ borderBottom: 'none', flexWrap: 'wrap', marginTop: 8 }}
          >
            <label className="muted" htmlFor="soup-output">
              output dir
            </label>
            <input
              id="soup-output"
              className="input"
              value={output}
              onChange={(e) => setOutput(e.target.value)}
              placeholder="checkpoints/soup"
              spellCheck={false}
              disabled={pending}
              style={{ flex: 1, minWidth: 220 }}
            />
          </div>

          <div style={{ marginTop: 12 }}>
            <button
              className="btn btn-primary"
              onClick={() => void onCreate()}
              disabled={!canCreate}
            >
              {pending ? '…creating' : '▶ Create Soup'}
            </button>
          </div>

          {selectedPaths.length < 2 && (
            <div className="muted" style={{ marginTop: 8 }}>
              select at least two checkpoints
            </div>
          )}

          {launched !== null && (
            <div className="muted mono" style={{ marginTop: 8 }}>
              launched {launched.name} ({launched.id}) — {launched.status}
            </div>
          )}
          {error !== null && (
            <div className="err" style={{ marginTop: 8 }}>
              {error}
            </div>
          )}
        </>
      )}
    </div>
  );
}
