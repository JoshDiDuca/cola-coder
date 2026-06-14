import { useCallback, useState } from 'react';
import type { Checkpoint, ModelCard } from '../types';
import { isApiError } from '../types';
import { getModelCard } from '../api';

const MAX_ROWS = 30;

function humanParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

function renderCell(value: unknown): string {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function KvTable({ title, data }: { title: string; data: Record<string, unknown> }) {
  const entries = Object.entries(data);
  return (
    <div>
      <div className="card-title">{title}</div>
      {entries.length === 0 ? (
        <div className="muted">none</div>
      ) : (
        <table className="tbl">
          <tbody>
            {entries.map(([k, v]) => (
              <tr key={k}>
                <td className="k">{k}</td>
                <td className="right mono">{renderCell(v)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default function ModelCardPanel({ checkpoints }: { checkpoints: Checkpoint[] }) {
  const rows = [...checkpoints].sort((a, b) => b.step - a.step).slice(0, MAX_ROWS);

  const [selectedPath, setSelectedPath] = useState<string>('');
  const [card, setCard] = useState<ModelCard | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSelect = useCallback(async (path: string) => {
    setSelectedPath(path);
    setCard(null);
    setError(null);
    if (path === '') return;
    setLoading(true);
    try {
      const resp = await getModelCard(path);
      if (isApiError(resp)) setError(resp.error);
      else setCard(resp);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="card card-wide">
      <div className="card-title">Model Card</div>

      {rows.length === 0 ? (
        <div className="muted">no checkpoints — train a model first</div>
      ) : (
        <div className="row">
          <span className="k">checkpoint</span>
          <select
            className="select"
            value={selectedPath}
            onChange={(e) => void onSelect(e.target.value)}
          >
            <option value="">select a checkpoint…</option>
            {rows.map((ckpt) => (
              <option key={ckpt.path} value={ckpt.path}>
                {ckpt.model} · step {ckpt.step.toLocaleString()}
              </option>
            ))}
          </select>
        </div>
      )}

      {selectedPath !== '' && (
        <>
          <div className="muted mono">{selectedPath}</div>
          {error && <div className="err">{error}</div>}
          {loading && <div className="muted">loading…</div>}

          {card && (
            <>
              <div className="row">
                <span className="k">name</span>
                <span className="v">{card.name}</span>
              </div>
              <div className="row">
                <span className="k">num_params</span>
                <span className="v">{humanParams(card.num_params)}</span>
              </div>
              <div className="row">
                <span className="k">is_moe</span>
                <span>
                  <span className={`tag ${card.is_moe ? 'done' : 'failed'}`}>
                    {card.is_moe ? 'moe' : 'dense'}
                  </span>
                </span>
              </div>

              <KvTable title="architecture" data={card.architecture} />
              <KvTable title="training" data={card.training} />

              {card.tokenizer === null ? (
                <div>
                  <div className="card-title">tokenizer</div>
                  <div className="muted">none</div>
                </div>
              ) : (
                <KvTable title="tokenizer" data={card.tokenizer} />
              )}

              <div className="card-title">markdown</div>
              <pre className="pre scroll">{card.markdown}</pre>
            </>
          )}
        </>
      )}
    </div>
  );
}
