import type { Checkpoint } from '../types';

const MAX_ROWS = 10;

export default function CheckpointsPanel({ checkpoints }: { checkpoints: Checkpoint[] }) {
  const rows = [...checkpoints].sort((a, b) => b.step - a.step).slice(0, MAX_ROWS);

  return (
    <div className="card">
      <div className="card-title">Checkpoints</div>

      <div className="scroll">
        <table className="tbl">
          <thead>
            <tr>
              <th>model</th>
              <th>step</th>
              <th className="right">loss</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td className="muted" colSpan={3}>
                  none
                </td>
              </tr>
            ) : (
              rows.map((ckpt) => (
                <tr key={ckpt.path}>
                  <td>{ckpt.model}</td>
                  <td className="mono">{ckpt.step.toLocaleString()}</td>
                  <td className="right mono">{ckpt.loss == null ? '—' : ckpt.loss.toFixed(4)}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
