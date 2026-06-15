import type { Checkpoint } from '../types';
import { formatInteger, formatFloat, formatRelativeTime } from '../format';

interface ModelGroup {
  model: string;
  entries: Checkpoint[];
}

// Group checkpoints by model, newest-step first within each group, and order
// groups by their most recent checkpoint step. Pure — no side effects.
function groupByModel(checkpoints: Checkpoint[]): ModelGroup[] {
  const byModel = new Map<string, Checkpoint[]>();
  for (const ckpt of checkpoints) {
    const list = byModel.get(ckpt.model);
    if (list === undefined) byModel.set(ckpt.model, [ckpt]);
    else list.push(ckpt);
  }

  const groups: ModelGroup[] = [];
  for (const [model, entries] of byModel) {
    entries.sort((a, b) => b.step - a.step);
    groups.push({ model, entries });
  }
  groups.sort((a, b) => b.entries[0].step - a.entries[0].step);
  return groups;
}

function CheckpointRow({ ckpt, latest }: { ckpt: Checkpoint; latest: boolean }): JSX.Element {
  return (
    <div className="ckpt-row">
      <div className="ckpt-row-main">
        <div className="ckpt-step">
          <span className="ckpt-step-num mono">{formatInteger(ckpt.step)}</span>
          <span className="ckpt-step-label muted">step</span>
          {latest && <span className="tag done">latest</span>}
        </div>
        <div className="ckpt-meta">
          <span className="ckpt-loss">
            <span className="muted">loss</span>{' '}
            <span className="mono">{formatFloat(ckpt.loss, 4)}</span>
          </span>
          <span className="ckpt-saved muted">{formatRelativeTime(ckpt.mtime)}</span>
        </div>
      </div>
      <div className="ckpt-path muted mono" title={ckpt.path}>
        {ckpt.path}
      </div>
    </div>
  );
}

export default function CheckpointsPanel({ checkpoints }: { checkpoints: Checkpoint[] }): JSX.Element {
  const groups = groupByModel(checkpoints);

  return (
    <div className="card">
      <div className="card-title">Checkpoints</div>

      {groups.length === 0 ? (
        <div className="muted">no checkpoints saved yet</div>
      ) : (
        <div className="ckpt-groups scroll">
          {groups.map((group) => (
            <section className="ckpt-group" key={group.model}>
              <div className="ckpt-group-head">
                <span className="ckpt-group-name">{group.model}</span>
                <span className="ckpt-group-count muted mono">{group.entries.length}</span>
              </div>
              <div className="ckpt-list">
                {group.entries.map((ckpt, i) => (
                  <CheckpointRow key={ckpt.path} ckpt={ckpt} latest={i === 0} />
                ))}
              </div>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}
