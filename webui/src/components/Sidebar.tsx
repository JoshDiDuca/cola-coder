import type { TrainingStatus } from '../types';
import { NAV, type SectionId } from '../sections';
import Icon from './Icon';

interface SidebarProps {
  active: SectionId;
  onNavigate: (id: SectionId) => void;
  training: TrainingStatus | null;
  connected: boolean;
}

function pillText(connected: boolean, training: TrainingStatus | null): string {
  if (!connected) {
    return 'server offline';
  }
  if (training?.alive) {
    return training.step !== null ? `training · step ${training.step.toLocaleString()}` : 'training';
  }
  return 'idle';
}

export default function Sidebar({ active, onNavigate, training, connected }: SidebarProps) {
  const live = connected && (training?.alive ?? false);
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark">◆</span>
        <span className="brand-name">Cola-Coder</span>
      </div>

      <nav className="nav">
        {NAV.map((g) => (
          <div className="nav-group" key={g.group}>
            <div className="nav-group-label">{g.group}</div>
            {g.items.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`nav-item${active === item.id ? ' active' : ''}`}
                title={item.subtitle}
                onClick={() => onNavigate(item.id)}
              >
                <Icon name={item.icon} />
                <span className="nav-label">{item.label}</span>
              </button>
            ))}
          </div>
        ))}
      </nav>

      <div className="sidebar-foot">
        <div className={`train-pill${live ? ' live' : ''}`}>
          <span className={`dot ${live ? 'live' : 'dead'}`} />
          <span className="train-pill-text">{pillText(connected, training)}</span>
        </div>
      </div>
    </aside>
  );
}
