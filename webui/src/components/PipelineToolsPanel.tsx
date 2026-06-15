import { useState } from 'react';
import ConfigsPanel from './ConfigsPanel';
import VramEstimatePanel from './VramEstimatePanel';
import LrFinderPanel from './LrFinderPanel';
import ConfigDiffPanel from './ConfigDiffPanel';
import PipelinePanel from './PipelinePanel';
import PipelineManagerPanel from './PipelineManagerPanel';

// ── Tab identity ──────────────────────────────────────────────────────────────
// One polished tabbed container replaces the old six-panel "More tools" card
// grid. Each existing panel is wrapped unchanged, one per tab, and lazy-mounted
// on first open (Configs is the default tab).

type PipeTool = 'configs' | 'vram' | 'lr' | 'diff' | 'pipeline' | 'manager';

interface PipeToolDef {
  id: PipeTool;
  label: string;
}

const TABS: readonly PipeToolDef[] = [
  { id: 'configs', label: 'Configs' },
  { id: 'vram', label: 'VRAM estimate' },
  { id: 'lr', label: 'LR finder' },
  { id: 'diff', label: 'Config diff' },
  { id: 'pipeline', label: 'Pipeline' },
  { id: 'manager', label: 'Pipeline manager' },
];

// ── Panel ───────────────────────────────────────────────────────────────────

export default function PipelineToolsPanel(): JSX.Element {
  const [tab, setTab] = useState<PipeTool>('configs');

  // Lazy-mount: only the active tab's panel is rendered, so each panel's own
  // fetch fires on first open. Switching away unmounts it; switching back
  // re-mounts and refetches — cheap, and keeps each tool's data fresh.
  function renderTab(active: PipeTool): JSX.Element {
    switch (active) {
      case 'configs':
        return <ConfigsPanel />;
      case 'vram':
        return <VramEstimatePanel />;
      case 'lr':
        return <LrFinderPanel />;
      case 'diff':
        return <ConfigDiffPanel />;
      case 'pipeline':
        return <PipelinePanel />;
      case 'manager':
        return <PipelineManagerPanel />;
      default: {
        const _exhaustive: never = active;
        return _exhaustive;
      }
    }
  }

  return (
    <div className="card card-wide">
      <div className="md-detail-head">
        <h1 className="md-detail-title">Config &amp; pipeline tools</h1>
        <div className="md-toolbar tokscreen-tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              className={`btn${tab === t.id ? ' btn-primary' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      <div className="ptools-body">{renderTab(tab)}</div>
    </div>
  );
}
