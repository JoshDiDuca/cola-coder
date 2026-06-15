import { useState } from 'react';
import type { Checkpoint } from '../types';
import ModelCardPanel from './ModelCardPanel';
import TrainingManifestPanel from './TrainingManifestPanel';
import CheckpointAveragePanel from './CheckpointAveragePanel';
import RouterPanel from './RouterPanel';
import ExportPanel from './ExportPanel';

// ── Tab identity ──────────────────────────────────────────────────────────────
// One polished tabbed container replaces the old grid of five checkpoint-tool
// cards. Each existing panel is wrapped unchanged and rendered one at a time;
// only the selected tab's panel is mounted (lazy), so its fetch fires on open.
// Pattern mirrors TokenizerScreen exactly (typed tab union + exhaustive switch).

type CkptTool = 'card' | 'manifest' | 'average' | 'router' | 'export';

interface TabDef {
  id: CkptTool;
  label: string;
}

const TABS: readonly TabDef[] = [
  { id: 'card', label: 'Model Card' },
  { id: 'manifest', label: 'Training Manifest' },
  { id: 'average', label: 'Average (soup)' },
  { id: 'router', label: 'Router' },
  { id: 'export', label: 'Export' },
];

interface CheckpointToolsPanelProps {
  checkpoints: Checkpoint[];
}

export default function CheckpointToolsPanel(props: CheckpointToolsPanelProps): JSX.Element {
  const { checkpoints } = props;
  const [tab, setTab] = useState<CkptTool>('card');

  // Lazy-mount: only the opened tab's panel exists in the tree, so its data
  // fetch fires on first open. Switching tabs re-mounts and refetches — cheap,
  // and keeps each panel's data fresh.
  function renderTab(active: CkptTool): JSX.Element {
    switch (active) {
      case 'card':
        return <ModelCardPanel checkpoints={checkpoints} />;
      case 'manifest':
        return <TrainingManifestPanel />;
      case 'average':
        return <CheckpointAveragePanel checkpoints={checkpoints} />;
      case 'router':
        return <RouterPanel />;
      case 'export':
        return <ExportPanel />;
      default: {
        const _exhaustive: never = active;
        return _exhaustive;
      }
    }
  }

  return (
    <div className="card card-wide">
      <div className="md-detail-head">
        <h1 className="md-detail-title">Checkpoint tools</h1>
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

      <div className="ctools-body">{renderTab(tab)}</div>
    </div>
  );
}
