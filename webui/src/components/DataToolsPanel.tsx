import { useState } from 'react';
import CombineDatasetsPanel from './CombineDatasetsPanel';
import DataSourcesPanel from './DataSourcesPanel';
import SftDataPanel from './SftDataPanel';
import VectorIndexPanel from './VectorIndexPanel';
import SecurityScanPanel from './SecurityScanPanel';
import FiltersCatalogPanel from './FiltersCatalogPanel';
import ScoringConfigPanel from './ScoringConfigPanel';
import RepoScoresPanel from './RepoScoresPanel';

// ── Tab identity ──────────────────────────────────────────────────────────────
// One polished tabbed container replaces the old eight-panel card grid ("More
// tools"). Each existing panel is wrapped unchanged, one per tab, and only the
// active tab is mounted (lazy mount) so each panel's fetch fires on first open.

type DataTool =
  | 'combine'
  | 'sources'
  | 'sft'
  | 'vector'
  | 'security'
  | 'filters'
  | 'scoring'
  | 'repos';

interface TabDef {
  id: DataTool;
  label: string;
}

const TABS: readonly TabDef[] = [
  { id: 'combine', label: 'Combine' },
  { id: 'sources', label: 'Data sources' },
  { id: 'sft', label: 'SFT data' },
  { id: 'vector', label: 'Vector index' },
  { id: 'security', label: 'Security scan' },
  { id: 'filters', label: 'Filters' },
  { id: 'scoring', label: 'Scoring config' },
  { id: 'repos', label: 'Repo scores' },
];

// ── Container ───────────────────────────────────────────────────────────────────

export default function DataToolsPanel(): JSX.Element {
  const [tab, setTab] = useState<DataTool>('combine');

  // Lazy-load: only the opened tab's panel is mounted, so its fetch fires on
  // first open. Switching back later re-mounts and refetches — cheap, and keeps
  // each panel's data fresh.
  function renderTab(active: DataTool): JSX.Element {
    switch (active) {
      case 'combine':
        return <CombineDatasetsPanel />;
      case 'sources':
        return <DataSourcesPanel />;
      case 'sft':
        return <SftDataPanel />;
      case 'vector':
        return <VectorIndexPanel />;
      case 'security':
        return <SecurityScanPanel />;
      case 'filters':
        return <FiltersCatalogPanel />;
      case 'scoring':
        return <ScoringConfigPanel />;
      case 'repos':
        return <RepoScoresPanel />;
      default: {
        const _exhaustive: never = active;
        return _exhaustive;
      }
    }
  }

  return (
    <div className="card card-wide">
      <div className="md-detail-head">
        <h1 className="md-detail-title">Data tools</h1>
        <div className="md-toolbar tokscreen-tabs dtools-tabs">
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

      <div className="dtools-body">{renderTab(tab)}</div>
    </div>
  );
}
