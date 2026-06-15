import { useMemo, useState } from 'react';
import MasterDetail, { type MasterItem } from '../MasterDetail';

import EnvCheckPanel from '../EnvCheckPanel';
import ProjectHealthPanel from '../ProjectHealthPanel';
import FeaturesPanel from '../FeaturesPanel';
import ReasoningPanel from '../ReasoningPanel';
import ReasoningProblemsPanel from '../ReasoningProblemsPanel';
import ProjectMemoryPanel from '../ProjectMemoryPanel';
import BacklogPanel from '../BacklogPanel';
import ResearchLogPanel from '../ResearchLogPanel';
import DocsBrowserPanel from '../DocsBrowserPanel';
import ScriptsCatalogPanel from '../ScriptsCatalogPanel';
import StoragePanel from '../StoragePanel';
import SpecialistsPanel from '../SpecialistsPanel';
import RetrievalSearchPanel from '../RetrievalSearchPanel';

// ── Master-detail "System & Tools" screen ───────────────────────────────────
// Left: a static, typed catalog of system/diagnostic tools, grouped by area via
// each item's subtitle. Right: the selected tool's existing panel, reused as-is
// (each panel fetches its own data). Replaces the old grid-of-cards layout.

interface SystemTool {
  id: string;
  title: string;
  /** Grouping label shown under the title in the list. */
  subtitle: string;
  render: () => JSX.Element;
}

const SYSTEM_TOOLS: readonly SystemTool[] = [
  { id: 'env', title: 'Environment Check', subtitle: 'Diagnostics', render: () => <EnvCheckPanel /> },
  { id: 'health', title: 'Project Health', subtitle: 'Diagnostics', render: () => <ProjectHealthPanel /> },
  { id: 'features', title: 'Features', subtitle: 'Configuration', render: () => <FeaturesPanel /> },
  { id: 'reasoning', title: 'Reasoning Config', subtitle: 'Configuration', render: () => <ReasoningPanel /> },
  { id: 'specialists', title: 'Domain Specialists', subtitle: 'Configuration', render: () => <SpecialistsPanel /> },
  { id: 'problems', title: 'Reasoning Problems', subtitle: 'Configuration', render: () => <ReasoningProblemsPanel /> },
  { id: 'search', title: 'Code Search', subtitle: 'Knowledge', render: () => <RetrievalSearchPanel /> },
  { id: 'memory', title: 'Project Memory', subtitle: 'Knowledge', render: () => <ProjectMemoryPanel /> },
  { id: 'backlog', title: 'Backlog', subtitle: 'Knowledge', render: () => <BacklogPanel /> },
  { id: 'research', title: 'Research Log', subtitle: 'Knowledge', render: () => <ResearchLogPanel /> },
  { id: 'docs', title: 'Docs', subtitle: 'Knowledge', render: () => <DocsBrowserPanel /> },
  { id: 'scripts', title: 'Scripts Catalog', subtitle: 'Reference', render: () => <ScriptsCatalogPanel /> },
  { id: 'storage', title: 'Storage', subtitle: 'Reference', render: () => <StoragePanel /> },
];

export default function SystemScreen(): JSX.Element {
  const [selectedId, setSelectedId] = useState<string>(SYSTEM_TOOLS[0].id);

  const items: MasterItem[] = useMemo(
    () =>
      SYSTEM_TOOLS.map((tool) => ({
        id: tool.id,
        title: tool.title,
        subtitle: tool.subtitle,
      })),
    [],
  );

  const selected = useMemo(
    () => SYSTEM_TOOLS.find((tool) => tool.id === selectedId) ?? null,
    [selectedId],
  );

  return (
    <MasterDetail
      items={items}
      selectedId={selectedId}
      onSelect={setSelectedId}
      listLabel={`${SYSTEM_TOOLS.length} tools`}
      emptyDetail="Select a tool to see details"
      detail={selected ? <div className="sys-detail">{selected.render()}</div> : null}
    />
  );
}
