import { useEffect, useState } from 'react';
import type { TrainingStatus, SystemStatus, Checkpoint, Job } from './types';
import TrainingPanel from './components/TrainingPanel';
import SystemPanel from './components/SystemPanel';
import CheckpointsPanel from './components/CheckpointsPanel';
import ActionsPanel from './components/ActionsPanel';
import JobsPanel from './components/JobsPanel';
import DatasetsPanel from './components/DatasetsPanel';
import ConfigsPanel from './components/ConfigsPanel';
import PipelinePanel from './components/PipelinePanel';
import PipelineManagerPanel from './components/PipelineManagerPanel';
import EvalsPanel from './components/EvalsPanel';
import LogsPanel from './components/LogsPanel';
import FeaturesPanel from './components/FeaturesPanel';
import ReasoningPanel from './components/ReasoningPanel';
import TokenizerPanel from './components/TokenizerPanel';
import CheckpointDetailPanel from './components/CheckpointDetailPanel';
import CheckpointComparePanel from './components/CheckpointComparePanel';
import ModelCardPanel from './components/ModelCardPanel';
import StoragePanel from './components/StoragePanel';
import MetricsChartPanel from './components/MetricsChartPanel';
import RouterPanel from './components/RouterPanel';
import ExportPanel from './components/ExportPanel';
import DataSourcesPanel from './components/DataSourcesPanel';
import EvalHistoryPanel from './components/EvalHistoryPanel';
import HealthPanel from './components/HealthPanel';
import TokenizePanel from './components/TokenizePanel';
import SftDataPanel from './components/SftDataPanel';
import ScriptsCatalogPanel from './components/ScriptsCatalogPanel';
import SystemInfoPanel from './components/SystemInfoPanel';
import ConfigDiffPanel from './components/ConfigDiffPanel';
import TokenizerHealthPanel from './components/TokenizerHealthPanel';
import DataStatsPanel from './components/DataStatsPanel';
import CheckpointHealthPanel from './components/CheckpointHealthPanel';
import ProjectMemoryPanel from './components/ProjectMemoryPanel';
import VectorIndexPanel from './components/VectorIndexPanel';
import SecurityScanPanel from './components/SecurityScanPanel';
import EnvCheckPanel from './components/EnvCheckPanel';
import VramEstimatePanel from './components/VramEstimatePanel';
import ProjectHealthPanel from './components/ProjectHealthPanel';
import BenchmarkResultsPanel from './components/BenchmarkResultsPanel';
import SafetyEvalPanel from './components/SafetyEvalPanel';
import FiltersCatalogPanel from './components/FiltersCatalogPanel';
import ReasoningProblemsPanel from './components/ReasoningProblemsPanel';
import ActionLauncherPanel from './components/ActionLauncherPanel';
import VocabExplorerPanel from './components/VocabExplorerPanel';
import ScoringConfigPanel from './components/ScoringConfigPanel';
import RegressionHistoryPanel from './components/RegressionHistoryPanel';
import LrFinderPanel from './components/LrFinderPanel';
import RepoScoresPanel from './components/RepoScoresPanel';
import TrainingManifestPanel from './components/TrainingManifestPanel';
import CheckpointAveragePanel from './components/CheckpointAveragePanel';
import BacklogPanel from './components/BacklogPanel';
import ResearchLogPanel from './components/ResearchLogPanel';
import DocsBrowserPanel from './components/DocsBrowserPanel';
import CollectDataPanel from './components/CollectDataPanel';
import CollapsibleSection from './components/CollapsibleSection';

interface Snapshot {
  training: TrainingStatus;
  system: SystemStatus;
  checkpoints: Checkpoint[];
  jobs: Job[];
}

interface EventStream {
  snap: Snapshot | null;
  reconnecting: boolean;
}

function useEventStream(): EventStream {
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [reconnecting, setReconnecting] = useState<boolean>(false);

  useEffect(() => {
    const es = new EventSource('/api/events');

    es.onmessage = (e: MessageEvent<string>) => {
      try {
        setSnap(JSON.parse(e.data) as Snapshot);
        setReconnecting(false);
      } catch {
        /* ignore malformed messages — keep last good snapshot */
      }
    };

    // EventSource auto-reconnects on drop; surface the gap for the header dot.
    es.onerror = () => setReconnecting(true);
    es.onopen = () => setReconnecting(false);

    return () => es.close();
  }, []);

  return { snap, reconnecting };
}

function useClock(): string {
  const [now, setNow] = useState<string>(() => new Date().toLocaleTimeString());

  useEffect(() => {
    const id = setInterval(() => setNow(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(id);
  }, []);

  return now;
}

export default function App() {
  const { snap, reconnecting } = useEventStream();
  const clock = useClock();
  const alive = !reconnecting && (snap?.training.alive ?? false);

  return (
    <>
      <header className="app-header">
        <span className={`dot ${alive ? 'live' : 'dead'}`} />
        <h1>Cola-Coder</h1>
        <span className="clock">{clock}</span>
      </header>

      <main className="app-grid">
        {/* Overview — the always-visible live dashboard. */}
        <CollapsibleSection title="Overview" storageKey="sec.overview" defaultOpen>
          <TrainingPanel training={snap?.training ?? null} />
          <SystemPanel system={snap?.system ?? null} />
          <MetricsChartPanel />
          <HealthPanel />
          <SystemInfoPanel />
        </CollapsibleSection>

        {/* Jobs & actions — run things, watch live logs. */}
        <CollapsibleSection title="Run & Jobs" storageKey="sec.jobs" defaultOpen>
          <ActionsPanel
            onRan={() => { /* jobs arrive via the event stream */ }}
            trainingAlive={alive}
          />
          <ActionLauncherPanel trainingAlive={alive} />
          <JobsPanel jobs={snap?.jobs ?? []} />
          <LogsPanel />
        </CollapsibleSection>

        {/* Checkpoints & models. */}
        <CollapsibleSection title="Checkpoints & Models" storageKey="sec.checkpoints">
          <CheckpointsPanel checkpoints={snap?.checkpoints ?? []} />
          <CheckpointDetailPanel checkpoints={snap?.checkpoints ?? []} />
          <CheckpointComparePanel checkpoints={snap?.checkpoints ?? []} />
          <CheckpointHealthPanel checkpoints={snap?.checkpoints ?? []} />
          <CheckpointAveragePanel checkpoints={snap?.checkpoints ?? []} />
          <TrainingManifestPanel />
          <ModelCardPanel checkpoints={snap?.checkpoints ?? []} />
          <RouterPanel />
          <ExportPanel />
        </CollapsibleSection>

        {/* Data. */}
        <CollapsibleSection title="Data" storageKey="sec.data">
          <CollectDataPanel />
          <DatasetsPanel />
          <DataStatsPanel />
          <DataSourcesPanel />
          <SftDataPanel />
          <VectorIndexPanel />
          <SecurityScanPanel />
          <FiltersCatalogPanel />
          <ScoringConfigPanel />
          <RepoScoresPanel />
        </CollapsibleSection>

        {/* Configs & pipeline. */}
        <CollapsibleSection title="Configs & Pipeline" storageKey="sec.pipeline">
          <ConfigsPanel />
          <VramEstimatePanel />
          <LrFinderPanel />
          <ConfigDiffPanel />
          <PipelinePanel />
          <PipelineManagerPanel />
        </CollapsibleSection>

        {/* Evaluation. */}
        <CollapsibleSection title="Evaluation" storageKey="sec.eval">
          <EvalsPanel />
          <EvalHistoryPanel />
          <BenchmarkResultsPanel />
          <SafetyEvalPanel />
          <RegressionHistoryPanel />
        </CollapsibleSection>

        {/* Tokenizer. */}
        <CollapsibleSection title="Tokenizer" storageKey="sec.tokenizer">
          <TokenizerPanel />
          <TokenizerHealthPanel />
          <TokenizePanel />
          <VocabExplorerPanel />
        </CollapsibleSection>

        {/* System & tools. */}
        <CollapsibleSection title="System & Tools" storageKey="sec.tools">
          <EnvCheckPanel />
          <ProjectHealthPanel />
          <FeaturesPanel />
          <ReasoningPanel />
          <ProjectMemoryPanel />
          <ReasoningProblemsPanel />
          <BacklogPanel />
          <ResearchLogPanel />
          <DocsBrowserPanel />
          <ScriptsCatalogPanel />
          <StoragePanel />
        </CollapsibleSection>
      </main>
    </>
  );
}
