import { useEffect, useState } from 'react';
import type { TrainingStatus, SystemStatus, Checkpoint, Job } from './types';
import { SECTION_BY_ID, type SectionId } from './sections';
import { useHashRoute } from './hooks/useHashRoute';
import Sidebar from './components/Sidebar';
import { formatFloat } from './format';

import CheckpointsScreen from './components/screens/CheckpointsScreen';
import DataScreen from './components/screens/DataScreen';
import EvalScreen from './components/screens/EvalScreen';
import RunScreen from './components/screens/RunScreen';
import SystemScreen from './components/screens/SystemScreen';
import PipelineScreen from './components/screens/PipelineScreen';
import TokenizerScreen from './components/screens/TokenizerScreen';
import InferenceScreen from './components/screens/InferenceScreen';
import ChatScreen from './components/screens/ChatScreen';
import FimScreen from './components/screens/FimScreen';
import BestOfNScreen from './components/screens/BestOfNScreen';

import LiveTrainingPanel from './components/LiveTrainingPanel';
import LossStabilityPanel from './components/LossStabilityPanel';
import TrainingPanel from './components/TrainingPanel';
import SystemPanel from './components/SystemPanel';
import MetricsChartPanel from './components/MetricsChartPanel';
import HealthPanel from './components/HealthPanel';
import SystemInfoPanel from './components/SystemInfoPanel';
import CheckpointToolsPanel from './components/CheckpointToolsPanel';
import DataToolsPanel from './components/DataToolsPanel';
import PipelineToolsPanel from './components/PipelineToolsPanel';
import ConfigEditorPanel from './components/ConfigEditorPanel';
import CommandPalette from './components/CommandPalette';
import CheckpointEvalSummary from './components/CheckpointEvalSummary';
import GpuProcessesPanel from './components/GpuProcessesPanel';

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

/** The panels that make up each routed page. Exactly one page renders at a time. */
function Page({
  section,
  snap,
  alive,
}: {
  section: SectionId;
  snap: Snapshot | null;
  alive: boolean;
}) {
  const checkpoints = snap?.checkpoints ?? [];
  switch (section) {
    case 'overview':
      return (
        <div className="page-grid">
          <LiveTrainingPanel training={snap?.training ?? null} />
          <LossStabilityPanel />
          <CheckpointEvalSummary />
          <TrainingPanel training={snap?.training ?? null} />
          <SystemPanel system={snap?.system ?? null} />
          <MetricsChartPanel />
          <HealthPanel />
          <SystemInfoPanel />
          <GpuProcessesPanel />
        </div>
      );
    case 'run':
      return <RunScreen jobs={snap?.jobs ?? []} trainingAlive={alive} />;
    case 'inference':
      return (
        <InferenceScreen checkpoints={checkpoints.map((c) => c.path)} trainingAlive={alive} />
      );
    case 'chat':
      return <ChatScreen checkpoints={checkpoints.map((c) => c.path)} trainingAlive={alive} />;
    case 'fim':
      return <FimScreen checkpoints={checkpoints.map((c) => c.path)} trainingAlive={alive} />;
    case 'bestof':
      return <BestOfNScreen checkpoints={checkpoints.map((c) => c.path)} trainingAlive={alive} />;
    case 'checkpoints':
      return (
        <>
          <CheckpointsScreen checkpoints={checkpoints} />
          <CheckpointToolsPanel checkpoints={checkpoints} />
        </>
      );
    case 'data':
      return (
        <>
          <DataScreen />
          <DataToolsPanel />
        </>
      );
    case 'pipeline':
      return (
        <>
          <PipelineScreen trainingAlive={alive} />
          <PipelineToolsPanel />
          <ConfigEditorPanel />
        </>
      );
    case 'eval':
      return <EvalScreen />;
    case 'tokenizer':
      return <TokenizerScreen />;
    case 'system':
      return <SystemScreen />;
    default: {
      const _exhaustive: never = section;
      return _exhaustive;
    }
  }
}

function TopStatus({ training, connected }: { training: TrainingStatus | null; connected: boolean }) {
  if (!connected) {
    return (
      <span className="status">
        <span className="dot dead" /> disconnected
      </span>
    );
  }
  if (!training?.alive) {
    return (
      <span className="status">
        <span className="dot" /> training idle
      </span>
    );
  }
  return (
    <span className="status">
      <span className="dot live" />
      step {training.step?.toLocaleString() ?? '—'}
      {training.loss !== null ? <> · loss {formatFloat(training.loss, 3)}</> : null}
      {training.ppl !== null ? <> · ppl {formatFloat(training.ppl, 1)}</> : null}
      {training.tok_per_s !== null ? <> · {formatFloat(training.tok_per_s, 0)} tok/s</> : null}
    </span>
  );
}

export default function App() {
  const { snap, reconnecting } = useEventStream();
  const clock = useClock();
  const [active, navigate] = useHashRoute();
  const connected = !reconnecting && snap !== null;
  const alive = connected && (snap?.training.alive ?? false);
  const nav = SECTION_BY_ID[active];

  return (
    <div className="app">
      <CommandPalette />
      <Sidebar active={active} onNavigate={navigate} training={snap?.training ?? null} connected={connected} />

      <div className="app-main">
        {reconnecting ? (
          <div className="conn-banner">
            <span className="dot dead" />
            <span>
              Can&rsquo;t reach the backend server. Is it running? Start it with{' '}
              <code>.\ps\cola-ui.ps1</code> and refresh this page.
            </span>
          </div>
        ) : null}
        <header className="topbar">
          <div className="topbar-title">
            <h1>{nav.label}</h1>
            <span className="topbar-sub">{nav.subtitle}</span>
          </div>
          <div className="spacer" />
          <TopStatus training={snap?.training ?? null} connected={connected} />
          <span className="clock">{clock}</span>
        </header>

        <main className="page">
          <Page section={active} snap={snap} alive={alive} />
        </main>
      </div>
    </div>
  );
}
