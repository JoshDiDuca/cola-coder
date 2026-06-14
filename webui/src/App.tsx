import { useEffect, useState } from 'react';
import type { TrainingStatus, SystemStatus, Checkpoint, Job } from './types';
import TrainingPanel from './components/TrainingPanel';
import SystemPanel from './components/SystemPanel';
import CheckpointsPanel from './components/CheckpointsPanel';
import ActionsPanel from './components/ActionsPanel';
import JobsPanel from './components/JobsPanel';
import DatasetsPanel from './components/DatasetsPanel';

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
        <TrainingPanel training={snap?.training ?? null} />
        <SystemPanel system={snap?.system ?? null} />
        <CheckpointsPanel checkpoints={snap?.checkpoints ?? []} />
        <ActionsPanel onRan={() => { /* jobs arrive via the event stream */ }} />
        <JobsPanel jobs={snap?.jobs ?? []} />
        <DatasetsPanel />
      </main>
    </>
  );
}
