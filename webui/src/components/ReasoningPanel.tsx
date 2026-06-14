import { useCallback, useEffect, useState } from 'react';
import type { JsonValue, ReasoningView } from '../types';
import { isApiError } from '../types';
import { getReasoning } from '../api';
import { formatJsonValue } from '../format';

interface SectionProps {
  title: string;
  data: Record<string, JsonValue>;
}

function Section({ title, data }: SectionProps) {
  return (
    <div>
      <div className="card-title">{title}</div>
      <pre className="pre scroll">{formatJsonValue(data)}</pre>
    </div>
  );
}

export default function ReasoningPanel() {
  const [view, setView] = useState<ReasoningView | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const resp = await getReasoning();
      if (isApiError(resp)) {
        setError(resp.error);
        setView(null);
      } else {
        setView(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const s = view?.summary;

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Reasoning</div>
        <button className="btn" onClick={() => void load()}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {view && <div className="muted mono">{view.path}</div>}

      {s && (
        <div>
          <div className="row">
            <span className="k">advantage_norm</span>
            <span className="v">{formatJsonValue(s.advantage_norm)}</span>
          </div>
          <div className="row">
            <span className="k">clip_epsilon</span>
            <span className="v">{formatJsonValue(s.clip_epsilon)}</span>
          </div>
          <div className="row">
            <span className="k">clip_epsilon_high</span>
            <span className="v">{formatJsonValue(s.clip_epsilon_high)}</span>
          </div>
          <div className="row">
            <span className="k">group_size</span>
            <span className="v">{formatJsonValue(s.group_size)}</span>
          </div>
          <div className="row">
            <span className="k">sft_warmup_enabled</span>
            <span className="v">{formatJsonValue(s.sft_warmup_enabled)}</span>
          </div>
          <div className="row">
            <span className="k">problem_source</span>
            <span className="v">{formatJsonValue(s.problem_source)}</span>
          </div>
        </div>
      )}

      {view && (
        <>
          <Section title="reasoning" data={view.reasoning} />
          <Section title="problem_set" data={view.problem_set} />
          <Section title="sft_warmup" data={view.sft_warmup} />
        </>
      )}
    </div>
  );
}
