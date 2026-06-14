import { useCallback, useEffect, useState } from 'react';
import type { FeaturesView, FeatureItem } from '../types';
import { isApiError } from '../types';
import { getFeatures } from '../api';

function formatValue(value: unknown): string | null {
  if (typeof value === 'boolean') return null;
  if (value === null || value === undefined) return null;
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function FeatureRow({ feat }: { feat: FeatureItem }) {
  const extra = formatValue(feat.value);
  return (
    <div className="row">
      <span className="mono">
        <span className={`dot ${feat.enabled ? 'live' : 'dead'}`} /> {feat.key}
      </span>
      <span className="v">{extra ?? (feat.enabled ? 'on' : 'off')}</span>
    </div>
  );
}

export default function FeaturesPanel() {
  const [view, setView] = useState<FeaturesView | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const resp = await getFeatures();
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
    let active = true;
    void (async () => {
      try {
        const resp = await getFeatures();
        if (!active) return;
        if (isApiError(resp)) setError(resp.error);
        else setView(resp);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="card card-wide">
      <div className="row">
        <div className="card-title">Features</div>
        <button className="btn" onClick={() => void load()}>
          refresh
        </button>
      </div>

      {error && <div className="err">{error}</div>}

      {view && (
        <div className="muted mono">
          {view.enabled} / {view.total} enabled · {view.path}
        </div>
      )}

      {view && view.groups.length === 0 && !error && (
        <div className="muted">no features defined</div>
      )}

      {view?.groups.map((group) => (
        <div key={group.category}>
          <div className="card-title">{group.category}</div>
          {group.features.map((feat) => (
            <FeatureRow key={feat.key} feat={feat} />
          ))}
        </div>
      ))}
    </div>
  );
}
