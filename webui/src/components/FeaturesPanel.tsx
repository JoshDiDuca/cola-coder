import { useCallback, useEffect, useState } from 'react';
import type { FeaturesView, FeatureItem } from '../types';
import { isApiError } from '../types';
import { formatJsonValue } from '../format';
import { getFeatures, setFeature } from '../api';

interface FeatureRowProps {
  feat: FeatureItem;
  busy: boolean;
  onToggle: (feat: FeatureItem) => void;
}

function FeatureRow({ feat, busy, onToggle }: FeatureRowProps) {
  // Booleans are conveyed by the on/off toggle and dot; show only non-boolean
  // values (numbers, strings, nested config) alongside the toggle.
  const extra = typeof feat.value === 'boolean' || feat.value === null ? null : formatJsonValue(feat.value);
  return (
    <div className="row">
      <span className="mono">
        <span className={`dot ${feat.enabled ? 'live' : 'dead'}`} /> {feat.key}
      </span>
      <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        {extra !== null && <span className="v">{extra}</span>}
        <button
          className={`tag ${feat.enabled ? 'done' : 'failed'}`}
          style={{ cursor: busy ? 'not-allowed' : 'pointer', opacity: busy ? 0.45 : 1 }}
          disabled={busy}
          onClick={() => onToggle(feat)}
        >
          {feat.enabled ? 'on' : 'off'}
        </button>
      </span>
    </div>
  );
}

export default function FeaturesPanel() {
  const [view, setView] = useState<FeaturesView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [toggleError, setToggleError] = useState<string | null>(null);

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

  const onToggle = useCallback(
    async (feat: FeatureItem) => {
      setToggleError(null);
      setBusyKey(feat.key);
      try {
        const resp = await setFeature(feat.key, !feat.enabled);
        if (isApiError(resp)) {
          setToggleError(`${feat.key}: ${resp.error}`);
        } else {
          await load();
        }
      } catch (e) {
        setToggleError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusyKey(null);
      }
    },
    [load]
  );

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
      {toggleError && <div className="err">{toggleError}</div>}

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
            <FeatureRow
              key={feat.key}
              feat={feat}
              busy={busyKey === feat.key}
              onToggle={(f) => void onToggle(f)}
            />
          ))}
        </div>
      ))}
    </div>
  );
}
