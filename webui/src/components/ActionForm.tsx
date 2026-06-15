import { useEffect, useMemo, useState } from 'react';
import type { ActionParam, ConfigFile } from '../types';
import { getConfigs } from '../api';

interface ActionFormProps {
  params: ActionParam[];
  // Available checkpoint paths for `checkpoint`-typed params. When omitted,
  // checkpoint params fall back to a free-text input.
  checkpoints?: string[];
  // Fired whenever a field changes; receives the fully-built CLI arg vector.
  onArgs: (args: string[]) => void;
}

// Field values are uniformly stored as strings, keyed by param.name.
// Booleans are stored as the literal strings "true" / "false" so the whole
// map stays a flat Record<string, string> (no union gymnastics downstream).
type FieldValues = Record<string, string>;

function initialValues(params: ActionParam[]): FieldValues {
  const values: FieldValues = {};
  for (const p of params) {
    if (p.type === 'bool') {
      // A bool default of "true"/"1"/"yes" pre-checks the box; else unchecked.
      values[p.name] = isTruthy(p.default) ? 'true' : 'false';
    } else {
      values[p.name] = p.default ?? '';
    }
  }
  return values;
}

function isTruthy(value: string | null | undefined): boolean {
  if (value === null || value === undefined) return false;
  const v = value.trim().toLowerCase();
  return v === 'true' || v === '1' || v === 'yes' || v === 'on';
}

/**
 * Pure, exported helper: turn the current field values into a CLI arg vector.
 *
 * Rules:
 *  - bool: push `flag` only when truthy (store_true semantics); never a value.
 *  - positional (flag === ''): push the bare value when non-empty.
 *  - everything else: push [flag, value] when the value is non-empty.
 *  - required fields are always included (their flag, plus value for non-bool),
 *    even when empty — so the backend reports the missing value rather than the
 *    UI silently dropping it.
 */
export function buildArgs(params: ActionParam[], values: FieldValues): string[] {
  const args: string[] = [];
  for (const p of params) {
    const raw = values[p.name] ?? '';
    const value = raw.trim();

    if (p.type === 'bool') {
      // store_true: presence of the flag is the signal.
      if (isTruthy(raw) || (p.required === true && p.flag !== '')) {
        if (p.flag !== '') args.push(p.flag);
      }
      continue;
    }

    const hasValue = value !== '';
    if (!hasValue && p.required !== true) {
      // Skip empty optional values.
      continue;
    }

    if (p.flag === '') {
      // Positional argument — value only.
      if (hasValue || p.required === true) args.push(value);
    } else if (hasValue) {
      args.push(p.flag, value);
    } else if (p.required === true) {
      // Required but empty: surface the flag so the backend complains clearly.
      args.push(p.flag, value);
    }
  }
  return args;
}

interface FieldProps {
  param: ActionParam;
  value: string;
  configs: ConfigFile[];
  checkpoints?: string[];
  onChange: (name: string, value: string) => void;
}

function FieldControl({ param, value, configs, checkpoints, onChange }: FieldProps): JSX.Element {
  const id = `arg-${param.name}`;
  const set = (next: string): void => onChange(param.name, next);

  switch (param.type) {
    case 'config':
      return (
        <select
          id={id}
          className="select arg-control"
          value={value}
          onChange={(e) => set(e.target.value)}
        >
          <option value="">(none)</option>
          {configs.map((c) => (
            <option key={c.path} value={c.path}>
              {c.rel}
            </option>
          ))}
        </select>
      );

    case 'checkpoint':
      if (checkpoints && checkpoints.length > 0) {
        return (
          <select
            id={id}
            className="select arg-control"
            value={value}
            onChange={(e) => set(e.target.value)}
          >
            <option value="">(none)</option>
            {checkpoints.map((ckpt) => (
              <option key={ckpt} value={ckpt}>
                {ckpt}
              </option>
            ))}
          </select>
        );
      }
      return (
        <input
          id={id}
          type="text"
          className="input arg-control mono"
          value={value}
          onChange={(e) => set(e.target.value)}
          placeholder="checkpoint path"
          spellCheck={false}
        />
      );

    case 'choice':
      return (
        <select
          id={id}
          className="select arg-control"
          value={value}
          onChange={(e) => set(e.target.value)}
        >
          {param.required !== true && <option value="">(none)</option>}
          {(param.choices ?? []).map((choice) => (
            <option key={choice} value={choice}>
              {choice}
            </option>
          ))}
        </select>
      );

    case 'bool':
      return (
        <input
          id={id}
          type="checkbox"
          className="arg-checkbox"
          checked={isTruthy(value)}
          onChange={(e) => set(e.target.checked ? 'true' : 'false')}
        />
      );

    case 'int':
      return (
        <input
          id={id}
          type="number"
          step={1}
          className="input arg-control"
          value={value}
          onChange={(e) => set(e.target.value)}
          placeholder={param.default ?? ''}
        />
      );

    case 'float':
      return (
        <input
          id={id}
          type="number"
          step="any"
          className="input arg-control"
          value={value}
          onChange={(e) => set(e.target.value)}
          placeholder={param.default ?? ''}
        />
      );

    case 'string':
    case 'path':
      return (
        <input
          id={id}
          type="text"
          className="input arg-control"
          value={value}
          onChange={(e) => set(e.target.value)}
          placeholder={param.default ?? ''}
          spellCheck={false}
        />
      );

    default: {
      const _exhaustive: never = param.type;
      return _exhaustive;
    }
  }
}

/**
 * Controlled form rendering one typed control per ActionParam.
 *
 * Holds field values in local state (all as strings) and emits the built CLI
 * arg vector via `onArgs` whenever a value changes or the param set updates.
 */
export default function ActionForm({ params, checkpoints, onArgs }: ActionFormProps): JSX.Element {
  const [values, setValues] = useState<FieldValues>(() => initialValues(params));
  const [configs, setConfigs] = useState<ConfigFile[]>([]);

  // Whether any param needs the configs list (avoids a needless fetch).
  const needsConfigs = useMemo(() => params.some((p) => p.type === 'config'), [params]);

  // Reset field values whenever the param set changes (keyed by name + type).
  const paramsKey = useMemo(
    () => params.map((p) => `${p.name}:${p.type}:${p.flag}`).join('|'),
    [params],
  );
  useEffect(() => {
    setValues(initialValues(params));
    // paramsKey is the structural identity of `params`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paramsKey]);

  useEffect(() => {
    if (!needsConfigs) return;
    let active = true;
    void (async () => {
      try {
        const list = await getConfigs();
        if (active) setConfigs(list);
      } catch {
        // Non-fatal: the config <select> just shows only "(none)".
        if (active) setConfigs([]);
      }
    })();
    return () => {
      active = false;
    };
  }, [needsConfigs]);

  // Emit the built args on every value (or param) change.
  useEffect(() => {
    onArgs(buildArgs(params, values));
    // onArgs identity is owned by the parent; re-emitting on its change is fine.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values, paramsKey]);

  const onChange = (name: string, next: string): void => {
    setValues((prev) => ({ ...prev, [name]: next }));
  };

  return (
    <div className="arg-form">
      {params.map((p) => (
        <div className="arg-field" key={p.name}>
          <label className="arg-field-label" htmlFor={`arg-${p.name}`}>
            {p.label}
            {p.required === true && <span className="arg-field-required"> *</span>}
          </label>
          <FieldControl
            param={p}
            value={values[p.name] ?? ''}
            configs={configs}
            checkpoints={checkpoints}
            onChange={onChange}
          />
          {p.help && (
            <span className="arg-field-hint muted" title={p.help}>
              {p.help}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}
