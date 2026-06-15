import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { NAV, type SectionId } from '../sections';
import Icon from './Icon';

// ⌘K / Ctrl-K command palette — a fuzzy quick-switcher over the app's nav
// sections. Self-contained: it owns its own open/closed state via a global
// keydown listener, and navigates by setting the URL hash exactly as
// useHashRoute/Sidebar do (`#/<id>`). Pure frontend, no backend.

interface Command {
  id: SectionId;
  label: string;
  subtitle: string;
  icon: import('../sections').IconName;
  group: string;
}

const COMMANDS: Command[] = NAV.flatMap((g) =>
  g.items.map((item) => ({
    id: item.id,
    label: item.label,
    subtitle: item.subtitle,
    icon: item.icon,
    group: g.group,
  })),
);

function filterCommands(query: string): Command[] {
  const needle = query.trim().toLowerCase();
  if (needle === '') {
    return COMMANDS;
  }
  return COMMANDS.filter((cmd) => {
    const haystack = `${cmd.label} ${cmd.subtitle} ${cmd.group}`.toLowerCase();
    return haystack.includes(needle);
  });
}

function clamp(value: number, min: number, max: number): number {
  if (value < min) {
    return min;
  }
  if (value > max) {
    return max;
  }
  return value;
}

export default function CommandPalette(): JSX.Element | null {
  const [open, setOpen] = useState<boolean>(false);
  const [query, setQuery] = useState<string>('');
  const [activeIndex, setActiveIndex] = useState<number>(0);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const results = useMemo<Command[]>(() => filterCommands(query), [query]);

  // Global toggle/close listener. Ctrl+K or Cmd+K toggles; Escape closes.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent): void => {
      const isToggle = (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k';
      if (isToggle) {
        event.preventDefault();
        setOpen((prev) => !prev);
        return;
      }
      if (event.key === 'Escape') {
        setOpen(false);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  // Reset query/selection and focus the input each time the palette opens.
  useEffect(() => {
    if (open) {
      setQuery('');
      setActiveIndex(0);
      inputRef.current?.focus();
    }
  }, [open]);

  // Keep the highlighted row in range as the result list shrinks/grows.
  useEffect(() => {
    setActiveIndex((prev) => clamp(prev, 0, Math.max(results.length - 1, 0)));
  }, [results.length]);

  const activate = useCallback((command: Command): void => {
    window.location.hash = `/${command.id}`;
    setOpen(false);
  }, []);

  const onInputKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLInputElement>): void => {
      if (event.key === 'ArrowDown') {
        event.preventDefault();
        setActiveIndex((prev) => clamp(prev + 1, 0, Math.max(results.length - 1, 0)));
        return;
      }
      if (event.key === 'ArrowUp') {
        event.preventDefault();
        setActiveIndex((prev) => clamp(prev - 1, 0, Math.max(results.length - 1, 0)));
        return;
      }
      if (event.key === 'Enter') {
        event.preventDefault();
        const command = results[activeIndex];
        if (command !== undefined) {
          activate(command);
        }
      }
    },
    [results, activeIndex, activate],
  );

  if (!open) {
    return null;
  }

  return (
    <div
      className="cmdk-backdrop"
      role="presentation"
      onMouseDown={() => setOpen(false)}
    >
      <div
        className="cmdk-panel"
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <input
          ref={inputRef}
          className="cmdk-input"
          type="text"
          placeholder="Jump to a section…"
          value={query}
          spellCheck={false}
          autoComplete="off"
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={onInputKeyDown}
        />
        <div className="cmdk-list" role="listbox" aria-label="Sections">
          {results.length === 0 ? (
            <div className="cmdk-empty">No matches</div>
          ) : (
            results.map((command, index) => (
              <button
                key={command.id}
                type="button"
                role="option"
                aria-selected={index === activeIndex}
                className={`cmdk-row${index === activeIndex ? ' active' : ''}`}
                onMouseEnter={() => setActiveIndex(index)}
                onClick={() => activate(command)}
              >
                <span className="cmdk-row-icon">
                  <Icon name={command.icon} />
                </span>
                <span className="cmdk-row-text">
                  <span className="cmdk-row-label">{command.label}</span>
                  <span className="cmdk-row-subtitle">{command.subtitle}</span>
                </span>
                <span className="cmdk-row-group">{command.group}</span>
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
