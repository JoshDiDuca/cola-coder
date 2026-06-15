import type { ReactNode } from "react";
import { useLocalStorage } from "../hooks/useLocalStorage";

interface CollapsibleSectionProps {
  title: string;
  storageKey: string;
  defaultOpen?: boolean;
  children: ReactNode;
}

// Reusable collapsible wrapper. Persists its open/closed state to localStorage
// under `storageKey` so the choice survives reloads. Designed to sit INSIDE the
// `.app-grid` CSS grid: `.section` spans the full grid width and `.section-body`
// is its own nested grid so cards tile identically to cards outside a section.
export default function CollapsibleSection(props: CollapsibleSectionProps): JSX.Element {
  const { title, storageKey, defaultOpen = true, children } = props;
  const [open, setOpen] = useLocalStorage<boolean>(storageKey, defaultOpen);

  const toggle = (): void => {
    setOpen(!open);
  };

  return (
    <section className="section">
      <button
        type="button"
        className="section-header"
        aria-expanded={open}
        onClick={toggle}
      >
        <span className="section-chevron">{open ? "▾" : "▸"}</span>
        <span>{title}</span>
      </button>
      {open ? <div className="section-body">{children}</div> : null}
    </section>
  );
}
