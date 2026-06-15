import { useCallback, useEffect, useState } from 'react';
import { DEFAULT_SECTION, SECTION_IDS, type SectionId } from '../sections';

/**
 * Hash-based router for the app shell: the active section lives in `location.hash`
 * (e.g. `#/data`) so navigation is bookmarkable and the browser back/forward
 * buttons work, with no routing dependency. Unknown/empty hashes fall back to the
 * default section. Returns the active id + a typed navigate function.
 */
function parseHash(): SectionId {
  if (typeof window === 'undefined') {
    return DEFAULT_SECTION;
  }
  const raw = window.location.hash.replace(/^#\/?/, '').trim();
  return (SECTION_IDS as readonly string[]).includes(raw) ? (raw as SectionId) : DEFAULT_SECTION;
}

export function useHashRoute(): [SectionId, (id: SectionId) => void] {
  const [active, setActive] = useState<SectionId>(parseHash);

  useEffect(() => {
    const onHash = (): void => setActive(parseHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const navigate = useCallback((id: SectionId): void => {
    window.location.hash = `/${id}`;
  }, []);

  return [active, navigate];
}
