// Navigation model for the app shell. The 8 sections are routed PAGES (one shown
// at a time via the hash router), not stacked panels. This module is the single
// source of truth for the nav structure, page titles, and the default route.

export type SectionId =
  | 'overview'
  | 'run'
  | 'pipeline'
  | 'checkpoints'
  | 'data'
  | 'eval'
  | 'tokenizer'
  | 'system';

export type IconName =
  | 'dashboard'
  | 'play'
  | 'sliders'
  | 'box'
  | 'database'
  | 'chart'
  | 'type'
  | 'settings';

export interface NavItem {
  id: SectionId;
  label: string;
  icon: IconName;
  subtitle: string;
}

export interface NavGroup {
  group: string;
  items: NavItem[];
}

export const NAV: NavGroup[] = [
  {
    group: 'Monitor',
    items: [
      { id: 'overview', label: 'Dashboard', icon: 'dashboard', subtitle: 'Live training, GPU & system health' },
      { id: 'run', label: 'Run & Jobs', icon: 'play', subtitle: 'Launch actions and watch background jobs' },
      { id: 'eval', label: 'Evaluation', icon: 'chart', subtitle: 'HumanEval, benchmarks, safety & regressions' },
    ],
  },
  {
    group: 'Build',
    items: [
      { id: 'data', label: 'Data', icon: 'database', subtitle: 'Collect, prepare, score & inspect datasets' },
      { id: 'pipeline', label: 'Configs & Pipeline', icon: 'sliders', subtitle: 'Configs, VRAM/LR tools & pipeline runs' },
      { id: 'checkpoints', label: 'Checkpoints', icon: 'box', subtitle: 'Inspect, compare, average & export models' },
    ],
  },
  {
    group: 'Inspect',
    items: [
      { id: 'tokenizer', label: 'Tokenizer', icon: 'type', subtitle: 'Tokenizer info, health & vocab explorer' },
      { id: 'system', label: 'System & Tools', icon: 'settings', subtitle: 'Features, memory, docs, backlog & research' },
    ],
  },
];

const _ITEMS: NavItem[] = NAV.flatMap((g) => g.items);

export const SECTION_IDS: readonly SectionId[] = _ITEMS.map((i) => i.id);

export const SECTION_BY_ID: Record<SectionId, NavItem> = _ITEMS.reduce(
  (acc, item) => {
    acc[item.id] = item;
    return acc;
  },
  {} as Record<SectionId, NavItem>,
);

export const DEFAULT_SECTION: SectionId = 'overview';
