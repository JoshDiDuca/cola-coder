"""Tests for the IDEA-017 risk–coverage curve over the specialist router.

Sweeps the confidence gate of ``route_with_abstention`` across a labelled corpus and
checks the selective-prediction properties: one point per grid value, monotonically
non-increasing coverage as the gate tightens, bounded accuracy, and a sensible
``best_operating_point`` pick.
"""

from cola_coder.features.domain_detector import (
    CoveragePoint,
    best_operating_point,
    risk_coverage_curve,
)

# A small labelled corpus: clear specialist snippets + a couple of ambiguous/generic ones.
REACT_SNIPPET = """
import React, { useState, useEffect } from 'react';

export const Counter: React.FC = () => {
  const [count, setCount] = useState(0);
  useEffect(() => { console.log(count); }, [count]);
  return <button onClick={() => setCount(count + 1)} className="btn">{count}</button>;
};
"""

ZOD_SNIPPET = """
import { z } from 'zod';

export const UserSchema = z.object({
  id: z.string(),
  age: z.number(),
  tags: z.array(z.string()),
});
export type User = z.infer<typeof UserSchema>;
const parsed = UserSchema.parse(input);
"""

PRISMA_SNIPPET = """
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();
async function main() {
  const users = await prisma.user.findMany();
  await prisma.post.create({ data: { title: 'hi' } });
}
"""

TESTING_SNIPPET = """
import { describe, it, expect, beforeEach } from 'vitest';

describe('adder', () => {
  beforeEach(() => {});
  it('adds', () => {
    expect(1 + 1).toBe(2);
  });
});
"""

GENERIC_SNIPPET_1 = """
export function add(a: number, b: number): number {
  return a + b;
}
"""

GENERIC_SNIPPET_2 = """
const x = 10;
let y = x * 2;
console.log(y);
"""

LABELED_CORPUS: list[tuple[str, str]] = [
    (REACT_SNIPPET, "react"),
    (ZOD_SNIPPET, "zod"),
    (PRISMA_SNIPPET, "prisma"),
    (TESTING_SNIPPET, "testing"),
    (GENERIC_SNIPPET_1, "general"),
    (GENERIC_SNIPPET_2, "general"),
]

DEFAULT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_curve_has_one_point_per_grid_value_in_order() -> None:
    curve = risk_coverage_curve(LABELED_CORPUS)
    assert len(curve) == len(DEFAULT_GRID)
    assert [p.min_confidence for p in curve] == DEFAULT_GRID


def test_coverage_monotonically_non_increasing() -> None:
    curve = risk_coverage_curve(LABELED_CORPUS)
    coverages = [p.coverage for p in curve]
    # A stricter gate routes fewer samples → coverage never rises as min_confidence rises.
    # Tiny epsilon guards against float noise in the normalized-confidence heuristic.
    eps = 1e-9
    for earlier, later in zip(coverages, coverages[1:]):
        assert later <= earlier + eps


def test_coverage_highest_at_zero_gate() -> None:
    curve = risk_coverage_curve(LABELED_CORPUS)
    max_coverage = max(p.coverage for p in curve)
    assert curve[0].min_confidence == 0.0
    assert curve[0].coverage == max_coverage
    # At least one specialist snippet must be routed at the most permissive gate.
    assert curve[0].n_covered > 0


def test_accuracy_and_counts_bounded() -> None:
    curve = risk_coverage_curve(LABELED_CORPUS)
    for p in curve:
        assert 0.0 <= p.specialist_accuracy <= 1.0
        assert 0.0 <= p.coverage <= 1.0
        assert p.n_covered <= p.n_total
        assert p.n_total == len(LABELED_CORPUS)


def test_empty_corpus_returns_empty_list() -> None:
    assert risk_coverage_curve([]) == []


def test_custom_grid_respected() -> None:
    grid = [0.2, 0.5, 0.8]
    curve = risk_coverage_curve(LABELED_CORPUS, confidence_grid=grid)
    assert [p.min_confidence for p in curve] == grid


def test_best_operating_point_meets_floor() -> None:
    curve = risk_coverage_curve(LABELED_CORPUS)
    floor = 0.8
    best = best_operating_point(curve, min_specialist_accuracy=floor)
    if best is not None:
        assert isinstance(best, CoveragePoint)
        assert best.specialist_accuracy >= floor
        # It must be the max-coverage qualifying point.
        qualifying = [p for p in curve if p.specialist_accuracy >= floor]
        assert best.coverage == max(p.coverage for p in qualifying)


def test_best_operating_point_none_when_floor_unreachable() -> None:
    curve = risk_coverage_curve(LABELED_CORPUS)
    # An impossible accuracy floor cannot be met.
    assert best_operating_point(curve, min_specialist_accuracy=1.1) is None


def test_best_operating_point_empty_curve() -> None:
    assert best_operating_point([]) is None
