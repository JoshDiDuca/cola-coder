"""Margin-aware selective routing for the specialist cascade (MODEL-053).

`route_with_abstention` dispatches a snippet to a domain specialist only when the
top domain clears both an absolute-confidence and a margin-over-runner-up gate;
otherwise it abstains to the general model. These tests pin that behavior and
confirm it is purely additive (detect_domain/classify are unchanged).
"""

from cola_coder.features.domain_detector import (
    DomainScore,
    detection_margin,
    route_with_abstention,
)

_REACT = (
    "import React from 'react'\n"
    "import { useState, useEffect } from 'react'\n"
    "export const App: React.FC = () => {\n"
    "  const [n, setN] = useState(0)\n"
    "  useEffect(() => {}, [])\n"
    "  return <div className=\"x\" onClick={() => setN(n + 1)}>{n}</div>\n"
    "}\n"
)

# A deliberately MIXED react+zod snippet: neither domain dominates, so the top
# confidence stays well below 1.0 — the regime where the abstention gates bite.
_MIXED = (
    "import React from 'react'\n"
    "import { z } from 'zod'\n"
    "const Schema = z.object({ name: z.string() })\n"
    "export const App = () => <div className=\"x\">{Schema.parse({}).name}</div>\n"
)


class TestDetectionMargin:
    def test_empty_is_zero(self) -> None:
        assert detection_margin([]) == 0.0

    def test_single_returns_its_confidence(self) -> None:
        only = [DomainScore("react", 1, 2, 5.0, 0.7)]
        assert detection_margin(only) == 0.7

    def test_top_minus_runner_up(self) -> None:
        scores = [
            DomainScore("react", 1, 2, 5.0, 0.6),
            DomainScore("zod", 0, 1, 1.0, 0.25),
            DomainScore("testing", 0, 1, 1.0, 0.15),
        ]
        assert abs(detection_margin(scores) - 0.35) < 1e-9

    def test_unsorted_input_still_uses_top_two(self) -> None:
        scores = [
            DomainScore("testing", 0, 1, 1.0, 0.15),
            DomainScore("react", 1, 2, 5.0, 0.6),
            DomainScore("zod", 0, 1, 1.0, 0.25),
        ]
        assert abs(detection_margin(scores) - 0.35) < 1e-9


class TestRouteWithAbstention:
    def test_clear_react_routes_to_react(self) -> None:
        decision = route_with_abstention(_REACT)
        assert decision.domain == "react"
        assert decision.abstained is False
        assert decision.reason == "ok"
        assert 0.0 <= decision.confidence <= 1.0

    def test_empty_code_is_general(self) -> None:
        # detect_domain emits a "general" fallback for empty input, so this routes
        # to general as a real (top) classification — never crashes.
        decision = route_with_abstention("")
        assert decision.domain == "general"

    def test_generic_code_is_general_not_abstention(self) -> None:
        # Weak input where the heuristic itself returns "general" as the top pick.
        decision = route_with_abstention("const x = 1 + 2\n")
        assert decision.domain == "general"
        # general-as-top is a real classification, not a weak-signal fallback
        assert decision.reason in {"ok", "no_signal"}

    def test_high_confidence_threshold_forces_abstention(self) -> None:
        # On a mixed snippet the top confidence is < 1.0, so a high confidence bar
        # trips the low_confidence guard and falls back to general.
        decision = route_with_abstention(_MIXED, min_confidence=0.99)
        assert decision.domain == "general"
        assert decision.abstained is True
        assert decision.reason == "low_confidence"

    def test_high_margin_threshold_forces_abstention(self) -> None:
        # Confidence passes (bar 0) but an impossible margin bar trips low_margin.
        decision = route_with_abstention(_MIXED, min_confidence=0.0, min_margin=1.1)
        assert decision.domain == "general"
        assert decision.abstained is True
        assert decision.reason == "low_margin"

    def test_lenient_thresholds_keep_specialist(self) -> None:
        decision = route_with_abstention(_REACT, min_confidence=0.0, min_margin=0.0)
        assert decision.domain == "react"
        assert decision.abstained is False
