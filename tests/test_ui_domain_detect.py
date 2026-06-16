"""Tests for :func:`cola_coder.ui.domain_detect_view.detect_domain_view`.

The view is a MAIN-SAFE wrapper around the pure-regex
:func:`cola_coder.features.domain_detector.detect_domain` heuristic
(react/nextjs/graphql/prisma/zod/testing/general). No model, GPU, or network
is involved, so these tests run fast and deterministically.
"""

from __future__ import annotations

import pytest

from cola_coder.ui.domain_detect_view import detect_domain_view

# Sample snippets exercised across several test classes.
REACT_SNIPPET = (
    "import React from 'react'\n"
    "export const App = () => {\n"
    " const [n, setN] = useState(0)\n"
    ' return <div onClick={() => setN(n+1)} className="x"/>\n'
    "}"
)
ZOD_SNIPPET = "import { z } from 'zod'\nconst S = z.object({ name: z.string() })"
PRISMA_SNIPPET = (
    "import { PrismaClient } from '@prisma/client'\n"
    "const p = new PrismaClient()\n"
    "await p.user.findMany()"
)
TESTING_SNIPPET = (
    "import { describe, it, expect } from 'vitest'\n"
    "describe('x', () => { it('works', () => expect(1).toBe(1)) })"
)
GENERIC_SNIPPET = "const x = 1 + 2"

EXPECTED_SCORE_KEYS = {
    "domain",
    "import_matches",
    "keyword_matches",
    "raw_score",
    "confidence",
}


def _score_for(result: dict, domain: str) -> dict | None:
    """Return the score entry for ``domain`` in a result dict, or ``None``."""
    for entry in result["scores"]:
        if entry["domain"] == domain:
            return entry
    return None


class TestTopDomainClassification:
    """The detector picks the expected top domain for clear snippets."""

    def test_react_snippet_top_domain_is_react(self) -> None:
        """A React component with hooks + JSX classifies as react."""
        result = detect_domain_view(REACT_SNIPPET)
        assert "error" not in result
        assert result["top_domain"] == "react"
        react = _score_for(result, "react")
        assert react is not None
        assert react["confidence"] > 0
        assert react["import_matches"] >= 1

    def test_zod_snippet_top_domain_is_zod(self) -> None:
        """A Zod schema snippet classifies as zod."""
        result = detect_domain_view(ZOD_SNIPPET)
        assert result["top_domain"] == "zod"

    def test_prisma_snippet_top_domain_is_prisma(self) -> None:
        """A Prisma client snippet classifies as prisma."""
        result = detect_domain_view(PRISMA_SNIPPET)
        assert result["top_domain"] == "prisma"

    def test_testing_snippet_top_domain_is_testing(self) -> None:
        """A vitest snippet classifies as testing."""
        result = detect_domain_view(TESTING_SNIPPET)
        assert result["top_domain"] == "testing"


class TestFilenameInfluence:
    """The ``filename`` argument feeds into keyword matching."""

    def test_test_filename_ranks_testing_higher(self) -> None:
        """A `.test.tsx` filename surfaces the testing domain.

        The heuristic adds keyword weight for filename matches (e.g.
        ``\\.test\\.(ts|tsx|js|jsx)$``). The assertion stays lenient: we only
        require a valid result whose scores include the testing domain, and
        that the testing keyword count is no lower than without a filename.
        """
        plain_code = "const handler = () => doThing()"
        with_filename = detect_domain_view(plain_code, "Button.test.tsx")
        without_filename = detect_domain_view(plain_code)

        assert "error" not in with_filename
        assert isinstance(with_filename["top_domain"], str)
        assert isinstance(with_filename["scores"], list)

        testing_with = _score_for(with_filename, "testing")
        testing_without = _score_for(without_filename, "testing")
        assert testing_with is not None
        assert testing_without is not None
        # Filename match should not reduce testing's keyword evidence.
        assert testing_with["keyword_matches"] >= testing_without["keyword_matches"]


class TestGenericAndEmpty:
    """Generic, empty, and whitespace inputs."""

    def test_generic_code_returns_valid_shape(self) -> None:
        """Unmatched code yields a valid dict; top is general or low-confidence."""
        result = detect_domain_view(GENERIC_SNIPPET)
        assert "error" not in result
        assert isinstance(result["top_domain"], str)
        assert isinstance(result["scores"], list)
        assert len(result["scores"]) > 0

    def test_empty_string_returns_error(self) -> None:
        """An empty code string returns an error dict."""
        result = detect_domain_view("")
        assert "error" in result
        assert isinstance(result["error"], str)
        assert "top_domain" not in result

    @pytest.mark.parametrize("blank", ["   ", "\n", "\t\n  ", "\r\n"])
    def test_whitespace_only_returns_error(self, blank: str) -> None:
        """Whitespace-only code returns an error dict."""
        result = detect_domain_view(blank)
        assert "error" in result
        assert isinstance(result["error"], str)


class TestSchemaIntegrity:
    """Every score entry has the correct keys, types, and ordering."""

    @pytest.mark.parametrize(
        "code",
        [REACT_SNIPPET, ZOD_SNIPPET, PRISMA_SNIPPET, TESTING_SNIPPET, GENERIC_SNIPPET],
    )
    def test_score_entries_have_correct_keys_and_types(self, code: str) -> None:
        """Each scores entry matches the documented schema exactly."""
        result = detect_domain_view(code)
        assert "error" not in result
        for entry in result["scores"]:
            assert set(entry.keys()) == EXPECTED_SCORE_KEYS
            assert isinstance(entry["domain"], str)
            assert isinstance(entry["import_matches"], int)
            assert isinstance(entry["keyword_matches"], int)
            assert isinstance(entry["raw_score"], float)
            assert isinstance(entry["confidence"], float)
            assert 0.0 <= entry["confidence"] <= 1.0

    @pytest.mark.parametrize(
        "code",
        [REACT_SNIPPET, ZOD_SNIPPET, PRISMA_SNIPPET, TESTING_SNIPPET, GENERIC_SNIPPET],
    )
    def test_scores_sorted_by_confidence_descending(self, code: str) -> None:
        """Scores are returned sorted by confidence, highest first."""
        result = detect_domain_view(code)
        confidences = [entry["confidence"] for entry in result["scores"]]
        assert confidences == sorted(confidences, reverse=True)

    def test_top_domain_matches_first_score_entry(self) -> None:
        """``top_domain`` equals the domain of the first (highest) score."""
        result = detect_domain_view(REACT_SNIPPET)
        assert result["top_domain"] == result["scores"][0]["domain"]
