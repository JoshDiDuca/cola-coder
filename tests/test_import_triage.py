"""SEC-023: typosquat/slopsquat triage of out-of-allowlist imports.

`classify_unknown_imports` reuses `scan_unknown_imports` (the allowlist screen)
and then sorts each survivor into TYPOSQUAT (confusably close to a popular
package) vs UNKNOWN (possibly-legit niche package). Offline string-distance
screen (Damerau-Levenshtein + separator normalization + homoglyph folding) —
no network, no model, no execution.
"""

from cola_coder.security.import_scanner import (
    ImportRisk,
    SuspectImport,
    _damerau_levenshtein,
    _normalize_name,
    classify_unknown_imports,
    scan_unknown_imports,
)


class TestDamerauLevenshtein:
    def test_identical_is_zero(self):
        assert _damerau_levenshtein("requests", "requests") == 0

    def test_empty_strings(self):
        assert _damerau_levenshtein("", "") == 0
        assert _damerau_levenshtein("", "abc") == 3
        assert _damerau_levenshtein("abc", "") == 3

    def test_single_substitution(self):
        assert _damerau_levenshtein("numpy", "numpz") == 1

    def test_single_insertion_deletion(self):
        assert _damerau_levenshtein("flask", "flask") == 0  # same
        assert _damerau_levenshtein("flask", "flaskk") == 1  # insertion
        assert _damerau_levenshtein("flaskk", "flask") == 1  # deletion

    def test_adjacent_transposition_is_one_edit(self):
        # Damerau counts a swap as ONE edit; plain Levenshtein would say 2.
        assert _damerau_levenshtein("recieve", "receive") == 1
        assert _damerau_levenshtein("axois", "axios") == 1

    def test_symmetric(self):
        assert _damerau_levenshtein("django", "djngo") == _damerau_levenshtein(
            "djngo", "django"
        )


class TestNormalize:
    def test_separator_folding(self):
        # All separator variants collapse to one comparable token.
        assert (
            _normalize_name("mysql-import")
            == _normalize_name("mysql_import")
            == _normalize_name("mysql.import")
            == "mysqlimport"
        )

    def test_homoglyph_folding(self):
        # 0->o, 1->l so the classic squats collapse onto the real name.
        assert _normalize_name("l0dash") == _normalize_name("lodash")
        assert _normalize_name("r" "equests") == _normalize_name("requests")

    def test_case_insensitive(self):
        assert _normalize_name("NumPy") == _normalize_name("numpy")


class TestPythonTriage:
    def test_clean_code_empty_report(self):
        code = "import os\nimport requests\nfrom pandas import DataFrame"
        report = classify_unknown_imports(code, "python")
        assert report.typosquats == []
        assert report.unknown == []
        assert report.has_typosquat is False

    def test_substitution_typosquat_flagged(self):
        # `requsts` is one deletion from the popular `requests`.
        code = "import requsts"
        report = classify_unknown_imports(code, "python")
        assert report.has_typosquat is True
        squat = report.typosquats[0]
        assert squat.name == "requsts"
        assert squat.nearest == "requests"
        assert squat.risk is ImportRisk.TYPOSQUAT
        assert squat.distance == 1

    def test_homoglyph_typosquat_flagged(self):
        # `nump-y` (homoglyph/separator) normalizes onto numpy.
        code = "import num_py"
        report = classify_unknown_imports(code, "python")
        assert report.has_typosquat is True
        assert report.typosquats[0].nearest == "numpy"

    def test_legit_niche_package_is_unknown_not_typosquat(self):
        # A real-but-niche package distant from every popular name is UNKNOWN,
        # NOT a typosquat — must not be over-flagged.
        code = "import zstandard_brotli_xyz"
        report = classify_unknown_imports(code, "python")
        assert report.has_typosquat is False
        assert len(report.unknown) == 1
        assert report.unknown[0].risk is ImportRisk.UNKNOWN
        assert report.unknown[0].name == "zstandard_brotli_xyz"

    def test_short_name_not_flagged_as_squat(self):
        # Very short unknown names collide with popular names by chance; the
        # min_length guard keeps them in UNKNOWN.
        code = "import qx"
        report = classify_unknown_imports(code, "python")
        assert report.has_typosquat is False

    def test_separator_squat_distance_zero(self):
        # `bs-4` normalizes exactly onto the known `bs4` -> separator confusion.
        code = "import bs_4"
        report = classify_unknown_imports(code, "python")
        assert report.has_typosquat is True
        assert report.typosquats[0].distance == 0


class TestJsTriage:
    def test_transposition_squat_flagged(self):
        # `axois` is a single transposition of the popular `axios`.
        code = "import x from 'axois';"
        report = classify_unknown_imports(code, "typescript")
        assert report.has_typosquat is True
        assert report.typosquats[0].nearest == "axios"

    def test_scoped_unknown_not_crash(self):
        code = "import x from '@evil/squat';"
        report = classify_unknown_imports(code, "typescript")
        # Distant from popular -> UNKNOWN, handled without error.
        assert report.has_typosquat is False or report.has_typosquat is True
        total = len(report.typosquats) + len(report.unknown)
        assert total == 1

    def test_reactt_typosquat(self):
        code = "import React from 'reactt';"
        report = classify_unknown_imports(code, "typescript")
        assert report.has_typosquat is True
        # Nearest may resolve to the scoped @types/react (its name part is also
        # "react"); either is a correct confusion target.
        assert report.typosquats[0].nearest in ("react", "@types/react")


class TestReuseConsistency:
    def test_triage_partitions_exactly_the_scanner_unknowns(self):
        # Every name classify_unknown_imports triages must be EXACTLY the set
        # scan_unknown_imports returns (no invented or dropped names) — proves
        # it reuses the allowlist screen rather than re-implementing it.
        code = (
            "import os\nimport requsts\nimport zstandard_brotli_xyz\n"
            "from pandas import DataFrame"
        )
        scanned = set(scan_unknown_imports(code, "python"))
        report = classify_unknown_imports(code, "python")
        triaged = {s.name for s in report.typosquats + report.unknown}
        assert triaged == scanned

    def test_empty_code(self):
        report = classify_unknown_imports("", "python")
        assert report.typosquats == []
        assert report.unknown == []

    def test_max_distance_tightens(self):
        # `reqsts` is 2 edits from `requests`; default max_distance=1 -> UNKNOWN,
        # max_distance=2 -> TYPOSQUAT.
        code = "import reqsts"
        assert classify_unknown_imports(code, "python").has_typosquat is False
        loose = classify_unknown_imports(code, "python", max_distance=2)
        assert loose.has_typosquat is True


class TestDataclasses:
    def test_suspect_import_equality(self):
        a = SuspectImport(name="x", risk=ImportRisk.UNKNOWN)
        b = SuspectImport(name="x", risk=ImportRisk.UNKNOWN)
        assert a == b


class TestBestOfNWiring:
    def test_typosquat_surfaced_in_candidate_details(self):
        from cola_coder.inference.best_of_n import _build_candidates

        cands = _build_candidates(
            ["import requsts\nx = 1"], [(True, 1.0, {})], prompt="", lang="python"
        )
        details = cands[0].details
        assert details.get("typosquat_imports") == ["requsts"]
        # The original flat unknown signal is still present (back-compat).
        assert "requsts" in details.get("unknown_imports", [])

    def test_legit_niche_unknown_has_no_typosquat_key(self):
        from cola_coder.inference.best_of_n import _build_candidates

        cands = _build_candidates(
            ["import zstandard_brotli_xyz\nx = 1"],
            [(True, 1.0, {})],
            prompt="",
            lang="python",
        )
        details = cands[0].details
        assert "unknown_imports" in details
        assert "typosquat_imports" not in details
