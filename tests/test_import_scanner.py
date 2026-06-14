"""SEC-020: hallucinated-import / slopsquatting scanner.

Flags imports of packages outside a known-safe allowlist (stdlib + popular). No
execution. A flagged import is a REVIEW signal, not a hard block.
"""

from cola_coder.security.import_scanner import (
    extract_imports,
    has_unknown_imports,
    scan_unknown_imports,
)


class TestPythonImports:
    def test_stdlib_and_popular_are_known(self):
        code = "import os\nimport json\nimport requests\nfrom pandas import DataFrame"
        assert scan_unknown_imports(code, "python") == []

    def test_hallucinated_package_flagged(self):
        code = "import requests\nimport totally_fake_pkg_xyz"
        flagged = scan_unknown_imports(code, "python")
        assert flagged == ["totally_fake_pkg_xyz"]

    def test_submodule_uses_root_package(self):
        # `import os.path` / `from numpy.linalg import inv` -> roots os / numpy.
        code = "import os.path\nfrom numpy.linalg import inv"
        assert scan_unknown_imports(code, "python") == []

    def test_relative_import_is_local_not_flagged(self):
        code = "from . import helpers\nfrom ..utils import thing"
        assert scan_unknown_imports(code, "python") == []

    def test_regex_fallback_on_unparseable_code(self):
        # Partial generation that doesn't parse -> regex still finds the import.
        code = "import faketelemetry_sdk\ndef broken(:"
        assert "faketelemetry_sdk" in scan_unknown_imports(code, "python")

    def test_extract_imports_roots(self):
        assert extract_imports("import a.b.c\nfrom d import e", "python") == {"a", "d"}


class TestJsImports:
    def test_builtins_and_popular_known(self):
        code = "import fs from 'fs';\nimport React from 'react';\nconst ax = require('axios');"
        assert scan_unknown_imports(code, "typescript") == []

    def test_hallucinated_npm_flagged(self):
        code = "import { z } from 'fake-ui-toolkit-9000';"
        assert scan_unknown_imports(code, "typescript") == ["fake-ui-toolkit-9000"]

    def test_scoped_package_kept_whole(self):
        code = "import { Client } from '@prisma/client';\nimport x from '@evil/squat';"
        flagged = scan_unknown_imports(code, "typescript")
        assert "@prisma/client" not in flagged   # known
        assert "@evil/squat" in flagged           # unknown

    def test_relative_import_not_flagged(self):
        code = "import { helper } from './utils';\nimport x from '../lib/x';"
        assert scan_unknown_imports(code, "typescript") == []

    def test_subpath_uses_package_root(self):
        code = "import debounce from 'lodash/debounce';"
        assert scan_unknown_imports(code, "typescript") == []


class TestApi:
    def test_has_unknown_imports(self):
        assert has_unknown_imports("import made_up_lib", "python") is True
        assert has_unknown_imports("import os", "python") is False

    def test_empty_code(self):
        assert scan_unknown_imports("", "python") == []
        assert has_unknown_imports("", "typescript") is False


class TestBestOfNWiring:
    def test_unknown_imports_surfaced_in_candidate_details(self):
        from cola_coder.inference.best_of_n import _build_candidates
        verdicts = [(True, 1.0, {})]
        texts = ["import fake_hallucinated_pkg\nx = 1"]
        cands = _build_candidates(texts, verdicts, prompt="", lang="python")
        assert cands[0].details.get("unknown_imports") == ["fake_hallucinated_pkg"]

    def test_clean_imports_no_flag(self):
        from cola_coder.inference.best_of_n import _build_candidates
        cands = _build_candidates([("import os\nx=1")], [(True, 1.0, {})],
                                  prompt="", lang="python")
        assert "unknown_imports" not in cands[0].details
