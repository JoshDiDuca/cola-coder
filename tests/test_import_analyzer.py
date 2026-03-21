"""Tests for import_analyzer.py."""

import pytest

from cola_coder.features.import_analyzer import ImportAnalyzer


@pytest.fixture
def analyzer():
    return ImportAnalyzer()


def test_feature_enabled():
    from cola_coder.features.import_analyzer import FEATURE_ENABLED, is_enabled

    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_python_simple_import(analyzer):
    code = "import os\nimport sys\n"
    report = analyzer.analyze(code, language="python")
    assert "os" in report.unique_modules
    assert "sys" in report.unique_modules
    assert report.total_import_lines == 2


def test_python_stdlib_classified(analyzer):
    code = "import json\nimport re\nimport numpy as np\n"
    report = analyzer.analyze(code, language="python")
    assert "json" in report.stdlib_modules
    assert "re" in report.stdlib_modules
    assert "numpy" in report.third_party_modules


def test_python_from_import(analyzer):
    code = "from pathlib import Path\nfrom typing import List, Optional\n"
    report = analyzer.analyze(code, language="python")
    assert "pathlib" in report.unique_modules
    assert "typing" in report.unique_modules


def test_python_relative_import(analyzer):
    code = "from . import utils\nfrom .models import User\n"
    report = analyzer.analyze(code, language="python")
    assert len(report.relative_imports) == 2


def test_ts_es_module_import(analyzer):
    code = 'import React from "react";\nimport { useState, useEffect } from "react";\n'
    report = analyzer.analyze(code, language="typescript")
    assert "react" in report.unique_modules
    assert report.total_import_lines == 2


def test_ts_third_party(analyzer):
    code = 'import axios from "axios";\nimport path from "path";\n'
    report = analyzer.analyze(code, language="ts")
    assert "axios" in report.third_party_modules
    assert "path" in report.stdlib_modules


def test_js_require(analyzer):
    code = 'const fs = require("fs");\nconst express = require("express");\n'
    report = analyzer.analyze(code, language="javascript")
    assert "fs" in report.unique_modules
    assert "express" in report.unique_modules


def test_summary_returns_string(analyzer):
    code = "import os\n"
    report = analyzer.analyze(code, "python")
    s = report.summary()
    assert isinstance(s, str)
    assert "language=python" in s


def test_empty_code(analyzer):
    report = analyzer.analyze("", "python")
    assert report.total_import_lines == 0
    assert report.unique_modules == []
