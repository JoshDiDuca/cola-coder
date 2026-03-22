"""Tests for repository context scanning (inference/repo_context.py).

Covers:
- parse_imports(): ES imports, default, namespace, require, type, multi-name, empty
- extract_exports(): interface, type alias, function, const
- jaccard_similarity(): identical, disjoint, overlapping, empty
- find_similar_files(): top-K, exclusions, empty corpus
- build_file_tree(): correct traversal, skips noise dirs, respects max_depth
- RepoScanner: scan(), get_context_for_file(), get_repo_summary()
- ContextAwareGenerator: context prepended to prompt, rescan()
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from cola_coder.inference.repo_context import (
    RepoScanner,
    build_file_tree,
    extract_exports,
    find_similar_files,
    jaccard_similarity,
    parse_imports,
)


# ── parse_imports ─────────────────────────────────────────────────────────────


class TestParseImports:
    def test_named_import(self):
        code = "import { User } from './types/user'"
        refs = parse_imports(code)
        assert len(refs) == 1
        ref = refs[0]
        assert ref.names == ["User"]
        assert ref.source == "./types/user"
        assert ref.is_relative is True

    def test_default_import(self):
        code = "import React from 'react'"
        refs = parse_imports(code)
        assert len(refs) == 1
        ref = refs[0]
        assert ref.names == ["React"]
        assert ref.source == "react"
        assert ref.is_relative is False

    def test_namespace_import(self):
        code = "import * as path from 'path'"
        refs = parse_imports(code)
        assert len(refs) == 1
        ref = refs[0]
        assert ref.names == ["path"]
        assert ref.source == "path"
        assert ref.is_relative is False

    def test_require(self):
        code = "const express = require('express')"
        refs = parse_imports(code)
        assert len(refs) == 1
        ref = refs[0]
        assert ref.source == "express"

    def test_type_import(self):
        code = "import type { Config } from '../config'"
        refs = parse_imports(code)
        assert len(refs) == 1
        ref = refs[0]
        assert "Config" in ref.names
        assert ref.source == "../config"
        assert ref.is_relative is True

    def test_multi_name_import(self):
        code = "import { a, b, c } from './utils'"
        refs = parse_imports(code)
        assert len(refs) == 1
        ref = refs[0]
        assert set(ref.names) == {"a", "b", "c"}
        assert ref.source == "./utils"

    def test_empty_string_returns_empty_list(self):
        assert parse_imports("") == []

    def test_comment_lines_skipped(self):
        code = "// import { Foo } from './foo'\nimport { Bar } from './bar'"
        refs = parse_imports(code)
        assert len(refs) == 1
        assert refs[0].names == ["Bar"]

    def test_multiple_imports_distinct_sources(self):
        code = (
            "import { useState } from 'react'\n"
            "import { User } from './types'\n"
        )
        refs = parse_imports(code)
        sources = {r.source for r in refs}
        assert "react" in sources
        assert "./types" in sources

    def test_relative_dot_dot_path(self):
        code = "import { helper } from '../../utils/format'"
        refs = parse_imports(code)
        assert refs[0].is_relative is True

    def test_deduplication(self):
        code = (
            "import { Foo } from './foo'\n"
            "import { Foo } from './foo'\n"
        )
        refs = parse_imports(code)
        assert len(refs) == 1

    def test_double_quotes(self):
        code = 'import { Bar } from "./bar"'
        refs = parse_imports(code)
        assert len(refs) == 1
        assert refs[0].source == "./bar"


# ── extract_exports ───────────────────────────────────────────────────────────


class TestExtractExports:
    def test_interface_export(self):
        code = "export interface User { id: string; name: string; }"
        result = extract_exports(code)
        assert "interface User" in result

    def test_type_alias_export(self):
        code = "export type Role = 'admin' | 'user';"
        result = extract_exports(code)
        assert "type Role" in result

    def test_function_export(self):
        code = "export function hello(name: string): string { return `Hello ${name}`; }"
        result = extract_exports(code)
        assert "function hello" in result

    def test_const_export(self):
        code = "export const MAX = 100;"
        result = extract_exports(code)
        assert "const MAX" in result

    def test_no_exports_returns_empty(self):
        code = "const x = 1;\nfunction helper() {}"
        result = extract_exports(code)
        assert result == ""

    def test_function_signature_gets_ellipsis_body(self):
        """Function exports should get ' { ... }' appended (not full body)."""
        code = "export function greet(name: string): string { return name; }"
        result = extract_exports(code)
        assert "{ ... }" in result

    def test_multiple_exports(self):
        code = (
            "export interface User { id: string; }\n"
            "export type Role = 'admin' | 'user';\n"
            "export const MAX_USERS = 100;\n"
        )
        result = extract_exports(code)
        assert "interface User" in result
        assert "type Role" in result
        assert "const MAX_USERS" in result

    def test_const_trimmed_to_one_line(self):
        """Const exports are trimmed to a single line ending with ';'."""
        code = "export const config = { foo: 1, bar: 2 };"
        result = extract_exports(code)
        lines = [line for line in result.splitlines() if "const config" in line]
        assert len(lines) == 1
        assert lines[0].endswith(";")

    def test_async_function_export(self):
        code = "export async function fetchUser(id: string): Promise<User> { return {} as User; }"
        result = extract_exports(code)
        assert "function fetchUser" in result


# ── jaccard_similarity ────────────────────────────────────────────────────────


class TestJaccardSimilarity:
    def test_identical_sets(self):
        tokens = {1, 2, 3, 4, 5}
        assert jaccard_similarity(tokens, tokens) == 1.0

    def test_disjoint_sets(self):
        a = {1, 2, 3}
        b = {4, 5, 6}
        assert jaccard_similarity(a, b) == 0.0

    def test_overlapping_sets(self):
        a = {1, 2, 3}
        b = {2, 3, 4}
        # intersection=2, union=4  → 0.5
        result = jaccard_similarity(a, b)
        assert abs(result - 0.5) < 1e-9

    def test_empty_sets_return_zero(self):
        assert jaccard_similarity(set(), set()) == 0.0

    def test_one_empty_set(self):
        assert jaccard_similarity({1, 2}, set()) == 0.0
        assert jaccard_similarity(set(), {1, 2}) == 0.0

    def test_subset(self):
        a = {1, 2}
        b = {1, 2, 3, 4}
        # intersection=2, union=4  → 0.5
        result = jaccard_similarity(a, b)
        assert abs(result - 0.5) < 1e-9

    def test_range_zero_to_one(self):
        a = {1, 2, 3, 4}
        b = {3, 4, 5, 6}
        result = jaccard_similarity(a, b)
        assert 0.0 <= result <= 1.0


# ── find_similar_files ────────────────────────────────────────────────────────


class TestFindSimilarFiles:
    def test_returns_top_k_sorted(self):
        target = {1, 2, 3, 4}
        corpus = {
            "a.ts": {1, 2, 3, 4},      # identical → 1.0
            "b.ts": {1, 2, 5, 6},      # 2/6 → 0.333
            "c.ts": {1, 2, 3, 7, 8},   # 3/6 → 0.5
        }
        results = find_similar_files(target, corpus, top_k=2)
        assert len(results) == 2
        # Highest similarity first
        assert results[0][0] == "a.ts"
        assert results[1][0] == "c.ts"

    def test_excludes_specified_paths(self):
        target = {1, 2, 3}
        corpus = {
            "self.ts": {1, 2, 3},
            "other.ts": {1, 2, 4},
        }
        results = find_similar_files(target, corpus, top_k=3, exclude={"self.ts"})
        paths = [r[0] for r in results]
        assert "self.ts" not in paths
        assert "other.ts" in paths

    def test_empty_corpus_returns_empty(self):
        assert find_similar_files({1, 2, 3}, {}) == []

    def test_empty_target_returns_empty(self):
        corpus = {"a.ts": {1, 2, 3}}
        assert find_similar_files(set(), corpus) == []

    def test_top_k_cap(self):
        target = {1, 2, 3}
        corpus = {f"file{i}.ts": {1, 2, i + 10} for i in range(10)}
        results = find_similar_files(target, corpus, top_k=3)
        assert len(results) <= 3

    def test_scores_in_result_tuples(self):
        target = {1, 2, 3}
        corpus = {"a.ts": {1, 2, 3, 4, 5}}
        results = find_similar_files(target, corpus, top_k=1)
        assert len(results) == 1
        path, score = results[0]
        assert path == "a.ts"
        assert 0.0 < score <= 1.0

    def test_zero_similarity_files_excluded(self):
        target = {1, 2, 3}
        corpus = {"no_overlap.ts": {4, 5, 6}}
        results = find_similar_files(target, corpus, top_k=3)
        assert results == []


# ── build_file_tree ───────────────────────────────────────────────────────────


class TestBuildFileTree:
    def test_basic_structure(self, tmp_path):
        (tmp_path / "index.ts").write_text("export {}")
        (tmp_path / "utils.ts").write_text("export const x = 1")
        tree = build_file_tree(tmp_path)
        assert "index.ts" in tree
        assert "utils.ts" in tree

    def test_nested_files(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "app.ts").write_text("const x = 1")
        tree = build_file_tree(tmp_path)
        assert "src/app.ts" in tree

    def test_skips_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "react.js").write_text("module.exports = {}")
        tree = build_file_tree(tmp_path)
        for entry in tree:
            assert "node_modules" not in entry

    def test_skips_dot_git(self, tmp_path):
        git = tmp_path / ".git"
        git.mkdir()
        (git / "HEAD").write_text("ref: refs/heads/main")
        tree = build_file_tree(tmp_path)
        for entry in tree:
            assert ".git" not in entry

    def test_skips_dist_build(self, tmp_path):
        for skip_dir in ("dist", "build"):
            d = tmp_path / skip_dir
            d.mkdir()
            (d / "bundle.js").write_text("!function(){}")
        tree = build_file_tree(tmp_path)
        for entry in tree:
            assert not entry.startswith("dist/")
            assert not entry.startswith("build/")

    def test_respects_max_depth_zero(self, tmp_path):
        """max_depth=0 should only return root-level files."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "deep.ts").write_text("x")
        (tmp_path / "root.ts").write_text("y")
        tree = build_file_tree(tmp_path, max_depth=0)
        assert "root.ts" in tree
        for entry in tree:
            assert "/" not in entry

    def test_respects_max_depth(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "file.ts").write_text("x")
        (tmp_path / "a" / "b").mkdir(exist_ok=True)
        (tmp_path / "a" / "b" / "shallow.ts").write_text("y")
        tree = build_file_tree(tmp_path, max_depth=2)
        assert "a/b/shallow.ts" in tree
        # depth=3 file (a/b/c/file.ts) should be excluded at max_depth=2
        assert "a/b/c/file.ts" not in tree

    def test_posix_paths(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "hooks" ).mkdir()
        (src / "hooks" / "useUser.ts").write_text("x")
        tree = build_file_tree(tmp_path)
        for entry in tree:
            assert "\\" not in entry  # always POSIX slashes

    def test_hidden_files_skipped(self, tmp_path):
        (tmp_path / ".env").write_text("SECRET=x")
        (tmp_path / "index.ts").write_text("x")
        tree = build_file_tree(tmp_path)
        for entry in tree:
            assert not entry.startswith(".")

    def test_empty_directory(self, tmp_path):
        tree = build_file_tree(tmp_path)
        assert tree == []


# ── RepoScanner ───────────────────────────────────────────────────────────────


def _make_mini_project(root: Path) -> None:
    """Create a minimal TS project in root for scanner tests."""
    pkg = {
        "name": "my-app",
        "version": "1.0.0",
        "dependencies": {
            "react": "^18.2.0",
            "next": "^14.2.3",
        },
        "devDependencies": {
            "typescript": "^5.3.0",
        },
    }
    (root / "package.json").write_text(json.dumps(pkg))

    tsconfig = {"compilerOptions": {"strict": True, "baseUrl": "."}}
    (root / "tsconfig.json").write_text(json.dumps(tsconfig))

    src = root / "src"
    (src / "types").mkdir(parents=True)
    (src / "hooks").mkdir(parents=True)

    (src / "types" / "user.ts").write_text(
        "export interface User { id: string; name: string; email: string; }\n"
        "export type UserRole = 'admin' | 'user';\n"
    )

    (src / "hooks" / "useUser.ts").write_text(
        "import { User } from '../types/user';\n"
        "import { useState } from 'react';\n"
        "\n"
        "export function useUser(id: string): User | null { return null; }\n"
    )


class TestRepoScannerScan:
    def test_scan_returns_repo_context(self, tmp_path):
        _make_mini_project(tmp_path)
        scanner = RepoScanner(tmp_path)
        ctx = scanner.scan()
        assert ctx is not None
        assert ctx.root == tmp_path.resolve()

    def test_scan_finds_ts_files(self, tmp_path):
        _make_mini_project(tmp_path)
        scanner = RepoScanner(tmp_path)
        ctx = scanner.scan()
        assert any("user.ts" in f for f in ctx.file_tree)
        assert any("useUser.ts" in f for f in ctx.file_tree)

    def test_scan_reads_framework_versions(self, tmp_path):
        _make_mini_project(tmp_path)
        scanner = RepoScanner(tmp_path)
        ctx = scanner.scan()
        assert "react" in ctx.framework_versions
        assert ctx.framework_versions["react"] == "18.2.0"
        assert "next" in ctx.framework_versions
        assert ctx.framework_versions["next"] == "14.2.3"

    def test_scan_reads_tsconfig(self, tmp_path):
        _make_mini_project(tmp_path)
        scanner = RepoScanner(tmp_path)
        ctx = scanner.scan()
        assert ctx.tsconfig is not None
        assert "compilerOptions" in ctx.tsconfig

    def test_scan_builds_import_graph(self, tmp_path):
        _make_mini_project(tmp_path)
        scanner = RepoScanner(tmp_path)
        ctx = scanner.scan()
        # At least one file should have imports
        all_imports = [v for v in ctx.import_graph.values() if v]
        assert len(all_imports) > 0

    def test_scan_missing_package_json(self, tmp_path):
        """Scanner should not crash when package.json is absent."""
        (tmp_path / "index.ts").write_text("export const x = 1")
        scanner = RepoScanner(tmp_path)
        ctx = scanner.scan()
        assert ctx.package_info == {}
        assert ctx.framework_versions == {}


class TestRepoScannerGetContextForFile:
    def test_context_includes_repo_tags(self, tmp_path):
        _make_mini_project(tmp_path)
        scanner = RepoScanner(tmp_path)
        scanner.scan()
        result = scanner.get_context_for_file("src/hooks/useUser.ts")
        assert "<|repo|>" in result
        assert "<|/repo|>" in result

    def test_context_includes_user_type(self, tmp_path):
        """Context for useUser.ts should include User from the relative import."""
        _make_mini_project(tmp_path)
        scanner = RepoScanner(tmp_path)
        scanner.scan()
        result = scanner.get_context_for_file("src/hooks/useUser.ts")
        # The User interface from ../types/user should appear
        assert "User" in result

    def test_context_includes_project_name(self, tmp_path):
        _make_mini_project(tmp_path)
        scanner = RepoScanner(tmp_path)
        scanner.scan()
        result = scanner.get_context_for_file("src/hooks/useUser.ts")
        assert "my-app" in result

    def test_context_for_nonexistent_file(self, tmp_path):
        """get_context_for_file() on a path with no imports returns a valid block."""
        _make_mini_project(tmp_path)
        scanner = RepoScanner(tmp_path)
        scanner.scan()
        result = scanner.get_context_for_file("src/new_file.ts")
        # Still wraps in repo tags
        assert "<|repo|>" in result

    def test_auto_scans_if_not_run(self, tmp_path):
        """Calling get_context_for_file() before scan() should trigger scan automatically."""
        _make_mini_project(tmp_path)
        scanner = RepoScanner(tmp_path)
        # Don't call scan() explicitly
        result = scanner.get_context_for_file("src/hooks/useUser.ts")
        assert "<|repo|>" in result

    def test_context_respects_token_budget(self, tmp_path):
        """Context block should stay within ~4x char budget of max_tokens."""
        _make_mini_project(tmp_path)
        scanner = RepoScanner(tmp_path)
        scanner.scan()
        max_tokens = 100
        result = scanner.get_context_for_file("src/hooks/useUser.ts", max_tokens=max_tokens)
        # Allow generous overhead for tags and header
        assert len(result) <= max_tokens * 4 + 200


class TestRepoScannerGetRepoSummary:
    def test_summary_before_scan(self, tmp_path):
        scanner = RepoScanner(tmp_path)
        summary = scanner.get_repo_summary()
        assert "not scanned" in summary.lower()

    def test_summary_after_scan(self, tmp_path):
        _make_mini_project(tmp_path)
        scanner = RepoScanner(tmp_path)
        scanner.scan()
        summary = scanner.get_repo_summary()
        # Summary line starts with "Repository: <dir-name>"
        assert summary.startswith("Repository:")

    def test_summary_includes_framework_versions(self, tmp_path):
        _make_mini_project(tmp_path)
        scanner = RepoScanner(tmp_path)
        scanner.scan()
        summary = scanner.get_repo_summary()
        assert "react" in summary.lower() or "next" in summary.lower()

    def test_summary_includes_file_count(self, tmp_path):
        _make_mini_project(tmp_path)
        scanner = RepoScanner(tmp_path)
        scanner.scan()
        summary = scanner.get_repo_summary()
        # Should mention "Files" somewhere
        assert "Files" in summary or "files" in summary

    def test_summary_mentions_tsconfig(self, tmp_path):
        _make_mini_project(tmp_path)
        scanner = RepoScanner(tmp_path)
        scanner.scan()
        summary = scanner.get_repo_summary()
        assert "tsconfig" in summary.lower()


# ── ContextAwareGenerator (mock) ──────────────────────────────────────────────


class TestContextAwareGenerator:
    """Tests that ContextAwareGenerator correctly wraps CodeGenerator with context."""

    def _make_mock_generator(self, return_value: str = "generated_code") -> MagicMock:
        """Return a MagicMock that quacks like CodeGenerator."""
        mock_gen = MagicMock()
        mock_gen.generate.return_value = return_value
        mock_gen.generate_stream.return_value = iter(["chunk1", "chunk2"])
        return mock_gen

    def test_context_prepended_to_prompt(self, tmp_path):
        """generate() should call underlying generator with context+prompt."""
        from cola_coder.inference.context_generator import ContextAwareGenerator

        _make_mini_project(tmp_path)
        mock_gen = self._make_mock_generator()

        ctx_gen = ContextAwareGenerator(mock_gen, tmp_path, eager_scan=True)
        ctx_gen.generate("src/hooks/useUser.ts", "// write a hook")

        # Verify the underlying generator was called
        mock_gen.generate.assert_called_once()
        # The first positional arg should contain the context block
        call_args = mock_gen.generate.call_args
        full_prompt = call_args[0][0]
        assert "<|repo|>" in full_prompt
        assert "// write a hook" in full_prompt
        # Context must come before the user prompt
        assert full_prompt.index("<|repo|>") < full_prompt.index("// write a hook")

    def test_generate_returns_underlying_result(self, tmp_path):
        from cola_coder.inference.context_generator import ContextAwareGenerator

        _make_mini_project(tmp_path)
        mock_gen = self._make_mock_generator("// generated output")

        ctx_gen = ContextAwareGenerator(mock_gen, tmp_path)
        result = ctx_gen.generate("src/hooks/useUser.ts", "prompt")
        assert result == "// generated output"

    def test_rescan_updates_context(self, tmp_path):
        """rescan() should call scan() again and update self.context."""
        from cola_coder.inference.context_generator import ContextAwareGenerator

        _make_mini_project(tmp_path)
        mock_gen = self._make_mock_generator()

        ctx_gen = ContextAwareGenerator(mock_gen, tmp_path)

        # Add a new file to the project
        (tmp_path / "src" / "new_component.ts").write_text(
            "export const NewComp = () => null;\n"
        )
        ctx_gen.rescan()
        new_context = ctx_gen.context

        # After rescan, the new file should appear in the file tree
        assert any("new_component.ts" in f for f in new_context.file_tree)

    def test_deferred_scan_triggers_on_first_generate(self, tmp_path):
        """eager_scan=False should defer scan until generate() is called."""
        from cola_coder.inference.context_generator import ContextAwareGenerator

        _make_mini_project(tmp_path)
        mock_gen = self._make_mock_generator()

        ctx_gen = ContextAwareGenerator(mock_gen, tmp_path, eager_scan=False)
        # Before generate(), context should be None
        assert ctx_gen.context is None

        ctx_gen.generate("src/hooks/useUser.ts", "prompt")
        # After generate(), context should be populated
        assert ctx_gen.context is not None

    def test_generate_stream_yields_chunks(self, tmp_path):
        """generate_stream() should yield all chunks from the underlying generator."""
        from cola_coder.inference.context_generator import ContextAwareGenerator

        _make_mini_project(tmp_path)
        mock_gen = self._make_mock_generator()
        mock_gen.generate_stream.return_value = iter(["chunk1", " chunk2"])

        ctx_gen = ContextAwareGenerator(mock_gen, tmp_path)
        chunks = list(ctx_gen.generate_stream("src/hooks/useUser.ts", "prompt"))
        assert chunks == ["chunk1", " chunk2"]

    def test_get_repo_summary_after_scan(self, tmp_path):
        from cola_coder.inference.context_generator import ContextAwareGenerator

        _make_mini_project(tmp_path)
        mock_gen = self._make_mock_generator()

        ctx_gen = ContextAwareGenerator(mock_gen, tmp_path)
        summary = ctx_gen.get_repo_summary()
        assert summary.startswith("Repository:")

    def test_kwargs_forwarded_to_generator(self, tmp_path):
        """Extra kwargs like temperature should be forwarded to CodeGenerator."""
        from cola_coder.inference.context_generator import ContextAwareGenerator

        _make_mini_project(tmp_path)
        mock_gen = self._make_mock_generator()

        ctx_gen = ContextAwareGenerator(mock_gen, tmp_path)
        ctx_gen.generate("src/hooks/useUser.ts", "prompt", temperature=0.7, top_k=40)

        call_kwargs = mock_gen.generate.call_args[1]
        assert call_kwargs.get("temperature") == 0.7
        assert call_kwargs.get("top_k") == 40
