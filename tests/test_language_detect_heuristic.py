"""DATA-041: broaden is_typescript's content heuristic.

The original 6-marker set (: string/: number/: boolean/interface/<T>/as const)
missed very common TS that carries no surface type annotations — enums, type
aliases via access modifiers, optional members — so those files were mis-tagged
as not-TS and skipped from tsc scoring on this TS-primary repo. The expanded set
adds TS-ONLY constructs (no valid-JS false positives), keeping the ≥2 threshold.
"""

from cola_coder.data.scorers.language_detect import is_typescript, is_js_ts


class TestTypeScriptHeuristic:
    def test_enum_and_modifier(self):
        code = "enum Color { Red, Green }\nclass C {\n  readonly id: number;\n}\n"
        assert is_typescript(code) is True

    def test_interface_with_optional(self):
        code = "interface User {\n  name: string;\n  age?: number;\n}\n"
        assert is_typescript(code) is True

    def test_implements_and_void(self):
        code = "class S implements I {\n  run(): void {}\n}\n"
        assert is_typescript(code) is True

    def test_type_annotations_classic(self):
        # The original markers still work.
        assert is_typescript("function f(x: string): boolean { return !!x; }") is True

    def test_namespace_and_satisfies(self):
        code = "namespace N {}\nconst c = config satisfies Config;\n"
        assert is_typescript(code) is True

    def test_plain_js_not_typescript(self):
        # Const/arrow/require are JS — must NOT be flagged as TS.
        js = "const add = (a, b) => a + b;\nmodule.exports = { add };\n"
        assert is_typescript(js) is False

    def test_js_ternary_not_mistaken_for_optional(self):
        # A ternary `a ? b : c` has no "?:" token, so it isn't a TS optional hit.
        js = "const x = cond ? 1 : 2;\nconst y = other ? 3 : 4;\n"
        assert is_typescript(js) is False

    def test_single_marker_below_threshold(self):
        # One marker alone (could be coincidental) stays under the >=2 bar.
        assert is_typescript("const label: string = 'x';\nfoo();\nbar();\n") is False

    def test_js_ts_still_catches_both(self):
        assert is_js_ts("const x = 1;\nimport y from 'z';\n") is True

    def test_metadata_extension_still_authoritative(self):
        # Non-TS-looking content but a .ts file_path → TS (unchanged path).
        assert is_typescript("doStuff();\n", {"file_path": "a/b.ts"}) is True
