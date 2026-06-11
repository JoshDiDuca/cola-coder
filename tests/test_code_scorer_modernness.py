"""DATA-022: the JS/TS modernness scorer must not penalize `!==` as loose `==`.

`_score_modernness_js_ts` detected deprecated loose equality with
`re.findall(r'(?<!=)==(?!=)', code)`, which ALSO matched the `==` inside `!==`
(strict inequality — a MODERN idiom). So clean TypeScript using `!==` was scored
as if it used deprecated `==`, deflating its modernness sub-score and therefore
its training weight (CodeScorer feeds score_data.py's .weights.npy). Fixed by
excluding `!` from the lookbehind: `(?<![=!])==(?!=)`.
"""

from cola_coder.features.code_scorer import CodeScorer


def _modernness(code: str) -> float:
    return CodeScorer()._score_modernness_js_ts(code)


class TestStrictInequalityNotPenalized:
    def test_strict_inequality_scores_at_least_as_high_as_loose(self):
        # Identical code differing only in !== vs ==. The modern (!==) version
        # must not score LOWER on modernness than the loose (==) version.
        strict = (
            "export function f(a: number, b: number): boolean {\n"
            "  if (a !== b && b !== 0) { return true; }\n"
            "  return false;\n"
            "}\n"
        )
        loose_eq = (
            "export function f(a: number, b: number): boolean {\n"
            "  if (a == b || b == 0) { return true; }\n"
            "  return false;\n"
            "}\n"
        )
        assert _modernness(strict) >= _modernness(loose_eq)

    def test_strict_inequality_only_is_not_treated_as_deprecated(self):
        # A file whose ONLY equality-ish operator is !== should not be dragged
        # down by the loose-equality penalty.
        only_strict = "const ok = (x: number) => x !== 0 && x !== 1 && x !== 2;\n"
        only_loose = "var ok = function (x) { return x == 0 || x == 1 || x == 2; };\n"
        assert _modernness(only_strict) > _modernness(only_loose)

    def test_real_loose_equality_still_penalized(self):
        # The fix must not stop detecting genuine `==`. Compare loose vs strict
        # equality (===) on otherwise identical modern code.
        with_strict_eq = "const f = (a: number, b: number) => a === b;\n"
        with_loose_eq = "const f = (a: number, b: number) => a == b;\n"
        assert _modernness(with_strict_eq) > _modernness(with_loose_eq)
