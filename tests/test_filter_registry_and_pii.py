"""DATA-020: register the orphaned FilterPlugins + fix PIIFilter over-matching.

Two coupled fixes:

1) Five FilterPlugins (pii, content, license, syntax, deduplication) conform to
   the FilterPlugin interface and are exported, but were NOT registered in the
   filter registry — the only mechanism the config-driven DataPipeline uses to
   instantiate filters by name. So the privacy (PII) and license filters were
   unreachable via config (orphaned). They are now registered.

2) PIIFilter's `password_assignment` pattern matched the bare `pass` keyword
   inside unrelated identifiers (`bypass`/`compass`/`surpass`), false-flagging
   valid code as PII and dropping the whole file (a DATA-015-class over-rejection
   that shrinks the corpus). A negative lookbehind now restricts the bare `pass`
   alternative to a real word boundary, while still catching `pass`/`db_pass`/
   `password`.
"""

from cola_coder.data.registry import list_filters, get_filter


class R:
    def __init__(self, content: str):
        self.content = content
        self.metadata: dict = {}


class TestFilterRegistration:
    def test_all_filters_registered(self):
        import cola_coder.data.filters  # noqa: F401 — fire @register_filter decorators

        names = set(list_filters())
        # The previously-orphaned filters must now be resolvable by name.
        for name in ("pii", "content", "license", "syntax", "deduplication"):
            assert name in names, f"{name} filter not registered"
        # Pre-existing registrations are untouched.
        for name in ("quality", "length", "quality_classifier"):
            assert name in names

    def test_registered_filters_are_constructible_and_setup_ok(self):
        # The DataPipeline contract: get_filter(name)() then .setup(cfg).
        import cola_coder.data.filters  # noqa: F401

        for name in ("pii", "content", "license", "syntax", "deduplication"):
            cls = get_filter(name)
            inst = cls()  # no-arg constructible
            assert inst.name()
            inst.setup({})  # must not raise on an empty config

    def test_pii_setup_applies_config(self):
        cls = get_filter("pii")
        inst = cls()
        inst.setup({"max_detections": 5, "check_false_positives": False})
        assert inst.max_detections == 5
        assert inst.check_false_positives is False


class TestPIIPasswordOverMatch:
    def test_identifiers_ending_in_pass_are_kept(self):
        f = get_filter("pii")()
        for code in (
            'bypass = "loginFlow123"',
            'compass = "northSouth01"',
            'const surpass = "thresholdValue"',
            'encompass = "wholeRegion42"',
        ):
            keep, reason = f.check(R(code))
            assert keep is True, f"{code!r} wrongly rejected as {reason}"

    def test_real_password_assignments_still_detected(self):
        f = get_filter("pii")()
        for code in (
            'password = "hunter2hunter"',
            'db_pass = "realSecretValue"',   # snake_case suffix, preceded by "_"
            'pass = "standaloneSecret9"',    # bare standalone keyword
            'PASSWORD = "MixedCaseSecret"',
        ):
            keep, reason = f.check(R(code))
            assert keep is False, f"{code!r} should be flagged as PII"
            assert "password_assignment" in reason

    def test_other_secret_patterns_unaffected(self):
        # Sanity: the fix is scoped to the password pattern; others still fire.
        f = get_filter("pii")()
        keep, reason = f.check(R('aws_key = "AKIA1234567890ABCDEF"'))
        assert keep is False and "aws_access_key" in reason
        keep, _ = f.check(R('-----BEGIN RSA PRIVATE KEY-----'))
        assert keep is False
