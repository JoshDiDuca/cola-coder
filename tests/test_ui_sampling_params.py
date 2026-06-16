"""Pin the advanced-sampling plumbing (min_p + top_n_sigma) through the UI layer.

This cycle ``min_p`` (confidence-scaled probability floor) and ``top_n_sigma``
(top-nsigma logit truncation) were plumbed through the UI request schemas and
endpoints. The generator already supported both. These tests pin the contract:

* the request schemas expose the new fields with the right defaults,
* ``BestOfNRequest`` gets ``min_p`` only (top-nsigma is not wired for best-of),
* the FastAPI app constructs cleanly (so the endpoint wiring that references
  ``req.min_p`` / ``req.top_n_sigma`` is at least import/construction-clean),
* the sampler still accepts both kwargs (the technique the UI now exposes),
* and ``_min_p_filter`` actually masks low-probability tokens (optional, torch).

No network, GPU, or model loads happen here.
"""

from __future__ import annotations

import inspect

import pytest

from cola_coder.ui import schemas as sch


class TestSamplingFieldDefaults:
    """Each inference request schema defaults both knobs to 0.0 (disabled)."""

    def test_inference_request_defaults(self) -> None:
        """``InferenceRequest`` defaults ``min_p`` and ``top_n_sigma`` to 0.0."""
        req = sch.InferenceRequest(prompt="hi", checkpoint="ckpt", config="cfg")
        assert req.min_p == 0.0
        assert req.top_n_sigma == 0.0

    def test_chat_request_defaults(self) -> None:
        """``ChatRequest`` defaults ``min_p`` and ``top_n_sigma`` to 0.0."""
        req = sch.ChatRequest(
            messages=[sch.ChatMessage(role="user", content="hi")],
            checkpoint="ckpt",
            config="cfg",
        )
        assert req.min_p == 0.0
        assert req.top_n_sigma == 0.0

    def test_fim_request_defaults(self) -> None:
        """``FimRequest`` defaults ``min_p`` and ``top_n_sigma`` to 0.0."""
        req = sch.FimRequest(prefix="pre", suffix="suf", checkpoint="ckpt", config="cfg")
        assert req.min_p == 0.0
        assert req.top_n_sigma == 0.0


class TestSamplingFieldOverrides:
    """Each schema accepts overrides for both knobs and round-trips them."""

    def test_inference_request_override_roundtrip(self) -> None:
        """``InferenceRequest`` accepts overrides and dumps them back."""
        req = sch.InferenceRequest(
            prompt="hi", checkpoint="ckpt", config="cfg", min_p=0.1, top_n_sigma=1.0
        )
        dumped = req.model_dump()
        assert dumped["min_p"] == 0.1
        assert dumped["top_n_sigma"] == 1.0

    def test_chat_request_override_roundtrip(self) -> None:
        """``ChatRequest`` accepts overrides and dumps them back."""
        req = sch.ChatRequest(
            messages=[sch.ChatMessage(role="user", content="hi")],
            checkpoint="ckpt",
            config="cfg",
            min_p=0.1,
            top_n_sigma=1.0,
        )
        dumped = req.model_dump()
        assert dumped["min_p"] == 0.1
        assert dumped["top_n_sigma"] == 1.0

    def test_fim_request_override_roundtrip(self) -> None:
        """``FimRequest`` accepts overrides and dumps them back."""
        req = sch.FimRequest(
            prefix="pre",
            suffix="suf",
            checkpoint="ckpt",
            config="cfg",
            min_p=0.1,
            top_n_sigma=1.0,
        )
        dumped = req.model_dump()
        assert dumped["min_p"] == 0.1
        assert dumped["top_n_sigma"] == 1.0


class TestBestOfNRequest:
    """Best-of-N gets ``min_p`` only; top-nsigma is intentionally not wired."""

    def test_min_p_default(self) -> None:
        """``BestOfNRequest`` defaults ``min_p`` to 0.0."""
        req = sch.BestOfNRequest(prompt="hi", checkpoint="ckpt", config="cfg")
        assert req.min_p == 0.0

    def test_min_p_override_roundtrip(self) -> None:
        """``BestOfNRequest`` accepts a ``min_p`` override and dumps it back."""
        req = sch.BestOfNRequest(prompt="hi", checkpoint="ckpt", config="cfg", min_p=0.1)
        assert req.model_dump()["min_p"] == 0.1

    def test_no_top_n_sigma_field(self) -> None:
        """``BestOfNRequest`` must NOT expose a ``top_n_sigma`` field."""
        assert "top_n_sigma" not in sch.BestOfNRequest.model_fields


class TestAppConstruction:
    """The FastAPI app imports and constructs without raising."""

    def test_app_module_imports(self) -> None:
        """``cola_coder.ui.app`` imports and exposes ``create_app``."""
        from cola_coder.ui import app as ui_app

        assert hasattr(ui_app, "create_app")

    def test_create_app_does_not_raise(self) -> None:
        """``create_app()`` builds the app (endpoint wiring is construction-clean)."""
        from cola_coder.ui.app import create_app

        app = create_app()
        assert app is not None


class TestSamplerContract:
    """The sampler exposes the kwargs the UI now plumbs through to it."""

    def test_sample_next_token_accepts_kwargs(self) -> None:
        """``sample_next_token`` accepts both ``min_p`` and ``top_n_sigma``."""
        from cola_coder.inference.sampling import sample_next_token

        params = inspect.signature(sample_next_token).parameters
        assert "min_p" in params
        assert "top_n_sigma" in params

    def test_min_p_filter_masks_low_prob_tokens(self) -> None:
        """``_min_p_filter`` masks below-threshold tokens to -inf (optional, torch).

        The schema/signature assertions above are the important ones; this is an
        extra numeric sanity check, skipped cleanly when torch is unavailable.
        """
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        from cola_coder.inference.sampling import _min_p_filter

        # One dominant token, one tiny-probability token. With a min_p floor the
        # low-probability token must be masked to -inf.
        logits = torch.tensor([10.0, -10.0, 9.0])
        filtered = _min_p_filter(logits, min_p=0.5)
        assert torch.isinf(filtered[1])
        assert filtered[1] < 0
        # The dominant token (max prob) is always kept.
        assert not torch.isinf(filtered[0])
