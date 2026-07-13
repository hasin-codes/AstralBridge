"""Tests for the AION inference engine.

These tests verify the engine's logic (encode -> forward -> argmax -> decode)
using stubbed components, so they run without downloading the 300M model or
requiring a GPU. A marked ``slow`` test that loads the real model is included
but skipped by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from astralbridge.inference.engine import (
    AionInferenceEngine,
    PredictionResult,
    resolve_device,
)


def test_resolve_device_explicit():
    """An explicit device string should be returned as-is."""

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_auto_without_torch():
    """``auto`` without torch should raise a clear RuntimeError."""

    with patch.dict("sys.modules", {"torch": None}):
        with pytest.raises(RuntimeError, match="PyTorch is required"):
            resolve_device("auto")


def test_prediction_result_as_dict_normalizes_tensors():
    """PredictionResult.as_dict should normalize tensor-like values to Python."""

    @dataclass
    class FakeTensor:
        def item(self):
            return 0.341

        def tolist(self):
            return 0.341

    result = PredictionResult(
        target="redshift",
        token_key="tok_z",
        predicted_tokens=FakeTensor(),
        predicted_value=FakeTensor(),
        token_counts={"tok_image": 576, "tok_flux_g": 1},
    )

    as_dict = result.as_dict()
    assert as_dict["target"] == "redshift"
    assert as_dict["token_key"] == "tok_z"
    assert as_dict["predicted_value"] == 0.341
    assert as_dict["input_token_counts"]["tok_image"] == 576


def test_engine_raises_without_torch():
    """The engine should give a clear error if torch is not installed."""

    engine = AionInferenceEngine(device="cpu")

    with patch.dict("sys.modules", {"torch": None, "aion.codecs": None, "aion.model": None}):
        with pytest.raises(RuntimeError, match="PyTorch is required"):
            engine._ensure_loaded()


def test_engine_predict_uses_real_forward_and_decode():
    """predict() should call model.forward then codec_manager.decode.

    This verifies the engine wires the real AION path (encode -> forward ->
    argmax -> decode) without downloading any weights. We stub the model and
    codec manager so only the engine logic is exercised.
    """

    engine = AionInferenceEngine(device="cpu")

    # Bypass lazy loading by injecting stubs.
    fake_logits = MagicMock()
    fake_logits.__getitem__ = MagicMock(return_value=fake_logits)
    fake_logits.argmax = MagicMock(return_value=42)

    fake_model = MagicMock()
    fake_model.return_value = {"tok_z": fake_logits}

    fake_decoded = MagicMock()
    fake_decoded.value = 0.341
    fake_codec_manager = MagicMock()
    fake_codec_manager.encode.return_value = {
        "tok_image": MagicMock(shape=[1, 576]),
        "tok_flux_g": MagicMock(shape=[1, 1]),
    }
    fake_codec_manager.decode.return_value = fake_decoded

    engine._model = fake_model
    engine._codec_manager = fake_codec_manager

    # A fake target modality class.
    class FakeTarget:
        token_key = "tok_z"
        name = "redshift"

    result = engine.predict([MagicMock()], FakeTarget, target_name="redshift")

    # The model forward must have been called with encoded tokens.
    assert fake_model.called
    # The codec manager must have decoded the predicted token.
    assert fake_codec_manager.decode.called
    # The result should carry the real decoded value.
    assert result.predicted_value == 0.341
    assert result.target == "redshift"
    assert result.token_key == "tok_z"


@pytest.mark.slow
def test_engine_predict_with_real_model():
    """Load the real AION model and run a redshift prediction end to end.

    This test is skipped by default. Run it manually with::

        pytest tests/astralbridge/test_inference.py -m slow

    It downloads the AION Base model (~1.2 GB) from HuggingFace on first run.
    Requires torch and network access. The input is kept minimal (one small
    image + four flux scalars) so the forward pass is light on CPU.
    """

    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    from astralbridge import CanonicalImage, CanonicalObservation, CanonicalScalar
    from astralbridge.integration import canonical_to_aion
    from astralbridge.inference import AionInferenceEngine

    # Build a minimal canonical observation with AION-supported modalities.
    # A 4-band 96x96 image (DES g/r/i/z) + four flux scalars + EBV.
    image = torch.zeros(1, 4, 96, 96, dtype=torch.float32)
    observation = CanonicalObservation(
        survey="integration test",
        images=[CanonicalImage(data=image, bands=["DES-G", "DES-R", "DES-I", "DES-Z"])],
        scalars=[
            CanonicalScalar("flux_g", 2.841, "nanomaggie", "flux_g"),
            CanonicalScalar("flux_r", 8.137, "nanomaggie", "flux_r"),
            CanonicalScalar("flux_i", 14.293, "nanomaggie", "flux_i"),
            CanonicalScalar("flux_z", 19.106, "nanomaggie", "flux_z"),
        ],
    )

    # Map into existing AION modality objects.
    modalities = canonical_to_aion(observation)
    assert len(modalities) == 5  # 1 image + 4 scalars

    # Run real AION inference on CPU.
    engine = AionInferenceEngine(device="cpu")
    prediction = engine.predict_by_name(modalities, "redshift")

    # The prediction must be a real numeric value, not None or a placeholder.
    value = prediction.predicted_value
    if hasattr(value, "item"):
        value = value.item()
    assert isinstance(value, (int, float))
    assert 0.0 <= value <= 1.0  # AION redshift range for the Z codec

    assert prediction.target == "redshift"
    assert prediction.token_key == "tok_z"
    assert "tok_image" in prediction.token_counts
