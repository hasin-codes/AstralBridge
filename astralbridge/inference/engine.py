"""Real AION inference engine.

Runs the complete prediction path using the existing AION implementation:

    modality objects -> CodecManager.encode -> AION.forward -> argmax ->
    CodecManager.decode -> real decoded value

The engine imports ``torch`` and ``aion`` lazily so the rest of AstralBridge can
be imported without the heavyweight model dependencies installed. This keeps the
adapter-generation and validation layers usable in lightweight environments
while the full pipeline still executes end to end when the model stack is
available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from astralbridge.integration.aion_bridge import (
    get_target_modality_by_name,
    target_modality_from_name,
)


@dataclass
class PredictionResult:
    """A real prediction produced by the AION model.

    Attributes:
        target: Human-friendly target name, e.g. "redshift".
        token_key: AION token key that was predicted, e.g. "tok_z".
        predicted_tokens: Integer token indices selected by argmax.
        predicted_value: Decoded value from CodecManager.decode.
            For redshift (``Z``) this is a float in the AION redshift range.
        token_counts: Number of input tokens consumed per input modality.
            Used as lineage by the interpreter.
    """

    target: str
    token_key: str
    predicted_tokens: Any
    predicted_value: Any
    token_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the prediction."""

        value = self.predicted_value
        # Normalize torch tensors / numpy scalars to plain Python for JSON.
        if hasattr(value, "tolist"):
            value = value.tolist()
        if hasattr(value, "item") and not isinstance(value, (list, dict)):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass
        tokens = self.predicted_tokens
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        return {
            "target": self.target,
            "token_key": self.token_key,
            "predicted_value": value,
            "predicted_tokens": tokens,
            "input_token_counts": self.token_counts,
        }


def resolve_device(device: str = "auto") -> str:
    """Resolve a device string, auto-selecting CUDA when available.

    Args:
        device: ``"auto"``, ``"cuda"``, ``"cpu"``, or an explicit device string.

    Returns:
        The resolved device string suitable for ``torch`` and AION.
    """

    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - exercised via error path below
        raise RuntimeError(
            "PyTorch is required to run AION inference. "
            "Install it with: pip install -e .[torch]"
        )


class AionInferenceEngine:
    """Execute the real AION inference pipeline.

    The engine loads the pretrained AION model and codecs from HuggingFace and
    runs the full ``encode -> forward -> decode`` path. It performs no
    retraining, no mock predictions, and no hardcoded values. Every prediction
    is produced by the model's own forward pass and decoded through the
    pretrained codec codebooks.
    """

    def __init__(
        self,
        device: str = "auto",
        model_name: str = "polymathic-ai/aion-base",
    ) -> None:
        self.device = resolve_device(device)
        self.model_name = model_name
        self._model = None
        self._codec_manager = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the AION model and codec manager on first use."""

        if self._model is not None and self._codec_manager is not None:
            return

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required to run AION inference. "
                "Install it with: pip install -e .[torch]"
            ) from exc

        try:
            from aion.codecs import CodecManager
            from aion.model import AION
        except ImportError as exc:
            raise RuntimeError(
                "AION is required for inference but could not be imported. "
                "Install the package with: pip install -e .[torch]"
            ) from exc

        with torch.no_grad():
            self._codec_manager = CodecManager(device=self.device)
            self._model = (
                AION.from_pretrained(self.model_name)
                .to(self.device)
                .eval()
            )

    @property
    def model(self):
        """The loaded AION model (lazily loaded)."""

        self._ensure_loaded()
        return self._model

    @property
    def codec_manager(self):
        """The loaded CodecManager (lazily loaded)."""

        self._ensure_loaded()
        return self._codec_manager

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        modalities: list[Any],
        target_modality: type,
        target_name: str | None = None,
    ) -> PredictionResult:
        """Run the full AION prediction path for scalar targets.

        This performs a single forward pass (no MaskGIT generation schedule),
        which is the correct AION inference path for scalar targets such as
        redshift (``Z``). The predicted token is decoded back into the original
        original value space through the pretrained codec codebook, so the returned
        value is a genuine model prediction.

        Args:
            modalities: AION modality objects (e.g. ``LegacySurveyImage``,
                ``LegacySurveyFluxG``) produced by the AstralBridge bridge.
            target_modality: The AION modality class to predict, e.g. ``Z``.
            target_name: Optional human-friendly target label for the result.

        Returns:
            A ``PredictionResult`` containing the decoded prediction.
        """

        import torch

        self._ensure_loaded()

        token_key = target_modality.token_key
        target_label = target_name or target_modality.name

        # 1. Encode input modalities into AION tokens.
        tokens = self.codec_manager.encode(*modalities)
        token_counts = {key: int(t.shape[1]) for key, t in tokens.items()}

        # 2. Run the AION transformer forward pass for the requested target.
        with torch.no_grad():
            logits = self.model(tokens, target_modality=target_modality)

        target_logits = logits[token_key]

        # 3. Select the most likely token via argmax.
        predicted_tokens = target_logits.argmax(dim=-1)

        # 4. Decode the predicted token back into the original value space
        #    using the pretrained codec codebook. For redshift (Z) this maps
        #    through GridScalarCodec's learned bucket boundaries.
        decoded = self.codec_manager.decode(
            {token_key: predicted_tokens}, target_modality
        )

        return PredictionResult(
            target=target_label,
            token_key=token_key,
            predicted_tokens=predicted_tokens,
            predicted_value=decoded.value,
            token_counts=token_counts,
        )

    def predict_by_name(
        self, modalities: list[Any], target_name: str
    ) -> PredictionResult:
        """Predict a target identified by human-friendly name.

        Args:
            modalities: AION modality objects.
            target_name: Target name such as ``"redshift"`` or ``"z"``.

        Returns:
            A ``PredictionResult`` for the requested target.
        """

        target_modality = target_modality_from_name(target_name)
        return self.predict(modalities, target_modality, target_name=target_name)


__all__ = [
    "AionInferenceEngine",
    "PredictionResult",
    "resolve_device",
    "get_target_modality_by_name",
]
