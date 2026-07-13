"""Real AION inference execution.

This module wires the existing AION model, codecs, and prediction targets into a
single engine that runs the complete inference path:

    AION modality objects
        -> CodecManager.encode(...)
        -> AION transformer forward(...)
        -> argmax over predicted token logits
        -> CodecManager.decode(...)
        -> real decoded prediction

No new model code is introduced. The engine only calls AION's existing
``forward`` and ``CodecManager.decode`` so predictions are produced by the
pretrained model exactly as AION already supports.
"""

from astralbridge.inference.engine import (
    AionInferenceEngine,
    PredictionResult,
    resolve_device,
)

__all__ = [
    "AionInferenceEngine",
    "PredictionResult",
    "resolve_device",
]
