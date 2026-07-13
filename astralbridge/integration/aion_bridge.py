"""Deterministic mapping from AstralBridge observations to AION modalities.

This module maps ``CanonicalObservation`` objects into existing AION modality
classes. It is intentionally conservative: it only maps to modality types
already present in AION's pretrained vocabulary and refuses unsupported bands
or scalar names.

The ``torch`` and ``aion`` imports are deferred to call time so the rest of
AstralBridge can be imported without the heavyweight model stack installed.
"""

from __future__ import annotations

from typing import Any


class AionBridgeError(ValueError):
    """Raised when a canonical observation cannot be mapped to AION."""


# These mappings are populated lazily on first use to avoid importing torch and
# aion at module load time. They map canonical names to AION modality classes.
_SCALAR_MODALITY_BY_NAME: dict[str, type] | None = None
_TARGET_MODALITY_BY_NAME: dict[str, type] | None = None


def _load_mappings() -> tuple[dict[str, type], dict[str, type]]:
    """Lazily import AION modalities and build the name -> class mappings."""

    global _SCALAR_MODALITY_BY_NAME, _TARGET_MODALITY_BY_NAME

    if _SCALAR_MODALITY_BY_NAME is not None:
        return _SCALAR_MODALITY_BY_NAME, _TARGET_MODALITY_BY_NAME

    from aion.modalities import (
        Dec,
        GaiaParallax,
        LegacySurveyEBV,
        LegacySurveyFluxG,
        LegacySurveyFluxI,
        LegacySurveyFluxR,
        LegacySurveyFluxW1,
        LegacySurveyFluxW2,
        LegacySurveyFluxW3,
        LegacySurveyFluxW4,
        LegacySurveyFluxZ,
        LegacySurveyShapeE1,
        LegacySurveyShapeE2,
        LegacySurveyShapeR,
        Ra,
        Z,
    )

    _SCALAR_MODALITY_BY_NAME = {
        "flux_g": LegacySurveyFluxG,
        "flux_r": LegacySurveyFluxR,
        "flux_i": LegacySurveyFluxI,
        "flux_z": LegacySurveyFluxZ,
        "flux_w1": LegacySurveyFluxW1,
        "flux_w2": LegacySurveyFluxW2,
        "flux_w3": LegacySurveyFluxW3,
        "flux_w4": LegacySurveyFluxW4,
        "shape_r": LegacySurveyShapeR,
        "shape_e1": LegacySurveyShapeE1,
        "shape_e2": LegacySurveyShapeE2,
        "ebv": LegacySurveyEBV,
        "ra": Ra,
        "dec": Dec,
        "parallax": GaiaParallax,
        "z": Z,
    }
    _TARGET_MODALITY_BY_NAME = {
        "z": Z,
        "redshift": Z,
    }
    return _SCALAR_MODALITY_BY_NAME, _TARGET_MODALITY_BY_NAME


def _as_float_tensor(value: Any):
    """Normalize a scalar-like value to a [batch, 1] float tensor."""

    import torch

    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.dim() == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.dim() == 1:
        tensor = tensor.reshape(-1, 1)
    return tensor


def _image_tensor(value: Any):
    """Normalize image data to a [batch, bands, height, width] float tensor."""

    import torch

    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 4:
        raise AionBridgeError(
            f"Image data must have shape [batch,bands,height,width], got {tuple(tensor.shape)}"
        )
    return tensor


def canonical_to_aion(observation):
    """Convert a canonical observation into existing AION modality objects.

    This function is intentionally conservative. It only maps to modality types
    already present in AION's pretrained vocabulary.
    """

    from astralbridge.canonical.validators import validate_canonical_observation

    scalar_map, _ = _load_mappings()

    report = validate_canonical_observation(observation)
    if not report.passed:
        raise AionBridgeError(report.as_text())

    # Import here to avoid loading torch at module import time.
    from aion.modalities import LegacySurveyImage

    modalities: list = []

    for image in observation.images:
        des_bands = [band for band in image.bands if band.startswith("DES-")]
        if len(des_bands) != len(image.bands):
            raise AionBridgeError(
                "Only DES-* image bands currently map to LegacySurveyImage. "
                f"Received {image.bands}."
            )
        modalities.append(
            LegacySurveyImage(flux=_image_tensor(image.data), bands=image.bands)
        )

    for scalar in observation.scalars:
        modality_type = scalar_map.get(scalar.name)
        if modality_type is None:
            raise AionBridgeError(f"Unsupported scalar for AION bridge: {scalar.name}")
        modalities.append(modality_type(value=_as_float_tensor(scalar.value)))

    return modalities


def target_modality_from_name(name: str):
    """Return an AION target modality class by human-friendly name.

    Args:
        name: Target name such as ``"redshift"`` or ``"z"``.

    Returns:
        The AION modality class for the requested target.

    Raises:
        AionBridgeError: If the target name is not supported.
    """

    _, target_map = _load_mappings()

    normalized = name.strip().lower()
    if normalized not in target_map:
        raise AionBridgeError(f"Unsupported prediction target: {name}")
    return target_map[normalized]


# Re-exported for convenient imports. Resolves lazily; callers should use
# target_modality_from_name() for the actual class lookup.
def get_target_modality_by_name() -> dict[str, type]:
    """Return the target name -> modality class mapping (lazily loaded)."""

    _, target_map = _load_mappings()
    return target_map
