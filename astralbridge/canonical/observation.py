"""Survey-neutral observation containers.

These classes are intentionally smaller than AION modalities. They represent
the bridge format produced by survey adapters before a deterministic mapper
converts the data into existing AION modality classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CanonicalImage:
    """A survey-neutral image observation.

    Args:
        data: Image-like object, normally a torch tensor or numpy array with
            shape [batch, bands, height, width] or [bands, height, width].
        bands: Band names already mapped to AION-compatible band labels such as
            DES-G, DES-R, DES-I, DES-Z, HSC-G, HSC-R, HSC-I, HSC-Z, HSC-Y.
        unit: Physical unit of the values after adapter conversion.
        source_fields: Original field or extension names used to build this
            image. This preserves data lineage for generated adapters.
    """

    data: Any
    bands: list[str]
    unit: str | None = None
    source_fields: list[str] = field(default_factory=list)


@dataclass
class CanonicalScalar:
    """A survey-neutral scalar observation.

    Args:
        name: Canonical scalar name. Supported AION bridge names include
            flux_g, flux_r, flux_i, flux_z, flux_w1, flux_w2, flux_w3, flux_w4,
            shape_r, shape_e1, shape_e2, ebv, ra, dec, parallax, z.
        value: Scalar-like object, normally a float, list, numpy array, or torch
            tensor. The bridge normalizes it to the AION [batch, 1] convention.
        unit: Unit after adapter conversion.
        source_field: Original source field name.
    """

    name: str
    value: Any
    unit: str | None = None
    source_field: str | None = None


@dataclass
class CanonicalObservation:
    """Bridge object produced by survey adapters.

    The observation can contain any subset of supported fields. Missing
    modalities are normal in astronomy and remain missing until a downstream
    model predicts them.
    """

    survey: str
    images: list[CanonicalImage] = field(default_factory=list)
    scalars: list[CanonicalScalar] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    observation_id: str | None = None

    def token_lineage(self) -> dict[str, Any]:
        """Return deterministic lineage metadata without model interpretation."""

        return {
            "survey": self.survey,
            "observation_id": self.observation_id,
            "image_count": len(self.images),
            "scalar_count": len(self.scalars),
            "image_sources": [image.source_fields for image in self.images],
            "scalar_sources": {
                scalar.name: scalar.source_field for scalar in self.scalars
            },
        }
