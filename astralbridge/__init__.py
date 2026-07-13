"""AstralBridge: Gemma-powered survey onboarding for AION.

AstralBridge adds a survey onboarding layer around the AION astronomy
foundation model. It does not add new AION modalities or alter pretrained
codecs; it converts external survey data into AION-compatible modality
objects, runs real AION inference, and explains the result with Gemma.

The package is organized into these layers:

- ``canonical``      Survey-neutral observation containers and validators.
- ``adapters``       Gemma-powered adapter generation, validation, and repair.
- ``gemma``          Gemma 4 client (official google-generativeai SDK) and prompts.
- ``integration``    Deterministic CanonicalObservation -> AION modality mapping.
- ``inference``      Real AION inference engine (encode -> forward -> decode).
- ``interpretation`` Constrained interpretation of AION outputs.
"""

from astralbridge.canonical.observation import (
    CanonicalImage,
    CanonicalObservation,
    CanonicalScalar,
)

__all__ = [
    "CanonicalImage",
    "CanonicalObservation",
    "CanonicalScalar",
]
