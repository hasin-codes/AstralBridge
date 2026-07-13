"""Validation helpers for canonical observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_IMAGE_BANDS = {
    "DES-G",
    "DES-R",
    "DES-I",
    "DES-Z",
    "HSC-G",
    "HSC-R",
    "HSC-I",
    "HSC-Z",
    "HSC-Y",
}

SUPPORTED_SCALARS = {
    "flux_g",
    "flux_r",
    "flux_i",
    "flux_z",
    "flux_w1",
    "flux_w2",
    "flux_w3",
    "flux_w4",
    "shape_r",
    "shape_e1",
    "shape_e2",
    "ebv",
    "ra",
    "dec",
    "parallax",
    "z",
}


@dataclass
class ValidationIssue:
    """A single validation issue found in a canonical observation or adapter."""

    code: str
    message: str
    path: str | None = None


@dataclass
class ValidationReport:
    """Validation result with machine-readable issues."""

    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def as_text(self) -> str:
        """Return a concise text form suitable for Gemma repair prompts."""

        if self.passed:
            return "PASSED"
        return "\n".join(
            f"{issue.code}: {issue.message}"
            + (f" at {issue.path}" if issue.path else "")
            for issue in self.issues
        )


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        if isinstance(value, list):
            dims = []
            current = value
            while isinstance(current, list):
                dims.append(len(current))
                current = current[0] if current else []
            return tuple(dims)
        return None
    return tuple(int(dim) for dim in shape)


def validate_canonical_observation(observation: Any) -> ValidationReport:
    """Validate a CanonicalObservation without requiring AION weights."""

    issues: list[ValidationIssue] = []

    if not hasattr(observation, "survey") or not observation.survey:
        issues.append(
            ValidationIssue("missing_survey", "CanonicalObservation.survey is required")
        )

    if not getattr(observation, "images", None) and not getattr(
        observation, "scalars", None
    ):
        issues.append(
            ValidationIssue(
                "empty_observation",
                "At least one image or scalar modality must be present",
            )
        )

    for index, image in enumerate(getattr(observation, "images", [])):
        if not image.bands:
            issues.append(
                ValidationIssue("missing_bands", "Image bands are required", f"images[{index}]")
            )
        unsupported = [band for band in image.bands if band not in SUPPORTED_IMAGE_BANDS]
        if unsupported:
            issues.append(
                ValidationIssue(
                    "unsupported_bands",
                    f"Unsupported AION bridge bands: {unsupported}",
                    f"images[{index}].bands",
                )
            )

        shape = _shape_of(image.data)
        if shape is None:
            issues.append(
                ValidationIssue(
                    "missing_shape",
                    "Image data must expose a shape attribute",
                    f"images[{index}].data",
                )
            )
        elif len(shape) not in (3, 4):
            issues.append(
                ValidationIssue(
                    "invalid_image_rank",
                    f"Image data must be rank 3 or 4, got shape {shape}",
                    f"images[{index}].data",
                )
            )
        else:
            band_axis = 1 if len(shape) == 4 else 0
            if shape[band_axis] != len(image.bands):
                issues.append(
                    ValidationIssue(
                        "band_count_mismatch",
                        "Image band count must match the image channel dimension: "
                        f"{len(image.bands)} bands for shape {shape}",
                        f"images[{index}]",
                    )
                )

    for index, scalar in enumerate(getattr(observation, "scalars", [])):
        if scalar.name not in SUPPORTED_SCALARS:
            issues.append(
                ValidationIssue(
                    "unsupported_scalar",
                    f"Unsupported scalar name: {scalar.name}",
                    f"scalars[{index}].name",
                )
            )

    return ValidationReport(passed=not issues, issues=issues)
