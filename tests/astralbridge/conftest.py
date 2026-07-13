"""Shared fixtures for AstralBridge tests.

These fixtures are test-only. They are not part of the shipped package and are
not used by the production pipeline. They replace the old ``astralbridge.samples``
module, which was removed because the production pipeline is input-driven and
must not contain built-in datasets.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_roman_like_sample() -> dict:
    """Return a small Roman-like sample as plain Python data."""

    image = [
        [[[0.0 for _ in range(96)] for _ in range(96)] for _ in range(4)]
    ]
    image[0][0][48][48] = 2.0
    image[0][1][48][48] = 4.0
    image[0][2][48][48] = 5.0
    image[0][3][48][48] = 6.0
    return {
        "survey": "Roman WFI demo",
        "observation_id": "roman-demo-001",
        "image": image,
        "flux_g": 2.841,
        "flux_r": 8.137,
        "flux_i": 14.293,
        "flux_z": 19.106,
        "ebv": 0.023,
    }


@pytest.fixture
def roman_like_sample() -> dict:
    """Return a Roman-like sample dict for tests."""

    return _make_roman_like_sample()


@pytest.fixture
def sample_json_file(tmp_path: Path, roman_like_sample: dict) -> Path:
    """Write the Roman-like sample to a temp JSON file and return its path."""

    path = tmp_path / "sample.json"
    path.write_text(json.dumps(roman_like_sample), encoding="utf-8")
    return path


def valid_file_path_adapter_code() -> str:
    """Return a valid adapter that reads a sample file from a path.

    This adapter follows the file-path-based contract: ``convert(self,
    sample_path)`` opens and parses the JSON file itself.
    """

    return '''import json

from astralbridge.canonical import CanonicalImage, CanonicalObservation, CanonicalScalar


ADAPTER_MANIFEST = {
    "survey": "Roman WFI demo",
    "source_fields": ["image", "flux_g", "flux_r", "flux_i", "flux_z", "ebv"],
    "target_aion_modalities": [
        "LegacySurveyImage",
        "LegacySurveyFluxG",
        "LegacySurveyFluxR",
        "LegacySurveyFluxI",
        "LegacySurveyFluxZ",
        "LegacySurveyEBV",
    ],
    "transformations": [
        "Map Roman-like optical bands into AION-compatible DES-G/DES-R/DES-I/DES-Z labels.",
        "Preserve flux-like scalar values as nanomaggie bridge scalars for the AION codec layer.",
        "Return CanonicalObservation so the deterministic AION bridge owns final modality construction.",
    ],
}


class GeneratedSurveyAdapter:
    """Generated adapter for a Roman-like WFI sample."""

    def convert(self, sample_path):
        with open(sample_path) as f:
            raw_data = json.load(f)
        image = CanonicalImage(
            data=raw_data["image"],
            bands=["DES-G", "DES-R", "DES-I", "DES-Z"],
            unit="nanomaggie",
            source_fields=["image"],
        )
        scalars = [
            CanonicalScalar("flux_g", raw_data["flux_g"], "nanomaggie", "flux_g"),
            CanonicalScalar("flux_r", raw_data["flux_r"], "nanomaggie", "flux_r"),
            CanonicalScalar("flux_i", raw_data["flux_i"], "nanomaggie", "flux_i"),
            CanonicalScalar("flux_z", raw_data["flux_z"], "nanomaggie", "flux_z"),
            CanonicalScalar("ebv", raw_data["ebv"], None, "ebv"),
        ]
        return CanonicalObservation(
            survey=raw_data.get("survey", "Roman WFI demo"),
            images=[image],
            scalars=scalars,
            metadata={"adapter": "GeneratedSurveyAdapter"},
            observation_id=raw_data.get("observation_id"),
        )
'''


def broken_adapter_code() -> str:
    """Return an adapter that fails validation (empty observation)."""

    return '''from astralbridge.canonical import CanonicalObservation


class GeneratedSurveyAdapter:
    def convert(self, sample_path):
        return CanonicalObservation(survey="broken")
'''
