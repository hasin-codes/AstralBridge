"""Prompt builders for Gemma adapter generation and interpretation.

These prompts define the contract that generated adapters must follow. The
adapter owns all file I/O: it receives a sample file path and is responsible
for opening, parsing, and converting the survey data into a
``CanonicalObservation``.
"""

from __future__ import annotations

from typing import Protocol


class AdapterPromptRequest(Protocol):
    survey_name: str
    documentation_text: str
    schema_text: str
    sample_summary: str
    user_instructions: str


def build_adapter_prompt(request: AdapterPromptRequest) -> str:
    """Build the prompt Gemma receives for adapter generation.

    The generated adapter must read a sample file from disk (the path is passed
    to ``convert``), parse it, and return a ``CanonicalObservation``. This makes
    the adapter survey-specific and input-driven rather than tied to any
    built-in dataset.
    """

    return f"""You are writing a Python survey adapter for AstralBridge.

Goal:
Convert an external astronomy survey file into a
astralbridge.canonical.CanonicalObservation that maps into AION's existing
modality classes.

Rules:
- Do not create new AION modalities.
- Do not alter AION codecs or model weights.
- Generate executable Python code only.
- Define ADAPTER_MANIFEST as a dict with keys:
  survey, source_fields, target_aion_modalities, transformations.
- The adapter must define one class named GeneratedSurveyAdapter.
- The class must implement convert(self, sample_path).
- sample_path is a string path to a file on disk. The adapter must open and
  parse this file itself (e.g. json.load, astropy.io.fits, csv, numpy.load).
- convert() must return CanonicalObservation.
- Use CanonicalImage and CanonicalScalar where appropriate.
- Only use AION-supported image bands: DES-G, DES-R, DES-I, DES-Z, HSC-G,
  HSC-R, HSC-I, HSC-Z, HSC-Y.
- Only use AION-supported scalar names: flux_g, flux_r, flux_i, flux_z,
  flux_w1, flux_w2, flux_w3, flux_w4, shape_r, shape_e1, shape_e2, ebv, ra,
  dec, parallax, z.
- Image data must have shape [batch, bands, height, width] or
  [bands, height, width]. Height and width must each be 96 to match AION's
  LegacySurveyImage token grid.
- Preserve units in metadata or unit fields.
- The manifest must describe what the adapter maps, not marketing text.
- Use only the Python standard library plus numpy and torch. Do not import
  astralbridge modules other than the canonical types.

Survey name:
{request.survey_name}

User instructions:
{request.user_instructions}

Schema:
{request.schema_text}

Documentation:
{request.documentation_text}

Sample summary:
{request.sample_summary}
"""


def build_repair_prompt(
    request: AdapterPromptRequest, previous_code: str, validation_errors: str
) -> str:
    """Build the prompt Gemma receives after validation failure."""

    return f"""Repair this AstralBridge adapter.

The adapter must still define GeneratedSurveyAdapter.convert(self, sample_path)
and return CanonicalObservation.
It must also define a complete ADAPTER_MANIFEST dict.
The convert method must open and parse the file at sample_path itself.

Validation errors:
{validation_errors}

Previous code:
```python
{previous_code}
```

Original request:
{build_adapter_prompt(request)}
"""


def build_interpretation_prompt(
    prediction_value: str,
    target: str,
    modalities: dict[str, int],
    lineage: dict,
) -> str:
    """Build a constrained interpretation prompt.

    Gemma explains the AION prediction without changing the predicted value.
    It must not invent confidence scores, natural-language classifications, or
    extra measurements that AION did not produce.
    """

    return f"""Explain an AION prediction without changing the predicted value.

Rules:
- Do not invent confidence scores.
- Do not claim AION produced natural-language classifications.
- Reference only provided modalities, token counts, target, and lineage.
- Use concise prose.
- State the predicted value clearly and explain what it represents
  in astronomical terms.

Target:
{target}

Prediction:
{prediction_value}

Input modalities and token counts:
{modalities}

Data lineage:
{lineage}
"""
