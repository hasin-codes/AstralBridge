# AstralBridge Architecture

AstralBridge solves the layer between a standardized astronomy archive product
and the exact input contract of an astronomy foundation model.

It does not replace FITS, Astropy, MAST, AION codecs, or AION model weights.
Those already solve different parts of the astronomy workflow. AstralBridge
focuses on the model-compatibility step and runs the complete inference
pipeline end to end.

## Why This Exists

AION can process the modalities it was trained with. A released survey product
can be a valid observation and still not be directly consumable by AION.

The missing conversion usually includes:

- identifying the physical role of fields,
- mapping source fields to AION-supported observations,
- converting units,
- reshaping arrays,
- preserving metadata,
- producing the correct modality objects,
- validating that the result can actually enter AION,
- running AION inference and decoding the prediction,
- explaining the result in astronomical terms.

AstralBridge automates this entire workflow with a Gemma-assisted adapter agent,
a deterministic validation layer, a real AION inference engine, and a
constrained interpreter.

## Complete Pipeline

```text
User provides:
    survey documentation + schema + sample observation file
        |
        v
Gemma 4 generates survey adapter (GeneratedSurveyAdapter)
        |
        v
Validation and repair loop (compile, execute, check manifest + observation)
        |
        v
CanonicalObservation  (from the validated adapter running on the sample file)
        |
        v
AION modality objects  (canonical_to_aion, deterministic mapping)
        |
        v
CodecManager.encode(...)  (AION's pretrained codecs)
        |
        v
AION transformer forward(...)  (real model inference, single forward pass)
        |
        v
argmax over predicted token logits
        |
        v
CodecManager.decode(...)  (AION's pretrained codebook -> real decoded prediction)
        |
        v
Gemma interpretation  (constrained, does not change the prediction)
```

Every arrow in this diagram is executable code. There are no placeholder
predictions, no mocked inference, and no hardcoded outputs.

## Component Responsibilities

### Gemma 4 Adapter Agent

Gemma reads survey documentation, schema text, a sample summary, and user
instructions. It generates Python code that defines:

- `ADAPTER_MANIFEST`
- `GeneratedSurveyAdapter`
- `GeneratedSurveyAdapter.convert(self, sample_path)` — opens and parses the
  sample file itself, then returns a `CanonicalObservation`.

Gemma is used through the official `google-genai` SDK via the Google AI
Studio API.

### Adapter Manifest

Every generated adapter must define a machine-readable `ADAPTER_MANIFEST`.

Required fields:

- `survey`
- `source_fields`
- `target_aion_modalities`
- `transformations`

This prevents the output from being only a code blob. The generated adapter
must explain which data fields it maps and which existing AION modalities
it targets.

### Canonical Observation

`CanonicalObservation` is the intermediate bridge format. It is intentionally
smaller than AION's modality system. It holds survey-neutral images, scalars,
metadata, and lineage fields before a deterministic mapper converts the
observation into existing AION modality objects.

### Validation Loop

Generated code is compiled and executed against the user's sample file.

Validation checks:

- Python syntax,
- expected class and method,
- manifest completeness,
- canonical observation structure,
- image rank and band count,
- supported bands and scalar names,
- compatibility with the deterministic AION bridge.

When validation fails, AstralBridge sends the error report back to Gemma for
repair. This produces an iterative adapter workflow instead of one-shot code
generation.

### AION Bridge

The bridge maps `CanonicalObservation` into existing AION modality objects. It
is conservative by design. It refuses unsupported bands or scalar names instead
of pretending AION knows new modalities.

Examples:

- `CanonicalImage(..., bands=["DES-G", "DES-R", "DES-I", "DES-Z"])`
  maps to `LegacySurveyImage`.
- `CanonicalScalar("flux_g", ...)` maps to `LegacySurveyFluxG`.
- `CanonicalScalar("z", ...)` maps to `Z`.

### Inference Engine

The `AionInferenceEngine` runs the complete AION prediction path:

1. `CodecManager.encode(*modalities)` converts modality objects into tokens.
2. `AION.forward(tokens, target_modality)` runs the transformer (a single
   forward pass for scalar targets — no MaskGIT schedule needed).
3. `argmax` selects the most likely predicted token.
4. `CodecManager.decode({token_key: predicted_token}, target_modality)` maps
   the token back into the original value space through the pretrained
   codebook.

For redshift (`Z`), the codebook is a `GridScalarCodec` whose bucket boundaries
are loaded from the pretrained checkpoint, so the decoded value is a genuine
redshift prediction produced by the model.

### Interpreter

After AION produces a prediction, Gemma explains the output using only:

- the real predicted value (from the inference engine),
- the target name,
- input modalities and token counts,
- data lineage.

It must not invent confidence scores, natural-language AION classifications, or
extra measurements.

## CLI

Full end-to-end pipeline (primary command):

```bash
astralbridge run \
  --survey "My Survey" \
  --docs survey_docs.txt \
  --schema schema.json \
  --sample observation.json \
  --target redshift \
  --output-dir out/
```

Adapter generation only (no AION inference):

```bash
astralbridge generate \
  --survey "My Survey" \
  --docs survey_docs.txt \
  --schema schema.json \
  --sample observation.json \
  --output-dir out/
```

AION inference on an existing adapter:

```bash
astralbridge predict \
  --adapter out/adapter.py \
  --sample observation.json \
  --target redshift \
  --output-dir out/ \
  --interpret
```

## Boundary With AION

AION remains responsible for inference. AstralBridge remains
responsible for dataset onboarding and output explanation.

This boundary is important:

- AION is not a chatbot.
- AstralBridge does not retrain AION.
- Gemma does not change AION predictions.
- Generated adapters target existing AION modalities only.

The result is a complete workflow: Gemma creates and repairs the adapter,
AstralBridge validates it, AION runs the prediction, and Gemma explains the
prediction.
