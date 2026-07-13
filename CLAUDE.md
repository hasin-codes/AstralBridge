# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

**AstralBridge** is the primary contribution of this repository. It is a
Gemma-powered survey onboarding layer for the AION astronomy foundation model.
It accepts arbitrary astronomy survey products, generates an executable adapter
with Gemma 4, validates it, runs real AION inference, and explains the
prediction with Gemma.

The upstream AION model (developed by Polymathic AI) is included as a
dependency and kept intact. AstralBridge does not retrain AION, add new
modalities, or alter pretrained codecs.

### AstralBridge Layers

- `astralbridge/canonical/` — Survey-neutral `CanonicalObservation` and validators.
- `astralbridge/adapters/` — Gemma adapter generation, validation, repair loop.
- `astralbridge/gemma/` — Gemma 4 client (official `google-genai` SDK) + prompts.
- `astralbridge/integration/` — Deterministic `CanonicalObservation` → AION modality mapping.
- `astralbridge/inference/` — Real AION inference engine (encode → forward → decode).
- `astralbridge/interpretation/` — Constrained interpretation of AION outputs.
- `astralbridge/cli.py` — CLI: `run` (full pipeline), `generate` (adapter only), `predict` (inference only).

### AION (upstream, kept intact)

AION (AstronomIcal Omnimodal Network) is a large omnimodal transformer model
for astronomical surveys. It processes 39 distinct astronomical data modalities
using a two-stage architecture:

1. **Modality-specific tokenizers** transform raw inputs into discrete tokens.
2. **Unified encoder-decoder transformer** processes all token streams via
   multimodal masked modeling (4M).

The model comes in three variants: Base (300M), Large (800M), and XLarge (3B).

## Development Commands

### Testing
```bash
pytest tests/astralbridge -q      # Run AstralBridge tests (no GPU/API needed)
pytest tests/codecs/              # Run AION codec tests
pytest                            # Run all tests
```

### Linting and Code Quality
```bash
ruff check .                      # Check code style and lint
ruff check . --fix                # Auto-fix linting issues
```

### Installation for Development
```bash
pip install -e .[torch,gemma,dev]  # Editable install with all extras
```

### Documentation
```bash
cd docs && make html              # Build Sphinx documentation
```

## Architecture Overview

### AstralBridge Pipeline

```
User inputs (docs + schema + sample file)
  -> Gemma 4 generates adapter
  -> validation + repair loop
  -> CanonicalObservation
  -> canonical_to_aion (existing AION modalities)
  -> CodecManager.encode
  -> AION.forward (single forward pass for scalar targets)
  -> argmax -> CodecManager.decode -> real prediction
  -> Gemma interpretation of the prediction
```

### Key Design Decisions

1. **Input-driven**: No built-in datasets. Users supply their own survey data.
2. **Real inference**: `AionInferenceEngine` calls the existing AION `forward`
   and `CodecManager.decode` — no mocks or placeholders.
3. **Gemma as the engine**: The official `google-genai` SDK generates
   adapters, repairs them, and interprets predictions.
4. **Architectural honesty**: The bridge refuses unsupported bands/scalars
   rather than pretending AION understands them.
5. **Lazy imports**: `torch` and `aion` are imported lazily so adapter
   generation works without the model stack installed.

### AION Core Components (upstream)

- `aion/model.py`: Main AION wrapper, inherits from FM (4M) transformer.
- `aion/fourm/`: 4M transformer implementation (encoder-decoder, embeddings).
- `aion/codecs/`: Modality tokenization system (`CodecManager` loads codecs
  from HuggingFace).
- `aion/modalities.py`: Type definitions for all astronomical data types.

## Code Conventions

- Type hints are mandatory, using `jaxtyping` for tensor shapes.
- Modality classes use `@dataclass` and inherit from `Modality`.
- All tensor operations should handle device placement explicitly.
- AstralBridge modules use lazy imports for `torch`/`aion` to keep the package
  lightweight when only adapter generation is needed.
- Test data is pre-computed and stored in `tests/test_data/` as `.pt` files.

## Gemma Configuration

- Primary: `GEMINI_API_KEY` environment variable (free key from
  https://aistudio.google.com/app/apikey).
- Model: `ASTRALBRIDGE_GEMMA_MODEL` (defaults to `gemma-4-31b-it`).
- Alternative: `ASTRALBRIDGE_GEMMA_COMMAND` for local runners.
- Official docs: https://ai.google.dev/gemma/docs

## Astronomical Context

AION processes data from major surveys:
- **Legacy Survey**: Optical images and catalogs (g,r,i,z bands + WISE).
- **HSC (Hyper Suprime-Cam)**: Deep optical imaging (g,r,i,z,y bands).
- **Gaia**: Astrometry, photometry, and BP/RP spectra.
- **SDSS/DESI**: Optical spectra.

AstralBridge adapters map external survey data into these existing modality
types. The primary prediction target is redshift (`Z`).
