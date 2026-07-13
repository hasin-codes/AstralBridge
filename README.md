# AstralBridge

AstralBridge is an AI assisted onboarding system for astronomy foundation
models. It introduces a canonical observation layer that enables new
astronomical surveys to be connected to the AION astronomy foundation model
through AI generated adapters instead of manually engineered integration
pipelines.

AstralBridge does not attempt to teach AION new astronomy. Instead, it helps
existing astronomical observations reach the representations that AION already
understands.

```text
Astronomical Survey
        │
        ▼
Gemma 4  (Adapter Generation)
        │
        ▼
Validation & Repair
        │
        ▼
Canonical Observation
        │
        ▼
AION Encoder-Decoder Transformer
        │
        ▼
Decoded Prediction
        │
        ▼
Gemma 4  (Interpretation)
```

Astronomy is entering an era where telescopes are producing unprecedented
amounts of observations. Missions such as the Roman Space Telescope, Euclid,
Gaia, DESI, JWST, and ground based surveys are continuously expanding our
understanding of the universe.

However, collecting observations is only one part of the challenge. Modern
astronomy foundation models can extract information from these datasets, but
every new survey introduces engineering work before the model can be used.
AION, an omnimodal encoder-decoder transformer pretrained on 39 astronomical
modalities, can analyze multiple types of observations together.

Different missions observe related physical phenomena but represent the
resulting observations differently. They use different data structures, file
formats, metadata systems, units, observation conventions, and preprocessing
requirements. Before AION can understand a new survey, someone must manually
build the connection between the dataset and the model.

New surveys continue to emerge while existing models often require additional
engineering before they can use them. AstralBridge explores whether AI can
automate that engineering work instead of asking every research group to solve
the same problem independently.

## Canonical Observation Architecture

AstralBridge is designed around the observation that the physical meaning of
astronomical measurements often remains the same even when different missions
store them differently. A galaxy observed by different telescopes is still
emitting the same photons. What changes is the computational representation.
AstralBridge focuses on translating those representations while preserving the
underlying meaning.

The core problem is not just file format conversion. One mission might store
photometric measurements with pixel scale and exposure time in one set of
units and field names. Another might store the same types of measurements
under entirely different names and units. Without a shared representation,
supporting N surveys across M models requires N times M custom integrations.
AstralBridge reduces this to N survey adapters and one model bridge by
introducing a shared intermediate representation.

The canonical observation layer is the architectural boundary of AstralBridge.
It is a survey independent representation that separates how astronomical data
is stored from how observations are represented for a model. Every generated
adapter targets this representation rather than AION directly, allowing
survey specific logic to remain isolated from model specific logic.

This architecture scales naturally. Without a canonical layer, supporting N
astronomical surveys across M foundation models requires N times M independent
integrations. By introducing a shared intermediate representation, each survey
only needs one adapter into the canonical layer, while each model only needs
one bridge out of it.

```text
Roman     Rubin     Euclid     JWST     Gaia
  |          |         |         |        |
  v          v         v         v        v
  [ Gemma 4 generated adapters ]
              |
              v
  Canonical Observation Layer
              |
              v
  AION modality objects  (codec encode -> transformer -> codec decode)
```

This follows a common software engineering principle: introduce a stable
intermediate representation between heterogeneous inputs and model specific
interfaces. The canonical layer is what makes the generated code meaningful
rather than arbitrary conversion logic.

Gemma 4 is not simply generating boilerplate conversion code. It is
interpreting documentation, understanding observational schemas, producing
executable adapters, validating their outputs, and iteratively repairing them
until they satisfy the expected interface.

AstralBridge intentionally leaves the pretrained AION model unchanged. Its
purpose is to extend the engineering workflow around the model rather than
modify the model itself. Existing weights, codecs, and learned representations
remain untouched. By combining a canonical observation layer with AI generated
connectors and output interpretation, AstralBridge lowers the barrier for
students, educators, and citizen scientists who want to explore astronomical
observations using modern AI.

## Why AION

AION (AstronomIcal Omnimodal Network) is a 4M encoder-decoder transformer.
4M stands for Massively Multimodal Masked Modeling, the training paradigm
inherited from EPFL and Apple's FourM framework. Instead of predicting the
next token like a GPT model, AION masks random tokens across all modalities
and learns to reconstruct them from the visible ones.

Earlier astronomy AI systems were often designed around a single type of
observation. Astronomy, however, is naturally multimodal. A single
astronomical object can be described through telescope images, flux
measurements, spectra, redshift information, coordinates, and physical
properties.

AION learns from these different observation types together. Its architecture
uses specialized codecs to transform astronomical measurements into discrete
tokens, allowing the encoder-decoder transformer to reason across different
modalities. The encoder processes input modalities, the decoder generates
target modalities, and which modalities are inputs versus targets can be
swapped at inference time.

However, like many AI systems, AION expects observations in specific formats.
AstralBridge focuses on solving that connection problem.

AION was chosen because it already provides pretrained multimodal
representations and inference capabilities. AstralBridge focuses on making
those capabilities easier to apply to future surveys without retraining the
model.

## How AstralBridge Works

The workflow follows the path from a new astronomical survey to a decoded
prediction.

### 1. Survey Understanding

A user provides survey documentation, a dataset schema, and sample
observations. These describe how a telescope mission stores and represents
its observations.

Gemma 4 analyzes this information to understand available measurements, column
meanings, units, metadata relationships, and observation structure.

### 2. AI Generated Survey Adapter

Gemma 4 generates an executable adapter that converts the new survey format
into a canonical observation. This adapter handles tasks such as identifying
the physical role of each field, mapping data fields to canonical names,
converting units, organizing observations, and preserving data lineage.

The canonical observation is a survey independent representation that holds
images, scalars, metadata, and lineage. It does not replace AION modalities.
It acts as a stable interface between heterogeneous survey formats and the
model specific modality contract.

Instead of a researcher manually writing a custom conversion pipeline,
AstralBridge produces an initial implementation automatically.

### 3. Validation and Repair

Generated code is not accepted blindly. AstralBridge executes the adapter
against sample observations and validates the result.

The system checks required fields, expected data structures, tensor dimensions,
and data compatibility. If a problem is detected, the validation feedback is
returned to Gemma 4, allowing it to revise the adapter.

This creates an iterative development workflow:

```text
Generate
   |
   v
Execute
   |
   v
Validate
   |
   v
Repair
   |
   v
Ready for inference
```

### 4. Canonical Observation to AION

After validation, AstralBridge maps the canonical observation into the existing
AION modality objects. This bridge is deterministic and conservative: it only
maps to modality types already present in AION's pretrained vocabulary, and
refuses unsupported bands or scalar names rather than silently mapping them.

AION performs inference using its pretrained architecture. AstralBridge focuses
on making new data accessible to that capability.

### 5. Interpretation with Gemma 4

Predictions are only useful when humans can understand them. After AION
produces a prediction, AstralBridge uses Gemma 4 again to generate an
explanation of the result.

The explanation connects the astronomical observation used, the model
prediction, the relevant input modalities, and the meaning of the result in
astronomical terms.

Instead of requiring every user to understand model internals, AstralBridge
creates a bridge between advanced astronomy AI and human understanding.

This creates opportunities for students learning astronomy, researchers
exploring new datasets, citizen scientists exploring observations, and
educators demonstrating modern AI astronomy workflows.

## Why Gemma 4

Gemma 4 performs two independent engineering workflows. Before inference, it
studies survey documentation, understands schemas, generates executable
adapters, and repairs them using validation feedback. After inference, it
interprets decoded predictions and translates them into structured
explanations that connect the numerical outputs back to the underlying
astronomical observations.

Together, these capabilities reduce two barriers: connecting new data to
powerful models, and understanding what those models produce.

## Installation

```bash
git clone https://github.com/hasin-codes/AstralBridge.git
cd AstralBridge
pip install -e ".[torch,gemma]"
```

The `torch` extra installs PyTorch, required for AION inference. The `gemma`
extra installs the official `google-genai` SDK for Gemma 4.

The AION model and codecs download automatically from HuggingFace on first
inference run (~1.2 GB, cached after). No manual checkpoint download needed.

Optional:

```bash
pip install -e ".[torch,gemma,dev]"   # adds pytest, ruff for development
pip install astropy                    # for FITS sample file inspection
```

## Configure Gemma 4

AstralBridge uses the official `google-genai` SDK to call Gemma 4 through the
Google AI Studio API.

1. Obtain a free API key from
   [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Set it as an environment variable:

   Linux / macOS:
   ```bash
   export GEMINI_API_KEY=YOUR_API_KEY
   ```

   Windows (PowerShell):
   ```powershell
   $env:GEMINI_API_KEY="YOUR_API_KEY"
   ```

3. (Optional) Override the default model (`gemma-4-31b-it`):
   ```bash
   export ASTRALBRIDGE_GEMMA_MODEL=gemma-4-31b-it
   ```

If the key is missing, the CLI prints a clear error explaining how to obtain
one instead of crashing.

### Alternative: local Gemma runner

If you run Gemma locally (e.g. GGUF via llama.cpp), set
`ASTRALBRIDGE_GEMMA_COMMAND` to a command that reads a prompt from stdin and
writes the completion to stdout:

```bash
export ASTRALBRIDGE_GEMMA_COMMAND="ollama run gemma3"
```

Official Gemma documentation: <https://ai.google.dev/gemma/docs>

## Usage

The `examples/` directory contains a Legacy Survey DR9 sample you can use
immediately. Supply your own docs, schema, and sample file for other surveys.

### Full pipeline (primary command)

```bash
astralbridge run \
  --survey "Legacy Survey DR9" \
  --docs examples/survey_docs.txt \
  --schema examples/survey_schema.json \
  --sample examples/sample_observation.json \
  --target redshift \
  --output-dir out/
```

AstralBridge will analyze the survey information, generate an adapter, validate
the adapter, transform observations into AION compatible inputs, run AION
inference, and generate an explanation.

This writes four artifacts to `out/`:

- `adapter.py`: the generated and validated survey adapter
- `validation_report.json`: the validation history
- `prediction.json`: the real AION prediction
- `interpretation.txt`: Gemma's explanation

### Adapter generation only

```bash
astralbridge generate \
  --survey "Legacy Survey DR9" \
  --docs examples/survey_docs.txt \
  --schema examples/survey_schema.json \
  --sample examples/sample_observation.json \
  --output-dir out/
```

### AION inference on an existing adapter

```bash
astralbridge predict \
  --adapter out/adapter.py \
  --sample examples/sample_observation.json \
  --target redshift \
  --output-dir out/ \
  --interpret
```

## Demo

The `examples/` directory contains a Legacy Survey DR9 sample (the survey AION
was trained on, so it is fully compatible). Use it to verify the pipeline runs
before supplying your own data.

### Quick demo (local)

```bash
export GEMINI_API_KEY=YOUR_API_KEY
python demo.py
```

This runs the full pipeline: Gemma generates an adapter, validation runs, AION
produces a real redshift prediction, and Gemma explains it. Output goes to
`demo_out/`.

To test adapter generation only (no AION model download needed):

```bash
python demo.py --no-inference
```

### Kaggle demo

Upload `kaggle_demo.ipynb` to Kaggle, enable GPU, add `GEMINI_API_KEY` as a
secret, and run all cells. The notebook handles installation, model download,
and the full pipeline. See [KAGGLE.md](KAGGLE.md) for details.

## Technical Architecture

```text
Astronomical Documentation
        +
Survey Schema
        +
Sample Observation
              |
              v
        Gemma 4
              |
              v
   Generated Survey Adapter
              |
              v
 Validation and Repair System
              |
              v
 Canonical Observation Layer
              |
              v
       AION Modality Layer
              |
              v
       AION Codec Manager
              |
              v
       AION Encoder-Decoder Transformer
              |
              v
      Decoded Prediction
              |
              v
        Gemma 4 Explanation
```

## Main Modules

```text
astralbridge/
  canonical/       CanonicalObservation and validation rules
  adapters/        Gemma adapter generation, validation, and repair
  gemma/           Official google-genai client and prompts
  integration/     Deterministic CanonicalObservation to AION mapping
  inference/       Real AION inference engine
  interpretation/  Constrained interpretation of AION outputs
  cli.py           Runnable CLI (run / generate / predict)

aion/
  Upstream AION model, codecs, modalities, and inference code (kept intact)
```

## Tests

```bash
python -m pytest tests/astralbridge -q
```

The AstralBridge tests cover generated adapter repair, canonical observation
validation, deterministic mapping into AION modalities, inference engine
logic, CLI wiring and error handling, and constrained interpretation prompt
usage.

A `@pytest.mark.slow` test that loads the real AION model is included but
skipped by default.

## Built With

- AION omnimodal encoder-decoder transformer
- Gemma 4
- PyTorch
- Hugging Face ecosystem
- Astronomical data processing tools

## Impact

Astronomy has made observations increasingly accessible through open data
archives, but using modern AI models on those observations still requires
significant engineering effort. AstralBridge explores a future where onboarding
new surveys becomes largely automated through AI generated adapters and a
shared canonical representation. By reducing the effort required to connect new
observations with existing foundation models, the project aims to make
advanced astronomical AI more approachable for researchers, educators,
students, and citizen scientists exploring public survey data.

## Open Source Foundation

AstralBridge builds upon the open source AION astronomy model developed by
[Polymathic AI](https://github.com/PolymathicAI). This project extends the
accessibility of AION by creating tools for connecting new astronomical
observations with its existing capabilities.
