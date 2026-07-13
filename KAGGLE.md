# Running AstralBridge on Kaggle

AstralBridge runs end to end on Kaggle with a GPU accelerator. The AION Base
model (300M) fits comfortably in a Kaggle T4 GPU. This guide walks through the
full setup.

## Option A: Use the demo notebook (easiest)

Upload `kaggle_demo.ipynb` (included in the repo) to Kaggle, enable GPU, add
`GEMINI_API_KEY` as a secret, and run all cells. The notebook handles
installation, model download, and the full pipeline automatically.

## Option B: Manual setup

### 1. Create a Kaggle Notebook

1. Go to [kaggle.com](https://www.kaggle.com/) and create a new notebook.
2. In the right sidebar, set **Accelerator** to **GPU T4 x2** (or P100).
3. Set **Internet** to **On** (needed for package installs and HuggingFace
   downloads).

### 2. Add Kaggle Secrets

Add the following secrets via **Add-ons → Secrets** (or the notebook's
"Add input" → "Secrets"):

| Secret name          | Required | Description                                              |
|----------------------|----------|----------------------------------------------------------|
| `GEMINI_API_KEY`     | Yes      | Free API key from Google AI Studio.                      |
| `ASTRALBRIDGE_GEMMA_MODEL` | No  | Override the default model (`gemma-4-31b-it`).           |
| `HF_TOKEN`           | No       | HuggingFace token, if the AION download needs auth.      |

Obtain a free Gemma API key at
[https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

### 3. Install AstralBridge

Clone the repository and install with the required extras:

```python
!git clone https://github.com/Hasin-Raiyan/AstralBridge.git /kaggle/working/AstralBridge
%cd /kaggle/working/AstralBridge
!pip install -e .[torch,gemma] -q
!pip install astropy -q
```

### 4. Unpack Secrets into Environment Variables

```python
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["GEMINI_API_KEY"] = secrets.get_secret("GEMINI_API_KEY")
# Optional:
# os.environ["ASTRALBRIDGE_GEMMA_MODEL"] = secrets.get_secret("ASTRALBRIDGE_GEMMA_MODEL")
# os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
```

### 5. Use the Example Data or Your Own

The repo includes example Legacy Survey data in `examples/`. To use your own
survey, upload your documentation, schema, and sample observation file to the
notebook.

### 6. Run the Full Pipeline

```bash
!astralbridge run \
  --survey "Legacy Survey DR9" \
  --docs /kaggle/working/AstralBridge/examples/survey_docs.txt \
  --schema /kaggle/working/AstralBridge/examples/survey_schema.json \
  --sample /kaggle/working/AstralBridge/examples/sample_observation.json \
  --target redshift \
  --output-dir /kaggle/working/out/ \
  --device cuda
```

### 7. Expected Output

The pipeline runs the complete end-to-end flow:

1. Gemma 4 generates a survey adapter from your documentation, schema, and
   sample data.
2. The validation loop compiles, executes, and checks the adapter. If it
   fails, errors go back to Gemma for repair.
3. The validated adapter produces a `CanonicalObservation`.
4. AstralBridge maps it into existing AION modality objects.
5. The AION model runs a real forward pass and decodes a real prediction
   (e.g. a redshift value) through the pretrained codec codebook.
6. Gemma explains the prediction in concise astronomical terms.

Artifacts are written to the output directory:

- `adapter.py` — the generated and validated adapter.
- `validation_report.json` — validation history.
- `prediction.json` — the real AION prediction.
- `interpretation.txt` — Gemma's explanation of the prediction.

## Notes

- The first run downloads the AION model (~1.2 GB) and codecs from
  HuggingFace. Subsequent runs use the cached files.
- If you encounter a CUDA out-of-memory error, ensure no other process is
  using the GPU. The Base model (300M) fits in 4 GB of VRAM.
- The default Gemma model is `gemma-4-31b-it`. Override it with
  `ASTRALBRIDGE_GEMMA_MODEL` if your API key has access to a different variant.
- Official Gemma documentation: <https://ai.google.dev/gemma/docs>
