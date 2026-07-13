# AstralBridge Documentation

AstralBridge is a Gemma-powered dataset onboarding layer built on the AION
astronomy model. It accepts arbitrary astronomy survey products,
generates executable adapter code with Gemma 4, validates the adapter, runs real
AION inference, and explains the prediction with Gemma.

The entire pipeline is input-driven. A judge or researcher supplies their own
survey documentation, schema, and sample observations; AstralBridge does the
rest. No built-in datasets are used.

## What To Read First

- [AstralBridge architecture](astralbridge.md)
- [API reference](api.rst)
- [Kaggle setup](KAGGLE.md)

## Core Flow

1. The user provides survey documentation, schema text, and a sample
   observation file.
2. Gemma 4 reads the documentation, schema, and sample structure, then
   generates an executable survey adapter.
3. AstralBridge validates the adapter and its manifest. If validation fails,
   errors are returned to Gemma for repair until the adapter passes or the
   attempt limit is reached.
4. The validated adapter creates a `CanonicalObservation` from the sample file.
5. The bridge maps that observation into existing AION modality objects.
6. The inference engine runs real AION inference: `CodecManager.encode` →
   `AION.forward` → `CodecManager.decode`.
7. Gemma explains the real decoded prediction without changing the value.

## Non-goals

AstralBridge does not create new AION modalities dynamically, retrain AION,
alter pretrained codecs, or turn AION into a natural-language agent.

## Gemma Integration

AstralBridge uses the official `google-genai` SDK to call Gemma 4
through the Google AI Studio API. Obtain a free API key from
[Google AI Studio](https://aistudio.google.com/app/apikey) and set it as the
`GEMINI_API_KEY` environment variable. Official Gemma documentation:
<https://ai.google.dev/gemma/docs>.

```{toctree}
:hidden:
:maxdepth: 2

astralbridge
api
```
