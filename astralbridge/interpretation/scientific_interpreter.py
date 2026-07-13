"""Gemma-backed interpretation for AION outputs.

The interpreter consumes the real decoded prediction produced by AION and
asks Gemma to explain it in concise prose. Gemma does not change the
predicted value and must not invent confidence scores or extra measurements.
"""

from __future__ import annotations


from astralbridge.gemma.client import GemmaClient
from astralbridge.gemma.prompts import build_interpretation_prompt


class ScientificInterpreter:
    """Generate constrained prose from real model outputs."""

    def __init__(self, client: GemmaClient) -> None:
        self.client = client

    def interpret(
        self,
        prediction_value: str,
        target: str,
        modalities: dict[str, int],
        lineage: dict,
    ) -> str:
        """Ask Gemma to explain a real AION output without changing it.

        Args:
            prediction_value: The decoded prediction value from AION (e.g.
                ``"0.341"`` for redshift). This is the real model output, not
                a placeholder.
            target: Human-friendly target name (e.g. ``"redshift"``).
            modalities: Input modality token keys mapped to token counts.
            lineage: Data lineage from the canonical observation.
        """

        prompt = build_interpretation_prompt(
            prediction_value=prediction_value,
            target=target,
            modalities=modalities,
            lineage=lineage,
        )
        return self.client.complete(prompt).strip()

    def interpret_result(self, result, observation) -> str:
        """Interpret a ``PredictionResult`` produced by the inference engine.

        This is the primary entry point for the end-to-end pipeline. It passes
        the *real* decoded prediction value (not a placeholder) to Gemma along
        with the token counts and data lineage from the observation.

        Args:
            result: A ``PredictionResult`` from ``AionInferenceEngine.predict``.
            observation: The ``CanonicalObservation`` that was predicted from.
        """

        value = result.predicted_value
        # Normalize tensors to a readable string.
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass
        if hasattr(value, "tolist"):
            value = value.tolist()
        return self.interpret(
            prediction_value=str(value),
            target=result.target,
            modalities=result.token_counts,
            lineage=observation.token_lineage(),
        )
