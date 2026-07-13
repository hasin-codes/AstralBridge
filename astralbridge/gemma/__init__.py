"""Gemma client interface and prompt builders.

The primary client uses the official ``google-generativeai`` SDK to call Gemma 4
through the Google AI Studio API. See ``astralbridge.gemma.client`` for details.
"""

from astralbridge.gemma.client import (
    CommandGemmaClient,
    GeminiClient,
    GemmaConfigError,
    GemmaClient,
    StaticGemmaClient,
    make_gemma_client,
)

__all__ = [
    "CommandGemmaClient",
    "GeminiClient",
    "GemmaConfigError",
    "GemmaClient",
    "StaticGemmaClient",
    "make_gemma_client",
]
