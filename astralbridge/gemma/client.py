"""Gemma client abstractions.

The primary client uses the official Google ``google-genai`` SDK to call Gemma 4
through the Google AI Studio API. An alternative command-based client is
provided for local Gemma runners (e.g. GGUF via llama.cpp). Tests use
``StaticGemmaClient`` so the repository remains reproducible without network
credentials.

Official documentation:
    - Gemma docs:        https://ai.google.dev/gemma/docs
    - Python SDK docs:   https://googleapis.github.io/python-genai/

API keys are obtained (free) from Google AI Studio:
    https://aistudio.google.com/app/apikey
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Protocol


# Default Gemma 4 model available through the Google AI Studio API. Override
# with the ASTRALBRIDGE_GEMMA_MODEL environment variable if a different Gemma
# variant is available in your account.
DEFAULT_GEMMA_MODEL = "gemma-4-31b-it"


class GemmaConfigError(RuntimeError):
    """Raised when Gemma is not configured and cannot be used."""


class GemmaClient(Protocol):
    """Protocol for Gemma-backed text generation."""

    def complete(self, prompt: str) -> str:
        """Return a model completion for a prompt."""
        ...


@dataclass
class StaticGemmaClient:
    """Deterministic client for examples and tests.

    Returns pre-seeded responses in order. This is a test-only client and is
    never used in the production pipeline.
    """

    responses: list[str]
    calls: list[str] = field(default_factory=list)

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self.responses:
            raise RuntimeError("StaticGemmaClient has no remaining responses")
        return self.responses.pop(0)


class GeminiClient:
    """Primary Gemma client using the official ``google-genai`` SDK.

    Uses Google's current official SDK to call Gemma 4 through the Google AI
    Studio API::

        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model="gemma-4-31b-it", contents=prompt
        )

    The SDK is imported lazily so the rest of AstralBridge can be used without
    it installed.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise GemmaConfigError(
                "GEMINI_API_KEY is not set. Obtain a free API key from "
                "Google AI Studio at https://aistudio.google.com/app/apikey, "
                "then set it as an environment variable:\n"
                "  Linux/macOS:  export GEMINI_API_KEY=YOUR_API_KEY\n"
                "  Windows:      $env:GEMINI_API_KEY=\"YOUR_API_KEY\"\n"
                "Alternatively, set ASTRALBRIDGE_GEMMA_COMMAND to use a local "
                "Gemma runner. See the README for details."
            )

        model_name = model or os.environ.get(
            "ASTRALBRIDGE_GEMMA_MODEL", DEFAULT_GEMMA_MODEL
        )

        try:
            from google import genai
        except ImportError as exc:
            raise GemmaConfigError(
                "The official Google SDK 'google-genai' is not "
                "installed. Install it with: pip install -e .[gemma]\n"
                "Documentation: https://ai.google.dev/gemma/docs"
            ) from exc

        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def complete(self, prompt: str) -> str:
        """Generate a completion for the given prompt via Gemma 4."""

        response = self._client.models.generate_content(
            model=self._model_name, contents=prompt
        )
        return response.text


@dataclass
class CommandGemmaClient:
    """Alternative client that runs Gemma through an external command.

    The command receives the prompt on stdin and must write the completion to
    stdout. Set ASTRALBRIDGE_GEMMA_COMMAND to use this from the CLI. This is
    intended for local Gemma runners (e.g. GGUF via llama.cpp) or Kaggle-native
    runners that are not exposed through the Google AI Studio API.
    """

    command: str
    timeout_seconds: int = 120

    @classmethod
    def from_env(cls) -> "CommandGemmaClient":
        command = os.environ.get("ASTRALBRIDGE_GEMMA_COMMAND")
        if not command:
            raise GemmaConfigError(
                "ASTRALBRIDGE_GEMMA_COMMAND is not set. Set it to a command "
                "that reads a prompt from stdin and writes the Gemma "
                "completion to stdout. Alternatively, set GEMINI_API_KEY to "
                "use the official Google AI Studio API."
            )
        return cls(command=command)

    def complete(self, prompt: str) -> str:
        result = subprocess.run(
            self.command,
            input=prompt,
            text=True,
            capture_output=True,
            shell=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Gemma command failed")
        return result.stdout


def make_gemma_client() -> GeminiClient | CommandGemmaClient:
    """Build the appropriate Gemma client from the environment.

    Resolution order:
        1. ``GEMINI_API_KEY`` set  -> ``GeminiClient`` (official SDK, primary)
        2. ``ASTRALBRIDGE_GEMMA_COMMAND`` set -> ``CommandGemmaClient``
        3. Neither set -> ``GemmaConfigError`` with setup instructions.

    This is the single entry point the CLI uses to obtain a Gemma client.
    """

    if os.environ.get("GEMINI_API_KEY"):
        return GeminiClient()
    if os.environ.get("ASTRALBRIDGE_GEMMA_COMMAND"):
        return CommandGemmaClient.from_env()
    raise GemmaConfigError(
        "No Gemma configuration found. Either:\n"
        "  1. Set GEMINI_API_KEY (free key from "
        "https://aistudio.google.com/app/apikey) to use the official Google "
        "AI Studio API, OR\n"
        "  2. Set ASTRALBRIDGE_GEMMA_COMMAND to a local Gemma runner command "
        "that reads a prompt from stdin and writes the completion to stdout.\n"
        "See the README for full setup instructions."
    )
