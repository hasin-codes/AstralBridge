"""Gemma-powered adapter generation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from astralbridge.adapters.validation import AdapterValidator, AdapterValidationResult
from astralbridge.gemma.client import GemmaClient
from astralbridge.gemma.prompts import build_adapter_prompt, build_repair_prompt


@dataclass
class AdapterGenerationRequest:
    """Inputs Gemma needs to generate a survey adapter."""

    survey_name: str
    documentation_text: str
    schema_text: str
    sample_summary: str
    user_instructions: str = ""


@dataclass
class GeneratedAdapter:
    """Final generated adapter and validation history."""

    code: str
    validation_history: list[AdapterValidationResult]

    @property
    def passed(self) -> bool:
        return bool(self.validation_history and self.validation_history[-1].passed)


def _extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    return (match.group(1) if match else text).strip()


class AdapterGenerator:
    """Generate, validate, and repair survey adapters with Gemma."""

    def __init__(
        self,
        client: GemmaClient,
        validator: AdapterValidator | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.client = client
        self.validator = validator or AdapterValidator()
        self.max_attempts = max_attempts

    def generate(
        self, request: AdapterGenerationRequest, sample_input
    ) -> GeneratedAdapter:
        """Generate an adapter and repair it until validation passes or stops."""

        prompt = build_adapter_prompt(request)
        code = _extract_python_code(self.client.complete(prompt))
        history: list[AdapterValidationResult] = []

        for attempt in range(self.max_attempts):
            result = self.validator.validate_code(code, sample_input)
            history.append(result)
            if result.passed:
                break
            if attempt == self.max_attempts - 1:
                break
            repair_prompt = build_repair_prompt(request, code, result.as_text())
            code = _extract_python_code(self.client.complete(repair_prompt))

        return GeneratedAdapter(code=code, validation_history=history)
