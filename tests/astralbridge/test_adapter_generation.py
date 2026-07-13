"""Tests for Gemma-powered adapter generation and the repair loop."""

from astralbridge.adapters import AdapterGenerationRequest, AdapterGenerator
from astralbridge.gemma import StaticGemmaClient

from tests.astralbridge.conftest import (
    broken_adapter_code,
    valid_file_path_adapter_code,
)


def test_adapter_generation_repairs_failed_adapter(sample_json_file):
    """Gemma should repair a broken adapter after validation feedback."""

    request = AdapterGenerationRequest(
        survey_name="Roman WFI demo",
        documentation_text="demo docs",
        schema_text="demo schema",
        sample_summary="demo sample",
    )
    client = StaticGemmaClient(
        responses=[broken_adapter_code(), valid_file_path_adapter_code()]
    )
    generator = AdapterGenerator(client=client, max_attempts=2)

    result = generator.generate(request, str(sample_json_file))

    assert result.passed
    assert len(result.validation_history) == 2
    assert not result.validation_history[0].passed
    assert result.validation_history[1].passed
    assert result.validation_history[1].manifest["survey"] == "Roman WFI demo"
    assert len(client.calls) == 2


def test_adapter_generation_passes_first_try(sample_json_file):
    """A valid adapter on the first attempt should pass immediately."""

    request = AdapterGenerationRequest(
        survey_name="Roman WFI demo",
        documentation_text="demo docs",
        schema_text="demo schema",
        sample_summary="demo sample",
    )
    client = StaticGemmaClient(responses=[valid_file_path_adapter_code()])
    generator = AdapterGenerator(client=client, max_attempts=3)

    result = generator.generate(request, str(sample_json_file))

    assert result.passed
    assert len(result.validation_history) == 1
    assert len(client.calls) == 1
