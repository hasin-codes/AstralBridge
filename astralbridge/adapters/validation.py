"""Adapter validation and runtime compatibility checks."""

from __future__ import annotations

import importlib.util
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astralbridge.canonical.validators import (
    ValidationIssue,
    ValidationReport,
    validate_canonical_observation,
)


@dataclass
class AdapterValidationResult:
    """Result from compiling and running a generated adapter."""

    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    adapter_path: str | None = None
    manifest: dict[str, Any] = field(default_factory=dict)

    def as_text(self) -> str:
        if self.passed:
            return "PASSED"
        report = ValidationReport(passed=False, issues=self.issues)
        return report.as_text()

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable validation report for demos and CI artifacts."""

        return {
            "passed": self.passed,
            "issues": [
                {"code": issue.code, "message": issue.message, "path": issue.path}
                for issue in self.issues
            ],
            "manifest": self.manifest,
        }


class AdapterValidator:
    """Compile and validate generated adapters against AION bridge contracts."""

    def __init__(self, require_aion_bridge: bool = True) -> None:
        self.require_aion_bridge = require_aion_bridge

    def validate_code(self, code: str, sample_input: Any) -> AdapterValidationResult:
        """Validate generated code using a local temporary module."""

        try:
            compile(code, "<generated_adapter>", "exec")
        except SyntaxError as exc:
            return AdapterValidationResult(
                passed=False,
                issues=[
                    ValidationIssue(
                        "syntax_error",
                        f"{exc.msg} on line {exc.lineno}",
                    )
                ],
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_path = Path(temp_dir) / "generated_adapter.py"
            adapter_path.write_text(code, encoding="utf-8")

            try:
                spec = importlib.util.spec_from_file_location(
                    "generated_adapter", adapter_path
                )
                if spec is None or spec.loader is None:
                    raise RuntimeError("Could not load generated adapter module")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                manifest = getattr(module, "ADAPTER_MANIFEST", {})
                manifest_issues = self._validate_manifest(manifest)
                if manifest_issues:
                    return AdapterValidationResult(
                        passed=False,
                        adapter_path=str(adapter_path),
                        issues=manifest_issues,
                        manifest=manifest if isinstance(manifest, dict) else {},
                    )
                adapter_cls = getattr(module, "GeneratedSurveyAdapter")
                adapter = adapter_cls()
                observation = adapter.convert(sample_input)
            except Exception as exc:
                return AdapterValidationResult(
                    passed=False,
                    adapter_path=str(adapter_path),
                    issues=[
                        ValidationIssue(
                            "runtime_error",
                            f"{exc.__class__.__name__}: {exc}",
                        )
                    ],
                )

            canonical_report = validate_canonical_observation(observation)
            if not canonical_report.passed:
                return AdapterValidationResult(
                    passed=False,
                    adapter_path=str(adapter_path),
                    issues=canonical_report.issues,
                    manifest=manifest,
                )

            if not self.require_aion_bridge:
                return AdapterValidationResult(
                    passed=True,
                    adapter_path=str(adapter_path),
                    issues=[],
                    manifest=manifest,
                )

            try:
                from astralbridge.integration.aion_bridge import canonical_to_aion

                canonical_to_aion(observation)
            except Exception as exc:
                return AdapterValidationResult(
                    passed=False,
                    adapter_path=str(adapter_path),
                    issues=[
                        ValidationIssue(
                            "aion_bridge_error",
                            f"{exc.__class__.__name__}: {exc}",
                        )
                    ],
                    manifest=manifest,
                )

            return AdapterValidationResult(
                passed=True,
                adapter_path=str(adapter_path),
                issues=[],
                manifest=manifest,
            )

    def _validate_manifest(self, manifest: Any) -> list[ValidationIssue]:
        """Validate adapter metadata that proves what the generated code maps."""

        if not isinstance(manifest, dict):
            return [
                ValidationIssue(
                    "invalid_manifest",
                    "Generated adapter must define ADAPTER_MANIFEST as a dict",
                )
            ]

        required = {
            "survey",
            "source_fields",
            "target_aion_modalities",
            "transformations",
        }
        missing = sorted(key for key in required if not manifest.get(key))
        if missing:
            return [
                ValidationIssue(
                    "incomplete_manifest",
                    f"ADAPTER_MANIFEST is missing required keys: {missing}",
                )
            ]

        if not isinstance(manifest["source_fields"], list):
            return [
                ValidationIssue(
                    "invalid_manifest_source_fields",
                    "ADAPTER_MANIFEST['source_fields'] must be a list",
                )
            ]

        if not isinstance(manifest["target_aion_modalities"], list):
            return [
                ValidationIssue(
                    "invalid_manifest_targets",
                    "ADAPTER_MANIFEST['target_aion_modalities'] must be a list",
                )
            ]

        if not isinstance(manifest["transformations"], list):
            return [
                ValidationIssue(
                    "invalid_manifest_transformations",
                    "ADAPTER_MANIFEST['transformations'] must be a list",
                )
            ]

        return []
