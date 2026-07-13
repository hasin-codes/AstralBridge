"""Tests for the AstralBridge CLI.

These tests verify argument parsing, error handling, and pipeline wiring
without requiring a GPU, the AION model, or a Gemma API key. Heavy components
are stubbed.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest

from astralbridge.cli import build_parser


def test_parser_has_run_generate_predict():
    """The CLI should expose run, generate, and predict subcommands."""

    parser = build_parser()

    for command in ("run", "generate", "predict"):
        with patch.object(sys, "argv", ["astralbridge", command, "--help"]):
            with pytest.raises(SystemExit):
                parser.parse_args()


def test_run_requires_sample(tmp_path):
    """run should error if the sample file does not exist."""

    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--survey", "Test",
            "--sample", str(tmp_path / "nonexistent.json"),
            "--target", "redshift",
        ]
    )

    exit_code = args.func(args)
    assert exit_code == 1


def test_run_missing_gemma_key_errors_clearly(tmp_path, sample_json_file, monkeypatch):
    """run should print a clear error if no Gemma configuration is set."""

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ASTRALBRIDGE_GEMMA_COMMAND", raising=False)

    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--survey", "Test",
            "--sample", str(sample_json_file),
            "--target", "redshift",
            "--output-dir", str(tmp_path / "out"),
        ]
    )

    exit_code = args.func(args)
    assert exit_code == 1


def test_generate_missing_gemma_key_errors_clearly(tmp_path, sample_json_file, monkeypatch):
    """generate should print a clear error if no Gemma configuration is set."""

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ASTRALBRIDGE_GEMMA_COMMAND", raising=False)

    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--survey", "Test",
            "--sample", str(sample_json_file),
            "--output-dir", str(tmp_path / "out"),
        ]
    )

    exit_code = args.func(args)
    assert exit_code == 1


def test_summarize_json_sample(tmp_path):
    """_summarize_sample should produce a readable JSON structure summary."""

    from astralbridge.cli import _summarize_sample

    data = {"image": [[[[0.0]]]], "flux_g": 2.841, "name": "test"}
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    summary = _summarize_sample(str(path))
    assert "flux_g" in summary
    assert "list" in summary  # the image is a nested list


def test_predict_missing_adapter_errors(tmp_path, sample_json_file):
    """predict should error if the adapter file does not exist."""

    parser = build_parser()
    args = parser.parse_args(
        [
            "predict",
            "--adapter", str(tmp_path / "nonexistent.py"),
            "--sample", str(sample_json_file),
        ]
    )

    exit_code = args.func(args)
    assert exit_code == 1
