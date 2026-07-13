"""Command line entry points for AstralBridge.

The CLI provides three commands that together cover the complete end-to-end
pipeline, each individually reachable:

    astralbridge run       Full pipeline: generate adapter -> validate ->
                           convert to AION modalities -> run AION inference ->
                           interpret with Gemma.

    astralbridge generate  Generate and validate a survey adapter only.

    astralbridge predict   Run AION inference on an existing validated adapter.

All commands are input-driven: the user supplies their own survey
documentation, schema, and sample data. No built-in datasets are used.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read(path: str | None) -> str:
    """Read a text file, returning an empty string if path is None."""

    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8")


def _summarize_sample(path: str) -> str:
    """Build a human-readable summary of a sample data file.

    The summary is included in the Gemma prompt so the model understands the
    structure of the data it must write an adapter for. Supports JSON and FITS
    (if astropy is installed); other files are described by their first bytes.
    """

    sample_path = Path(path)
    suffix = sample_path.suffix.lower()

    if suffix == ".json":
        try:
            data = json.loads(sample_path.read_text(encoding="utf-8"))
            return _summarize_json(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    if suffix in (".fits", ".fit"):
        try:
            from astropy.io import fits

            with fits.open(sample_path) as hdul:
                lines = [f"FITS file with {len(hdul)} HDU(s):"]
                for index, hdu in enumerate(hdul):
                    name = hdu.name or f"HDU {index}"
                    shape = hdu.data.shape if hdu.data is not None else "()"
                    dtype = (
                        hdu.data.dtype.name if hdu.data is not None else "empty"
                    )
                    lines.append(f"  {name}: shape {shape}, dtype {dtype}")
                    if hasattr(hdu, "columns"):
                        cols = ", ".join(hdu.columns.names)
                        lines.append(f"    columns: {cols}")
                return "\n".join(lines)
        except ImportError:
            return (
                f"FITS file ({sample_path.name}). Install astropy "
                "(pip install astropy) for detailed inspection."
            )

    # Fallback: read first bytes as text.
    try:
        text = sample_path.read_text(encoding="utf-8")[:2000]
        return f"Text file ({sample_path.name}):\n{text}"
    except UnicodeDecodeError:
        return f"Binary file ({sample_path.name}, {sample_path.stat().st_size} bytes)."


def _summarize_json(data: Any, depth: int = 0, max_depth: int = 3) -> str:
    """Summarize a JSON-compatible structure concisely."""

    indent = "  " * depth
    if isinstance(data, dict):
        if depth >= max_depth:
            return f"{indent}{{...}} ({len(data)} keys)"
        lines = [f"{indent}{{"]
        for key, value in data.items():
            lines.append(f"{indent}  {key}: {_describe_value(value, depth + 1)}")
        lines.append(f"{indent}}}")
        return "\n".join(lines)
    return _describe_value(data, depth)


def _describe_value(value: Any, depth: int) -> str:
    """Describe a single JSON value's type and shape."""

    if isinstance(value, list):
        shape = _list_shape(value)
        return f"list, shape {shape}"
    if isinstance(value, dict):
        return f"dict with {len(value)} keys"
    if isinstance(value, (int, float)):
        return f"{type(value).__name__}, value {value}"
    if isinstance(value, bool):
        return f"bool, value {value}"
    if value is None:
        return "null"
    return f"{type(value).__name__}"


def _list_shape(value: Any) -> tuple[int, ...]:
    """Compute the nested shape of a list-of-lists."""

    dims = []
    current = value
    while isinstance(current, list):
        dims.append(len(current))
        current = current[0] if current else []
    return tuple(dims)


def _load_adapter_module(adapter_path: Path):
    """Import a saved adapter module from a file path.

    Returns the loaded module object.
    """

    spec = importlib.util.spec_from_file_location("generated_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load adapter module from {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_adapter(adapter_path: Path, sample_path: str):
    """Load and run a saved adapter against a sample file.

    Returns the ``CanonicalObservation`` produced by the adapter.
    """

    module = _load_adapter_module(adapter_path)
    adapter_cls = getattr(module, "GeneratedSurveyAdapter", None)
    if adapter_cls is None:
        raise RuntimeError(
            "Adapter does not define 'GeneratedSurveyAdapter'. "
            "The generated adapter must define this class."
        )
    adapter = adapter_cls()
    observation = adapter.convert(sample_path)
    return observation


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_validation_report(path: Path, result) -> None:
    """Write the adapter validation report as JSON."""

    payload = {
        "passed": result.passed,
        "attempts": [
            {"attempt": index, **item.as_dict()}
            for index, item in enumerate(result.validation_history, start=1)
        ],
    }
    _write_json(path, payload)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Run the complete end-to-end pipeline.

    This is the primary command. It generates an adapter with Gemma, validates
    it, converts the observation into AION modalities, runs real AION
    inference, and produces a Gemma interpretation of the result.
    """

    docs = _read(args.docs)
    schema = _read(args.schema)
    sample_path = args.sample

    if not Path(sample_path).exists():
        print(f"Error: sample file not found: {sample_path}", file=sys.stderr)
        return 1

    sample_summary = _summarize_sample(sample_path)

    # --- Build the generation request from user-supplied inputs ---
    request_args = {
        "survey_name": args.survey,
        "documentation_text": docs,
        "schema_text": schema,
        "sample_summary": sample_summary,
        "user_instructions": args.instructions or "",
    }

    # Late import so --help works without the full dependency stack.
    from astralbridge.adapters import AdapterGenerationRequest, AdapterGenerator
    from astralbridge.adapters.validation import AdapterValidator
    from astralbridge.gemma import GemmaConfigError, make_gemma_client

    request = AdapterGenerationRequest(**request_args)

    # --- Obtain a real Gemma client ---
    try:
        client = make_gemma_client()
    except GemmaConfigError as exc:
        print(f"Gemma configuration error:\n{exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("AstralBridge: Full survey onboarding pipeline")
    print(f"{'=' * 55}")
    print(f"Survey: {args.survey}")
    print(f"Target: {args.target}")
    print()

    # --- Step 1: Generate adapter with Gemma 4 ---
    print("[1/5] Generating survey adapter with Gemma 4...")
    validator = AdapterValidator(require_aion_bridge=True)
    generator = AdapterGenerator(
        client=client, validator=validator, max_attempts=args.max_attempts
    )
    result = generator.generate(request, sample_path)

    for index, item in enumerate(result.validation_history, start=1):
        print(f"  Attempt {index}: {'PASSED' if item.passed else 'FAILED'}")
        if not item.passed:
            print(f"    {item.as_text()}")

    if not result.passed:
        print("\nAdapter validation failed after all repair attempts.", file=sys.stderr)
        _write_validation_report(output_dir / "validation_report.json", result)
        adapter_path = output_dir / "adapter.py"
        adapter_path.write_text(result.code, encoding="utf-8")
        print(f"  Last adapter code written to {adapter_path}", file=sys.stderr)
        print(f"  Validation report written to {output_dir / 'validation_report.json'}")
        return 1

    # --- Step 2: Save validated adapter ---
    adapter_path = output_dir / "adapter.py"
    adapter_path.write_text(result.code, encoding="utf-8")
    _write_validation_report(output_dir / "validation_report.json", result)
    print(f"  Adapter validated and written to {adapter_path}")

    # --- Step 3: Convert observation to AION modalities ---
    print("\n[2/5] Running adapter on sample data...")
    from astralbridge.integration.aion_bridge import canonical_to_aion

    observation = _run_adapter(adapter_path, sample_path)
    print(f"  CanonicalObservation: {len(observation.images)} image(s), "
          f"{len(observation.scalars)} scalar(s)")

    print("\n[3/5] Converting to AION modality objects...")
    modalities = canonical_to_aion(observation)
    modality_names = [type(m).__name__ for m in modalities]
    print(f"  Modalities: {', '.join(modality_names)}")

    # --- Step 4: Run real AION inference ---
    print("\n[4/5] Running AION inference...")
    from astralbridge.inference import AionInferenceEngine

    engine = AionInferenceEngine(device=args.device)
    prediction = engine.predict_by_name(modalities, args.target)

    value = prediction.predicted_value
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    print(f"  Predicted {prediction.target}: {value}")

    _write_json(output_dir / "prediction.json", prediction.as_dict())

    # --- Step 5: Interpret with Gemma 4 ---
    print("\n[5/5] Generating interpretation with Gemma 4...")
    from astralbridge.interpretation import ScientificInterpreter

    interpreter = ScientificInterpreter(client)
    interpretation = interpreter.interpret_result(prediction, observation)
    (output_dir / "interpretation.txt").write_text(interpretation, encoding="utf-8")
    print(f"  {interpretation}")

    # --- Summary ---
    print(f"\n{'=' * 55}")
    print(f"Pipeline complete. Artifacts written to {output_dir}/:")
    print("  - adapter.py")
    print("  - validation_report.json")
    print("  - prediction.json")
    print("  - interpretation.txt")

    return 0


def generate(args: argparse.Namespace) -> int:
    """Generate and validate a survey adapter without running AION inference."""

    docs = _read(args.docs)
    schema = _read(args.schema)
    sample_path = args.sample

    if not Path(sample_path).exists():
        print(f"Error: sample file not found: {sample_path}", file=sys.stderr)
        return 1

    sample_summary = _summarize_sample(sample_path)

    from astralbridge.adapters import AdapterGenerationRequest, AdapterGenerator
    from astralbridge.adapters.validation import AdapterValidator
    from astralbridge.gemma import GemmaConfigError, make_gemma_client

    request = AdapterGenerationRequest(
        survey_name=args.survey,
        documentation_text=docs,
        schema_text=schema,
        sample_summary=sample_summary,
        user_instructions=args.instructions or "",
    )

    try:
        client = make_gemma_client()
    except GemmaConfigError as exc:
        print(f"Gemma configuration error:\n{exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("AstralBridge: Adapter generation")
    print(f"{'=' * 40}")
    print(f"Survey: {args.survey}")
    print()

    print("Generating survey adapter with Gemma 4...")
    validator = AdapterValidator(require_aion_bridge=False)
    generator = AdapterGenerator(
        client=client, validator=validator, max_attempts=args.max_attempts
    )
    result = generator.generate(request, sample_path)

    for index, item in enumerate(result.validation_history, start=1):
        print(f"  Attempt {index}: {'PASSED' if item.passed else 'FAILED'}")
        if not item.passed:
            print(f"    {item.as_text()}")

    adapter_path = output_dir / "adapter.py"
    adapter_path.write_text(result.code, encoding="utf-8")
    _write_validation_report(output_dir / "validation_report.json", result)

    print(f"\nAdapter written to {adapter_path}")
    print(f"Validation report written to {output_dir / 'validation_report.json'}")

    return 0 if result.passed else 1


def predict(args: argparse.Namespace) -> int:
    """Run AION inference on an existing validated adapter."""

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"Error: adapter file not found: {adapter_path}", file=sys.stderr)
        return 1

    sample_path = args.sample
    if not Path(sample_path).exists():
        print(f"Error: sample file not found: {sample_path}", file=sys.stderr)
        return 1

    from astralbridge.integration.aion_bridge import canonical_to_aion
    from astralbridge.inference import AionInferenceEngine

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("AstralBridge: AION inference")
    print(f"{'=' * 40}")
    print(f"Target: {args.target}")
    print()

    print("Running adapter on sample data...")
    observation = _run_adapter(adapter_path, sample_path)
    print(f"  CanonicalObservation: {len(observation.images)} image(s), "
          f"{len(observation.scalars)} scalar(s)")

    print("Converting to AION modality objects...")
    modalities = canonical_to_aion(observation)
    modality_names = [type(m).__name__ for m in modalities]
    print(f"  Modalities: {', '.join(modality_names)}")

    print("Running AION inference...")
    engine = AionInferenceEngine(device=args.device)
    prediction = engine.predict_by_name(modalities, args.target)

    value = prediction.predicted_value
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    print(f"  Predicted {prediction.target}: {value}")

    _write_json(output_dir / "prediction.json", prediction.as_dict())

    if args.interpret:
        from astralbridge.gemma import GemmaConfigError, make_gemma_client
        from astralbridge.interpretation import ScientificInterpreter

        print("\nGenerating interpretation with Gemma 4...")
        try:
            client = make_gemma_client()
        except GemmaConfigError as exc:
            print(f"Gemma configuration error:\n{exc}", file=sys.stderr)
            return 1
        interpreter = ScientificInterpreter(client)
        interpretation = interpreter.interpret_result(prediction, observation)
        (output_dir / "interpretation.txt").write_text(interpretation, encoding="utf-8")
        print(f"  {interpretation}")

    print(f"\nPrediction written to {output_dir / 'prediction.json'}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astralbridge",
        description=(
            "AstralBridge: Gemma-powered survey onboarding for the AION "
            "astronomy foundation model. Input-driven: supply your own survey "
            "documentation, schema, and sample data."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_parser = subparsers.add_parser(
        "run",
        help="Full pipeline: generate adapter, validate, run AION inference, "
        "and interpret with Gemma.",
    )
    run_parser.add_argument("--survey", required=True, help="Survey name")
    run_parser.add_argument("--docs", help="Path to survey documentation (text)")
    run_parser.add_argument("--schema", help="Path to survey schema (text/JSON)")
    run_parser.add_argument(
        "--sample", required=True, help="Path to a sample observation file"
    )
    run_parser.add_argument(
        "--target",
        default="redshift",
        help="Prediction target (default: redshift)",
    )
    run_parser.add_argument("--instructions", help="Optional user instructions for Gemma")
    run_parser.add_argument(
        "--output-dir", default="out", help="Directory for output artifacts (default: out)"
    )
    run_parser.add_argument(
        "--device", default="auto", help="Device: auto, cuda, or cpu (default: auto)"
    )
    run_parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max adapter generation/repair attempts (default: 3)",
    )
    run_parser.set_defaults(func=run)

    # --- generate ---
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate and validate a survey adapter (no AION inference).",
    )
    gen_parser.add_argument("--survey", required=True, help="Survey name")
    gen_parser.add_argument("--docs", help="Path to survey documentation (text)")
    gen_parser.add_argument("--schema", help="Path to survey schema (text/JSON)")
    gen_parser.add_argument(
        "--sample", required=True, help="Path to a sample observation file"
    )
    gen_parser.add_argument("--instructions", help="Optional user instructions for Gemma")
    gen_parser.add_argument(
        "--output-dir", default="out", help="Directory for output artifacts (default: out)"
    )
    gen_parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max adapter generation/repair attempts (default: 3)",
    )
    gen_parser.set_defaults(func=generate)

    # --- predict ---
    pred_parser = subparsers.add_parser(
        "predict",
        help="Run AION inference on an existing validated adapter.",
    )
    pred_parser.add_argument(
        "--adapter", required=True, help="Path to a saved adapter .py file"
    )
    pred_parser.add_argument(
        "--sample", required=True, help="Path to a sample observation file"
    )
    pred_parser.add_argument(
        "--target",
        default="redshift",
        help="Prediction target (default: redshift)",
    )
    pred_parser.add_argument(
        "--output-dir", default="out", help="Directory for output artifacts (default: out)"
    )
    pred_parser.add_argument(
        "--device", default="auto", help="Device: auto, cuda, or cpu (default: auto)"
    )
    pred_parser.add_argument(
        "--interpret",
        action="store_true",
        help="Also generate a Gemma interpretation of the prediction",
    )
    pred_parser.set_defaults(func=predict)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
