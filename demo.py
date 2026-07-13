#!/usr/bin/env python
"""Local demo runner for AstralBridge.

Runs the full end-to-end pipeline using the example Legacy Survey data. Use
this to verify the installation works before supplying your own survey data.

Usage:
    python demo.py                    # Full pipeline with example data
    python demo.py --no-inference     # Adapter generation + validation only

Requirements:
    - GEMINI_API_KEY environment variable set (free key from https://aistudio.google.com/app/apikey)
    - PyTorch installed (pip install -e .[torch])
    - google-genai installed (pip install -e .[gemma])
    - Internet access (AION model downloads from HuggingFace on first run)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="AstralBridge local demo")
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Skip AION inference (adapter generation + validation only)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cuda, or cpu (default: auto)",
    )
    args = parser.parse_args()

    # --- Check prerequisites ---
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY is not set.")
        print("Get a free key at https://aistudio.google.com/app/apikey")
        print("Then run: export GEMINI_API_KEY=YOUR_KEY")
        return 1

    examples_dir = Path(__file__).parent / "examples"
    docs_path = examples_dir / "survey_docs.txt"
    schema_path = examples_dir / "survey_schema.json"
    sample_path = examples_dir / "sample_observation.json"

    if not all(p.exists() for p in [docs_path, schema_path, sample_path]):
        print("ERROR: Example data not found in examples/ directory.")
        return 1

    output_dir = Path("demo_out")
    output_dir.mkdir(exist_ok=True)

    print("=" * 55)
    print("AstralBridge Demo — Legacy Survey DR9 Example")
    print("=" * 55)

    # --- Load and show sample data ---
    with open(sample_path) as f:
        sample = json.load(f)
    print(f"\nSample survey: {sample['survey']}")
    print(f"Observation ID: {sample['observation_id']}")
    image = sample["image"]
    print(f"Image shape: [{len(image)}, {len(image[0])}, {len(image[0][0])}]")
    print(f"Flux g/r/i/z: {sample['flux_g']}, {sample['flux_r']}, {sample['flux_i']}, {sample['flux_z']}")

    # --- Step 1: Generate adapter with Gemma 4 ---
    print("\n" + "=" * 55)
    print("[1] Generating survey adapter with Gemma 4...")
    print("=" * 55)

    from astralbridge.adapters import AdapterGenerationRequest, AdapterGenerator
    from astralbridge.adapters.validation import AdapterValidator
    from astralbridge.gemma import make_gemma_client

    request = AdapterGenerationRequest(
        survey_name="Legacy Survey DR9",
        documentation_text=docs_path.read_text(encoding="utf-8"),
        schema_text=schema_path.read_text(encoding="utf-8"),
        sample_summary=_summarize(sample),
        user_instructions="Map Legacy Survey g/r/i/z bands to AION DES bands and predict redshift.",
    )

    client = make_gemma_client()
    validator = AdapterValidator(require_aion_bridge=not args.no_inference)
    generator = AdapterGenerator(client=client, validator=validator, max_attempts=3)

    start = time.time()
    result = generator.generate(request, str(sample_path))
    elapsed = time.time() - start

    for i, item in enumerate(result.validation_history, 1):
        status = "PASSED" if item.passed else "FAILED"
        print(f"  Attempt {i}: {status}")
        if not item.passed:
            print(f"    {item.as_text()}")

    print(f"\nAdapter generation took {elapsed:.1f}s")

    # --- Save adapter + report ---
    (output_dir / "adapter.py").write_text(result.code, encoding="utf-8")
    _write_report(output_dir / "validation_report.json", result)
    print(f"Adapter saved to {output_dir / 'adapter.py'}")

    if not result.passed:
        print("\nAdapter validation failed. Cannot proceed to inference.")
        return 1

    if args.no_inference:
        print("\n--no-inference: skipping AION inference. Demo complete.")
        return 0

    # --- Step 2: Run adapter on sample ---
    print("\n" + "=" * 55)
    print("[2] Running adapter on sample data...")
    print("=" * 55)

    import importlib.util

    spec = importlib.util.spec_from_file_location("adapter", output_dir / "adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observation = module.GeneratedSurveyAdapter().convert(str(sample_path))
    print(f"CanonicalObservation: {len(observation.images)} image(s), {len(observation.scalars)} scalar(s)")

    # --- Step 3: Convert to AION modalities ---
    print("\n" + "=" * 55)
    print("[3] Converting to AION modality objects...")
    print("=" * 55)

    from astralbridge.integration import canonical_to_aion

    modalities = canonical_to_aion(observation)
    names = [type(m).__name__ for m in modalities]
    print(f"Modalities: {', '.join(names)}")

    # --- Step 4: Download AION model if needed, then run inference ---
    print("\n" + "=" * 55)
    print("[4] Running AION inference...")
    print("=" * 55)

    print("Loading AION model (downloads ~1.2 GB on first run)...")
    start = time.time()

    from astralbridge.inference import AionInferenceEngine

    engine = AionInferenceEngine(device=args.device)
    prediction = engine.predict_by_name(modalities, "redshift")

    elapsed = time.time() - start
    value = prediction.predicted_value
    if hasattr(value, "item"):
        value = value.item()
    print(f"Model loaded + inference in {elapsed:.1f}s")
    print(f"\n  >>> Predicted redshift: z = {value} <<<")

    _write_json(output_dir / "prediction.json", prediction.as_dict())

    # --- Step 5: Interpret with Gemma ---
    print("\n" + "=" * 55)
    print("[5] Generating interpretation with Gemma 4...")
    print("=" * 55)

    from astralbridge.interpretation import ScientificInterpreter

    interpreter = ScientificInterpreter(client)
    interpretation = interpreter.interpret_result(prediction, observation)
    (output_dir / "interpretation.txt").write_text(interpretation, encoding="utf-8")
    print(interpretation)

    # --- Summary ---
    print("\n" + "=" * 55)
    print("Demo complete. Artifacts in demo_out/:")
    print("=" * 55)
    print("  adapter.py")
    print("  validation_report.json")
    print(f"  prediction.json  (z = {value})")
    print("  interpretation.txt")
    return 0


def _summarize(sample: dict) -> str:
    image = sample["image"]
    return (
        f"JSON file with fields: survey, observation_id, image, flux_g, "
        f"flux_r, flux_i, flux_z, ebv, ra, dec. "
        f"Image is a nested list with shape [{len(image)}, {len(image[0])}, {len(image[0][0])}], "
        f"representing 4 optical bands (g,r,i,z) of 96x96 pixels each. "
        f"Flux values are floats in nanomaggies."
    )


def _write_report(path: Path, result) -> None:
    _write_json(path, {
        "passed": result.passed,
        "attempts": [
            {"attempt": i, **item.as_dict()}
            for i, item in enumerate(result.validation_history, 1)
        ],
    })


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
