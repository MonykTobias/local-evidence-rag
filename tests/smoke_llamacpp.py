"""Opt-in read-only smoke check for live llama.cpp servers."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from claim_evidence import Settings
from claim_evidence.facts import FACT_EXTRACTION_SYSTEM
from claim_evidence.model_client import ModelClient
from claim_evidence.models import FactExtraction, GeometryPrecision, Region, RegionRole
from claim_evidence.vision import verify_visual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crop", type=Path, help="PNG or other Pillow-readable image")
    parser.add_argument(
        "--claim",
        default="The image contains legible information.",
        help="claim used for the visual request",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    if settings.model_backend != "llamacpp":
        raise SystemExit("set CLAIM_EVIDENCE_MODEL_BACKEND=llamacpp first")
    if not args.crop.is_file():
        raise SystemExit("crop does not exist or is not a file")

    client = ModelClient(settings)
    started = time.monotonic()
    reachable, models, problems = client.health()
    if not reachable or not all(model.available for model in models):
        raise SystemExit(problems[0] if problems else "a configured model is unavailable")
    print(f"health ok ({time.monotonic() - started:.2f}s)")

    started = time.monotonic()
    vectors = client.embed(
        [
            "Scope 1 and 2 emissions",
            "renewable electricity share",
            "water consumption in 2025",
            "climate footprint against baseline",
        ]
    )
    print(
        f"embeddings ok: {len(vectors)} x {len(vectors[0])} "
        f"({time.monotonic() - started:.2f}s)"
    )

    started = time.monotonic()
    client.structured(
        FactExtraction,
        FACT_EXTRACTION_SYSTEM,
        "<passage>Example Corp reduced emissions by 12% in 2025.</passage>",
    )
    print(f"structured chat ok ({time.monotonic() - started:.2f}s)")

    started = time.monotonic()
    result = verify_visual(
        client,
        args.crop,
        [
            Region(
                bbox=(0.0, 0.0, 1.0, 1.0),
                role=RegionRole.VISUAL_REGION,
                precision=GeometryPrecision.CROP,
            )
        ],
        args.claim,
    )
    if result.reason_code in {"crop_unavailable", "vision_unavailable"}:
        raise SystemExit(f"visual check failed: {result.reason_code}")
    print(f"vision ok: {result.result} ({time.monotonic() - started:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
