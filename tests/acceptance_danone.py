"""Live acceptance against the completed Danone run.

Needs the real output root, a running PostgreSQL, and the configured model server with the
configured models. Skips cleanly when any of those is missing.

    docker compose up -d
    ollama pull qwen3-embedding:4b
    python tests/test_danone_acceptance.py

The first run ingests 494 pages and takes a while; later runs reuse the ready
version because the fingerprint is unchanged.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


import psycopg
from claim_evidence import ClaimEvidence, Settings
from claim_evidence.model_client import ModelClient
from claim_evidence.models import EvidenceQuality, Verdict

OUTPUT_ROOT = Path(
    os.environ.get(
        "CLAIM_EVIDENCE_DANONE_ROOT",
        r"C:\Users\Tobia\Documents\Tobi&Anna\gw_detector_v2\outputs_full_run\danoneurdaccessible",
    )
)
SOURCE_PDF = Path(
    os.environ.get(
        "CLAIM_EVIDENCE_DANONE_PDF",
        r"C:\Users\Tobia\Documents\Tobi&Anna\gw_detector_v2\input\danoneurdaccessible.pdf",
    )
)
EXPECTED_PAGE = 359

# The reporting entity is stated explicitly: version 1 never infers it from a
# filename.
ENTITY = "Danone S.A."
SUPPORTED = (
    "Danone reduced Scope 1 and 2 energy and industry emissions by 40.2% in 2025 versus 2020."
)
CONTRADICTED = SUPPORTED.replace("40.2%", "90%")
INSUFFICIENT = "Danone reduced all carbon emissions by 90% from 2020 to 2025."


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def preflight(settings: Settings) -> str | None:
    if not (OUTPUT_ROOT / "manifest.json").is_file():
        return f"output root not present: {OUTPUT_ROOT}"
    try:
        psycopg.connect(settings.database_url, connect_timeout=5).close()
    except psycopg.OperationalError as exc:
        return f"postgres unavailable ({exc.__class__.__name__}); run: docker compose up -d"
    reachable, models, problems = ModelClient(settings).health()
    if not reachable:
        return f"{settings.model_backend} unavailable"
    if not all(model.available for model in models):
        return problems[0] if problems else "a configured model is unavailable"
    return None


def report_citations(result) -> None:
    for cite in result.citations:
        roles = ",".join(r.role for r in cite.regions)
        print(
            f"      p.{cite.pdf_page} {cite.source_kind} {cite.quality} "
            f"{cite.geometry_precision} regions=[{roles}]"
        )
        if cite.table_cells:
            print(f"        cells: {' | '.join(cite.table_cells)}")


def main() -> int:
    settings = Settings.from_env()
    skip = preflight(settings)
    if skip:
        print(f"[skip] {skip}")
        return 0

    with ClaimEvidence(settings) as client:
        client.init_db()

        started = time.monotonic()
        report = client.ingest_document(
            OUTPUT_ROOT,
            source_pdf=SOURCE_PDF,
            source_uri="https://www.danone.com/urd-2025",
            # The acceptance claims are table-backed; narrative extraction is
            # one LLM call per claim-like block and is exercised separately.
            extract_narrative_facts=False,
            reporting_entity=ENTITY,
        )
        print(
            f"ingested {report.pages} pages, {report.evidence_units} units, "
            f"{report.embedded_units} embedded, {report.facts} facts "
            f"in {time.monotonic() - started:.0f}s "
            f"({'reused' if report.reused_existing else 'built'})"
        )
        check(report.pages == 494, "all 494 pages indexed")

        print("\n-- 1. supported")
        supported = client.audit_claim(SUPPORTED, scope="all", reporting_entity=ENTITY)
        print(f"      {supported.verdict}: {supported.rationale}")
        report_citations(supported)
        check(supported.verdict is Verdict.SUPPORTED, "exact 40.2% claim is supported")
        check(
            supported.evidence_quality
            in (EvidenceQuality.DIRECT_TABLE, EvidenceQuality.DIRECT_TEXT),
            "supported by direct evidence",
        )
        pages = {c.pdf_page for c in supported.citations}
        check(EXPECTED_PAGE in pages, f"cites PDF page {EXPECTED_PAGE} (got {sorted(pages)})")
        table_cite = next(
            (c for c in supported.citations if c.pdf_page == EXPECTED_PAGE and c.table_cells),
            None,
        )
        check(table_cite is not None, "page 359 citation carries table cells")
        roles = {r.role for r in table_cite.regions}
        check(
            {"descriptor", "header", "value"} <= roles,
            f"descriptor, header, and value regions returned ({roles})",
        )
        check(
            any("40.2" in cell for cell in table_cite.table_cells),
            "the cited cells contain the displayed value",
        )

        print("\n-- 2. contradicted")
        contradicted = client.audit_claim(CONTRADICTED, scope="all", reporting_entity=ENTITY)
        print(f"      {contradicted.verdict}: {contradicted.rationale}")
        report_citations(contradicted)
        check(contradicted.verdict is Verdict.CONTRADICTED, "the 90% claim is contradicted")
        check(
            EXPECTED_PAGE in {c.pdf_page for c in contradicted.citations},
            f"contradiction cites PDF page {EXPECTED_PAGE}",
        )

        print("\n-- 3. insufficient")
        insufficient = client.audit_claim(INSUFFICIENT, scope="all", reporting_entity=ENTITY)
        print(f"      {insufficient.verdict}: {insufficient.rationale}")
        report_citations(insufficient)
        check(
            insufficient.verdict is Verdict.INSUFFICIENT,
            f"'all carbon emissions' is not forced into a verdict ({insufficient.verdict})",
        )
        check(
            insufficient.verdict is not Verdict.CONTRADICTED,
            "a vague scope never becomes a contradiction",
        )

    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
