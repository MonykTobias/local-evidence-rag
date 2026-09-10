"""End-to-end checks against the Compose PostgreSQL, with a fake Ollama.

Skips cleanly when the database is unreachable, so the deterministic suite
still runs on a machine with no Docker.

    docker compose up -d
    python tests/test_integration.py
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any


import psycopg
from fake_ollama import FakeSession
from fixtures import block, image_summary, kpi_table, write_output_root

from claim_evidence import ClaimEvidence, Settings
from claim_evidence.db import (
    SCHEMA_VERSION,
    connect,
    exact_vector_search,
    normalize_embedding,
    vector_search,
)
from claim_evidence.models import (
    EvidenceKind,
    EvidenceQuality,
    GeometryPrecision,
    Verdict,
    VersionStatus,
)
from claim_evidence.audit import MAX_PASSAGES, PASSAGE_CHARS
from claim_evidence.errors import (
    IndexNotReadyError,
    NotFoundError,
    ValidationError,
)
from claim_evidence.ingest import identity_key
from claim_evidence.ollama import OllamaClient

ADMIN_URL = os.environ.get(
    "CLAIM_EVIDENCE_DATABASE_URL",
    "postgresql://claim_evidence:claim_evidence@localhost:5433/claim_evidence",
)
TEST_DB = "claim_evidence_test"
DIMENSIONS = 8

SUPPORTED = "Danone reduced Scope 1 and 2 energy and industry emissions by 40.2% in 2025 versus 2020."
CONTRADICTED = SUPPORTED.replace("40.2%", "90%")
VAGUE = "Danone reduced all carbon emissions by 90% from 2020 to 2025."
# The reporting entity is stated explicitly: version 1 never infers it
# from a filename.
ENTITY = "Danone S.A."


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


# --- pytest-visible activation checks ---------------------------------------
#
# The rest of this file is a sequenced end-to-end script driven by main(). These
# few are written as independent pytest cases because activation is the gate the
# lean plan verifies by name, and a gate is worth being able to run on its own.


def _database_available() -> bool:
    try:
        with psycopg.connect(ADMIN_URL, connect_timeout=3):
            return True
    except psycopg.OperationalError:
        return False


def _ingest_into_a_fresh_database(root: Path):
    import pytest

    if not _database_available():
        pytest.skip("PostgreSQL is not reachable; run `docker compose up -d`")
    reset_database()
    client = make_client(default_session())
    client.init_db()
    return client


def test_activation_refuses_a_missing_evidence_artifact(tmp_path: Path) -> None:
    """A citation the product cannot show is not a citation."""
    from claim_evidence.ingest import IngestionError, _verify_artifacts

    root = build_root(tmp_path / "missing-artifact")
    client = _ingest_into_a_fresh_database(root)
    try:
        report = client.ingest_document(root, source_uri="urn:missing-artifact", reporting_entity=ENTITY)
        assert report.status is VersionStatus.READY

        # The artifact a stored unit names disappears -- a partially deleted
        # output directory, which is the realistic way this happens.
        (root / "blocks.jsonl").unlink()
        try:
            _verify_artifacts(client.conn, report.version_id, Path(root))
        except IngestionError as exc:
            assert "blocks.jsonl" in str(exc), exc
        else:
            raise AssertionError("a missing evidence artifact must block activation")
    finally:
        client.close()


def test_a_document_without_narrative_blocks_cites_no_narrative(tmp_path: Path) -> None:
    """Absent source data produces no evidence, never an unresolvable citation."""
    root = write_output_root(
        tmp_path / "no-blocks", pages=1, blocks=[], tables={1: [kpi_table()]}
    )
    client = _ingest_into_a_fresh_database(root)
    try:
        report = client.ingest_document(root, source_uri="urn:no-blocks", reporting_entity=ENTITY)
        assert report.status is VersionStatus.READY
        narrative = client.conn.execute(
            "SELECT count(*) AS n FROM evidence_unit"
            " WHERE version_id = %s AND kind = 'narrative'",
            (report.version_id,),
        ).fetchone()["n"]
        assert narrative == 0, "no narrative units without narrative blocks"
    finally:
        client.close()


def test_activation_refuses_an_artifact_outside_the_extraction_root(
    tmp_path: Path,
) -> None:
    """Containment is checked before activation, not at render time."""
    from claim_evidence.ingest import IngestionError
    from claim_evidence.source import OutputReader, page_units

    root = build_root(tmp_path / "escaping")
    client = _ingest_into_a_fresh_database(root)
    try:
        report = client.ingest_document(root, source_uri="urn:escaping", reporting_entity=ENTITY)
        assert report.status is VersionStatus.READY

        # Point one stored unit at a file outside the root and re-run the gate
        # directly: this is the state a tampered or buggy producer would leave.
        from claim_evidence.ingest import _verify_artifacts

        client.conn.execute(
            """
            UPDATE evidence_unit SET artifact_path = %s
            WHERE id = (SELECT min(id) FROM evidence_unit WHERE version_id = %s
                        AND citable)
            """,
            ("../outside/secrets.txt", report.version_id),
        )
        client.conn.commit()
        try:
            _verify_artifacts(client.conn, report.version_id, Path(root))
        except IngestionError as exc:
            assert "escape" in str(exc)
            assert "secrets.txt" not in str(exc), (
                "the escaping path is not echoed back to the caller"
            )
        else:
            raise AssertionError("an escaping artifact path must be refused")
    finally:
        client.close()


# --- harness ----------------------------------------------------------------


def database_url() -> str:
    return ADMIN_URL.rsplit("/", 1)[0] + f"/{TEST_DB}"


def reset_database() -> None:
    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{TEST_DB}"')


def settings() -> Settings:
    return Settings(
        database_url=database_url(),
        embed_dimensions=DIMENSIONS,
        embed_batch_size=16,
        chat_model="fake-chat",
        vision_model="fake-vision",
    )


def parsed_claim_reply(payload: dict[str, Any]) -> dict[str, Any]:
    """A deliberately thin parse, so the heuristic gap-fill is exercised."""
    claim = payload["messages"][1]["content"]
    scope = None
    if "Scope 1 and 2" in claim:
        scope = "Scope 1 and 2 energy and industry emissions"
    elif "all carbon" in claim:
        scope = "all carbon emissions"
    return {"subject": "Danone", "metric": claim, "scope": scope, "direction": "decrease"}


def adjudication_reply(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": "insufficient",
        "rationale": "The evidence does not state a comparable scope.",
        "supporting_evidence_ids": [],
        "missing_qualifiers": ["scope"],
    }


def build_root(root: Path, *, with_visual: bool = False) -> Path:
    blocks = [
        block(1, 1, "4.8.2 NATURE INDICATORS", kind="section_header",
              heading_path=["4.8.2 NATURE INDICATORS"]),
        block(1, 2, "The table below reports performance against the 2020 baseline.",
              heading_path=["4.8.2 NATURE INDICATORS"]),
        block(2, 1, "Water withdrawal intensity improved during the year.",
              heading_path=["4.8.3 WATER"]),
    ]
    return write_output_root(
        root,
        pages=2,
        blocks=blocks,
        tables={1: [kpi_table()]},
        images={2: [image_summary(2, 1, "Chart of emissions reduction")]} if with_visual else None,
    )


def make_client(session: FakeSession, conn: psycopg.Connection | None = None) -> ClaimEvidence:
    config = settings()
    return ClaimEvidence(
        config,
        conn or connect(config.database_url),
        OllamaClient(config, session),
    )


def default_session(**router: Any) -> FakeSession:
    return FakeSession(
        dimensions=DIMENSIONS,
        chat_router={
            "ParsedClaim": parsed_claim_reply,
            "FactExtraction": lambda _: {"facts": []},
            "Adjudication": adjudication_reply,
            "VisualVerification": lambda _: {
                "result": "illegible",
                "reason_code": "figures_not_legible",
            },
            **router,
        },
    )


# --- checks -----------------------------------------------------------------


def check_schema_is_repeatable(tmp: Path) -> None:
    with make_client(default_session()) as client:
        client.init_db()
        client.init_db()
        tables = {
            row["tablename"]
            for row in client.conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ).fetchall()
        }
        expected = {
            "document", "document_version", "page", "evidence_unit", "evidence_region",
            "entity", "entity_alias", "fact", "fact_evidence", "audit_run",
            "audit_candidate",
        }
        check(expected <= tables, f"all 11 domain tables present ({sorted(expected - tables)})")
        version = client.conn.execute("SELECT version FROM schema_meta").fetchone()
        check(version["version"] == SCHEMA_VERSION, "schema version recorded")


def check_ingestion_is_idempotent(tmp: Path) -> None:
    root = build_root(tmp / "run")
    with make_client(default_session()) as client:
        first = client.ingest_document(root, source_pdf=None, source_uri="urn:test", reporting_entity=ENTITY)
        check(first.status is VersionStatus.READY, "version reaches ready")
        check(first.pages == 2, "both pages indexed")
        check(first.evidence_units > 0, "evidence units stored")
        check(first.embedded_units > 0, "embeddings written")
        check(first.facts >= 4, f"table facts derived ({first.facts})")

        second = client.ingest_document(root, source_pdf=None, source_uri="urn:test", reporting_entity=ENTITY)
        check(second.reused_existing, "re-ingesting the same fingerprint is a no-op")
        check(second.version_id == first.version_id, "the same version is reused")
        check(
            second.evidence_units == first.evidence_units and second.facts == first.facts,
            "a no-op run reports the stored counts, not zeros",
        )

        units = client.conn.execute(
            "SELECT count(*) AS n FROM evidence_unit WHERE version_id = %s",
            (first.version_id,),
        ).fetchone()["n"]
        check(units == first.evidence_units, "no duplicate rows after a second run")


def check_interrupted_build_is_invisible(tmp: Path) -> None:
    root = build_root(tmp / "interrupted")
    config = settings()
    conn = connect(config.database_url)
    with make_client(default_session(), conn) as client:
        # Simulate a crash between "building" and activation.
        from claim_evidence.db import start_version, upsert_document, upsert_page, upsert_evidence
        from claim_evidence.ingest import identity_key
        from claim_evidence.source import OutputReader, page_units

        document_id = upsert_document(
            conn, "interrupted.pdf", "deadbeef", None, identity_key(root, "deadbeef", None)
        )
        version_id = start_version(
            conn, document_id, "never-finished", embed_model="fake",
            embed_dim=DIMENSIONS, output_root=str(root), source_pdf=None,
        )
        reader = OutputReader(root)
        page = reader.validate()[0]
        page_id = upsert_page(conn, version_id, 1, page.width, page.height, page.rel)
        upsert_evidence(
            conn, version_id, page_id,
            list(page_units(reader, page, reader.blocks_by_page().get(1, []))),
        )
        conn.commit()

        status = conn.execute(
            "SELECT status FROM document_version WHERE id = %s", (version_id,)
        ).fetchone()["status"]
        check(status == "building", "interrupted version stays in building")

        matches = client.search_evidence("Scope 1 and 2 emissions", limit=50)
        leaked = [m for m in matches if m.citation.document_name == "interrupted.pdf"]
        check(not leaked, "a building version never appears in query results")


def check_indexes_are_used(tmp: Path) -> None:
    with make_client(default_session()) as client:
        conn = client.conn
        conn.execute("SET enable_seqscan = off")
        fts = "\n".join(
            str(r)
            for r in conn.execute(
                """
                EXPLAIN SELECT id FROM evidence_unit
                WHERE text_search @@ websearch_to_tsquery('english', 'emissions')
                """
            ).fetchall()
        )
        check("evidence_unit_fts_idx" in fts, "GIN full-text index is used")

        vector = "\n".join(
            str(r)
            for r in conn.execute(
                "EXPLAIN SELECT id FROM evidence_unit WHERE embedding IS NOT NULL "
                "ORDER BY embedding <#> %s LIMIT 10",
                (normalize_embedding([1.0] * DIMENSIONS),),
            ).fetchall()
        )
        check("evidence_unit_embedding_idx" in vector, "HNSW index is used for ordering")
        conn.execute("SET enable_seqscan = on")


def check_hybrid_search_finds_exact_and_semantic(tmp: Path) -> None:
    with make_client(default_session()) as client:
        matches = client.search_evidence(SUPPORTED, limit=25)
        check(bool(matches), "hybrid search returns candidates")
        texts = [m.text for m in matches]
        check(any("(40.2) %" in t for t in texts), "the exact value is retrieved")
        check(
            any(m.lexical_rank is not None for m in matches)
            and any(m.vector_rank is not None for m in matches),
            "both lexical and vector retrievers contribute",
        )
        top = matches[0].citation
        check(top.pdf_page in (1, 2), "citation carries a 1-based pdf page")
        check(bool(top.regions), "every citation carries at least one region")
        # A claim says "Danone reduced ..."; the table never uses those words.
        # AND'ing the metric terms silently removed this leg from the fusion.
        check(
            any(m.graph_rank is not None for m in matches),
            "the graph leg contributes on a claim worded unlike the source",
        )


def check_vector_recall(tmp: Path) -> None:
    config = settings()
    with make_client(default_session()) as client:
        conn = client.conn
        session = FakeSession(dimensions=DIMENSIONS)
        queries = [
            "Scope 1 and 2 emissions versus 2020",
            "renewable electricity in the energy mix",
            "water withdrawal intensity",
            "performance history 2025",
            "nature indicators table",
        ]
        hits = 0
        total = 0
        for query in queries:
            embedding = session.vector(query)
            approximate = [r["id"] for r in vector_search(conn, embedding, None, 10)]
            exact = [r["id"] for r in exact_vector_search(conn, embedding, None, 10)]
            if not exact:
                continue
            hits += len(set(approximate) & set(exact))
            total += len(exact)
        recall = hits / total if total else 1.0
        check(recall >= 0.9, f"approximate top-10 recall is {recall:.0%} (>= 90%)")
        del config


def check_supported_claim_cites_the_table(tmp: Path) -> None:
    with make_client(default_session()) as client:
        result = client.audit_claim(SUPPORTED, scope="all", reporting_entity=ENTITY)
        check(result.verdict is Verdict.SUPPORTED, f"exact claim supported ({result.rationale})")
        check(
            result.evidence_quality is EvidenceQuality.DIRECT_TABLE,
            "supported by direct table evidence",
        )
        check(bool(result.citations), "verdict carries citations")
        cite = result.citations[0]
        check(cite.source_kind is EvidenceKind.TABLE_VALUE, "the cited unit is a table value")
        check(cite.geometry_precision is GeometryPrecision.CELL, "cell-precision geometry")
        roles = {r.role for r in cite.regions}
        check(
            {"descriptor", "header", "unit", "value"} <= roles,
            f"descriptor, header, unit, and value regions cited ({roles})",
        )
        check("(40.2) %" in " ".join(cite.table_cells), "the displayed value is quoted")
        check(result.audit_id is not None, "audit persisted")

        candidates = client.conn.execute(
            "SELECT count(*) AS n FROM audit_candidate WHERE audit_id = %s",
            (result.audit_id,),
        ).fetchone()["n"]
        check(candidates > 0, "retrieval candidates persisted for the audit trace")
        selected = client.conn.execute(
            "SELECT count(*) AS n FROM audit_candidate WHERE audit_id = %s AND selected",
            (result.audit_id,),
        ).fetchone()["n"]
        check(selected > 0, "selected citations flagged in the trace")


def check_adjudicator_prompt_is_bounded(tmp: Path) -> None:
    """An over-long prompt is truncated by the runtime with no signal, so the
    passage list is capped before it is sent."""
    seen: list[str] = []

    def capture(payload: dict[str, Any]) -> dict[str, Any]:
        seen.append(payload["messages"][1]["content"])
        return adjudication_reply(payload)

    with make_client(default_session(Adjudication=capture)) as client:
        client.audit_claim(VAGUE, scope="all", reporting_entity=ENTITY)
    check(bool(seen), "the adjudicator was consulted")
    import re

    passages = re.findall(r"<evidence [^>]*>\n(.*?)\n</evidence>", seen[0], re.S)
    check(bool(passages), "the prompt carries delimited evidence passages")
    check(len(passages) <= MAX_PASSAGES, f"passage count capped ({len(passages)})")
    longest = max(len(p) for p in passages)
    check(longest <= PASSAGE_CHARS + 200, f"each passage is truncated ({longest} chars)")


SPLIT_CLAIM = "Danone cut plastic packaging by 12.5% against its 2020 baseline."
# One sentence, printed as two boxes. Neither block contains the claim, so no
# amount of direct retrieval can return a passage that states it; the refined
# Markdown put it back together.
SPLIT_BLOCKS = [
    "Danone cut plastic packaging by 12.5%",
    "against its 2020 baseline.",
]
SPLIT_MARKDOWN = f"{SPLIT_BLOCKS[0]} {SPLIT_BLOCKS[1]}\n"


def _prompt_ids(payload: dict[str, Any], tag: str = "evidence") -> list[int]:
    import re

    return [
        int(i)
        for i in re.findall(rf'<{tag} id="(\d+)"', payload["messages"][1]["content"])
    ]


def check_mapped_markdown_bridges_two_blocks(tmp: Path) -> None:
    """Generated Markdown may join evidence; the citations stay the originals.

    This is the whole point of indexing final Markdown at all. Direct evidence
    comes back insufficient because the sentence exists in no single block, the
    mapped segment offers the two blocks together, and what gets cited is those
    two blocks and their own regions -- never the Markdown.
    """
    import re

    root = write_output_root(
        tmp / "split",
        pages=1,
        blocks=[block(1, index, text) for index, text in enumerate(SPLIT_BLOCKS, 1)],
        markdown={1: SPLIT_MARKDOWN},
    )
    prompts: list[str] = []

    def adjudicate(payload: dict[str, Any]) -> dict[str, Any]:
        prompts.append(payload["messages"][1]["content"])
        ids = _prompt_ids(payload)
        if "<page_context" not in payload["messages"][1]["content"]:
            return adjudication_reply(payload)
        return {
            "verdict": "supported",
            "rationale": "The two passages together state the reduction.",
            "supporting_evidence_ids": ids,
            "missing_qualifiers": [],
        }

    with make_client(default_session(Adjudication=adjudicate)) as client:
        report = client.ingest_document(
            root, source_uri="urn:split", reporting_entity=ENTITY
        )
        segments = client.conn.execute(
            "SELECT source_text, table_context FROM evidence_unit"
            " WHERE version_id = %s AND kind = %s",
            (report.version_id, str(EvidenceKind.PAGE_MARKDOWN)),
        ).fetchall()
        check(len(segments) == 1, f"one mapped Markdown segment indexed ({len(segments)})")
        check(
            len(segments[0]["table_context"]["sources"]) == 2,
            "and it names both blocks it was generated from",
        )

        # The segment is the strongest lexical match for the claim -- it is the
        # only indexed text that states it -- and it still may not be an answer.
        matches = client.search_evidence(SPLIT_CLAIM, document_ids=[report.document_id])
        check(bool(matches), "public search returns something for the claim")
        kinds = {m.citation.source_kind for m in matches}
        check(
            EvidenceKind.PAGE_MARKDOWN not in kinds,
            f"and never generated page Markdown ({kinds})",
        )
        one = client.search_evidence(
            SPLIT_CLAIM, document_ids=[report.document_id], limit=1
        )
        check(
            len(one) == 1
            and one[0].citation.source_kind is not EvidenceKind.PAGE_MARKDOWN,
            "a strong Markdown match cannot spend the caller's only result",
        )

        result = client.audit_claim(
            SPLIT_CLAIM, scope=[report.document_id], reporting_entity=ENTITY
        )

    check(len(prompts) == 2, f"direct evidence was tried first ({len(prompts)} prompts)")
    check("<page_context" not in prompts[0], "the first prompt is direct evidence only")
    check("<page_context" in prompts[1], "the fallback prompt carries mapped context")
    check(
        re.search(r'<page_context[^>]*\bid=', prompts[1]) is None,
        "the context block has no id, so it cannot be cited",
    )
    check(
        SPLIT_MARKDOWN.strip() in prompts[1],
        "the joined sentence is what the context contributes",
    )

    kinds = {c.source_kind for c in result.citations}
    check(
        result.verdict is Verdict.SUPPORTED and len(result.citations) == 2,
        f"the joined context resolved the claim ({result.verdict}, {len(result.citations)})",
    )
    check(kinds == {EvidenceKind.NARRATIVE}, f"both citations are original blocks ({kinds})")
    check(
        all(c.regions and c.artifact_path.endswith("blocks.jsonl") for c in result.citations),
        "each cites its own block region and artifact",
    )
    check(
        result.evidence_quality is EvidenceQuality.DIRECT_TEXT,
        f"quality is the original evidence's, not the Markdown's ({result.evidence_quality})",
    )


def check_unmapped_markdown_cannot_answer(tmp: Path) -> None:
    """A sentence the page never printed is not indexed, so it cannot be read."""
    root = write_output_root(
        tmp / "rewritten",
        pages=1,
        blocks=[block(1, 1, SPLIT_BLOCKS[0])],
        # The refinement asserted a baseline no block states.
        markdown={1: f"{SPLIT_BLOCKS[0]} against its 2020 baseline.\n"},
    )
    prompts: list[str] = []

    def adjudicate(payload: dict[str, Any]) -> dict[str, Any]:
        prompts.append(payload["messages"][1]["content"])
        return adjudication_reply(payload)

    with make_client(default_session(Adjudication=adjudicate)) as client:
        report = client.ingest_document(
            root, source_uri="urn:rewritten", reporting_entity=ENTITY
        )
        segments = client.conn.execute(
            "SELECT count(*) AS n FROM evidence_unit WHERE version_id = %s AND kind = %s",
            (report.version_id, str(EvidenceKind.PAGE_MARKDOWN)),
        ).fetchone()["n"]
        check(segments == 0, "partly-anchored Markdown is not indexed at all")
        result = client.audit_claim(
            SPLIT_CLAIM, scope=[report.document_id], reporting_entity=ENTITY
        )

    check(len(prompts) == 1, f"there was no fallback to fall back to ({len(prompts)})")
    check(
        result.verdict is Verdict.INSUFFICIENT and not result.citations,
        f"the rewritten sentence moved nothing ({result.verdict})",
    )


# A table the refinement rebuilt: no block is the row, and the two cells came
# from two different sentences. Direct retrieval can return either sentence and
# neither answers the claim.
ROW_CLAIM = "Home deliveries increased 48% in FY24 versus FY16."
ROW_BLOCKS = [
    "Home deliveries grew across every market.",
    "The FY24 change was 48% against FY16.",
]


def _repaired_row_root(tmp: Path, name: str, row: str) -> Path:
    return write_output_root(
        tmp / name,
        pages=1,
        blocks=[block(1, index, text) for index, text in enumerate(ROW_BLOCKS, 1)],
        markdown={1: f"| Metric | Change |\n|---|---|\n{row}\n"},
    )


def check_repaired_table_row_is_offered_only_as_fallback(tmp: Path) -> None:
    """Cells mapped to separate blocks: context in the second prompt, never a citation."""
    root = _repaired_row_root(tmp, "repaired", "| Home deliveries | 48% |")
    prompts: list[str] = []

    def adjudicate(payload: dict[str, Any]) -> dict[str, Any]:
        content = payload["messages"][1]["content"]
        prompts.append(content)
        if "<page_context" not in content:
            return adjudication_reply(payload)
        return {
            "verdict": "supported",
            "rationale": "The row's two cells are stated by the two passages.",
            "supporting_evidence_ids": _prompt_ids(payload),
            "missing_qualifiers": [],
        }

    with make_client(default_session(Adjudication=adjudicate)) as client:
        report = client.ingest_document(
            root, source_uri="urn:repaired", reporting_entity=ENTITY
        )
        segments = client.conn.execute(
            "SELECT id, source_text, table_context FROM evidence_unit"
            " WHERE version_id = %s AND kind = %s",
            (report.version_id, str(EvidenceKind.PAGE_MARKDOWN)),
        ).fetchall()
        check(len(segments) == 1, f"the repaired row is indexed once ({len(segments)})")
        check(
            len(segments[0]["table_context"]["sources"]) == 2,
            "and maps to both blocks its cells came from",
        )
        facts = client.conn.execute(
            "SELECT count(*) AS n FROM fact_evidence WHERE evidence_id = %s",
            (segments[0]["id"],),
        ).fetchone()["n"]
        check(facts == 0, "a Markdown row never produces a fact")

        result = client.audit_claim(
            ROW_CLAIM, scope=[report.document_id], reporting_entity=ENTITY
        )
        disposition = client.conn.execute(
            "SELECT ac.reason FROM audit_candidate ac"
            " WHERE ac.audit_id = %s AND ac.evidence_id = %s",
            (result.audit_id, segments[0]["id"]),
        ).fetchone()
        check(
            disposition["reason"] == "markdown group used as page context",
            f"the trace says how the row was used ({disposition['reason']})",
        )
        check(
            not client.conn.execute(
                "SELECT selected FROM audit_candidate"
                " WHERE audit_id = %s AND evidence_id = %s",
                (result.audit_id, segments[0]["id"]),
            ).fetchone()["selected"],
            "and that it was never selected as a citation",
        )

    check(len(prompts) == 2, f"direct evidence was tried first ({len(prompts)})")
    check("<page_context" not in prompts[0], "the first prompt is source evidence only")
    check("<page_context" in prompts[1], "the row is supplied only as fallback context")
    kinds = {c.source_kind for c in result.citations}
    check(
        result.verdict is Verdict.SUPPORTED and kinds == {EvidenceKind.NARRATIVE},
        f"and what gets cited is the original blocks ({result.verdict}, {kinds})",
    )


def check_a_row_with_an_unmapped_cell_offers_no_context(tmp: Path) -> None:
    """One cell nothing on the page states, and the whole row stays out."""
    root = _repaired_row_root(tmp, "unmapped-row", "| Home deliveries | -8% |")
    prompts: list[str] = []

    with make_client(
        default_session(Adjudication=lambda p: (
            prompts.append(p["messages"][1]["content"]), adjudication_reply(p)
        )[1])
    ) as client:
        report = client.ingest_document(
            root, source_uri="urn:unmapped-row", reporting_entity=ENTITY
        )
        segments = client.conn.execute(
            "SELECT count(*) AS n FROM evidence_unit WHERE version_id = %s AND kind = %s",
            (report.version_id, str(EvidenceKind.PAGE_MARKDOWN)),
        ).fetchone()["n"]
        check(segments == 0, "a row with an unaccounted cell is not indexed")
        result = client.audit_claim(
            ROW_CLAIM, scope=[report.document_id], reporting_entity=ENTITY
        )

    check(
        all("<page_context" not in p for p in prompts),
        "so no prompt can carry it as context",
    )
    check(
        result.verdict is Verdict.INSUFFICIENT and not result.citations,
        f"and the claim stays unanswered ({result.verdict})",
    )


def check_direct_evidence_never_reaches_the_fallback(tmp: Path) -> None:
    """A claim the table answers must not pay for the Markdown path."""
    prompts: list[str] = []

    with make_client(
        default_session(Adjudication=lambda p: (
            prompts.append(p["messages"][1]["content"]), adjudication_reply(p)
        )[1])
    ) as client:
        result = client.audit_claim(SUPPORTED, scope="all", reporting_entity=ENTITY)
        # The trace records what was retrieved. No Markdown row in it *is* the
        # evidence that the second pass never ran -- the fail-closed shape here
        # is an absence, not a synthetic "considered and rejected" row.
        markdown = client.conn.execute(
            "SELECT count(*) AS n FROM audit_candidate ac"
            " JOIN evidence_unit e ON e.id = ac.evidence_id"
            " WHERE ac.audit_id = %s AND e.kind = %s",
            (result.audit_id, str(EvidenceKind.PAGE_MARKDOWN)),
        ).fetchone()["n"]
        check(markdown == 0, f"no Markdown candidate was ever retrieved ({markdown})")
        check(
            all(
                r["reason"] and not r["reason"].startswith("markdown group")
                for r in client.conn.execute(
                    "SELECT reason FROM audit_candidate WHERE audit_id = %s",
                    (result.audit_id,),
                ).fetchall()
            ),
            "and the direct candidates keep their own reasons",
        )
    check(result.verdict is Verdict.SUPPORTED, "the table still decides it")
    check(not prompts, "the adjudicator was never consulted, so neither was Markdown")


def check_contradicted_claim(tmp: Path) -> None:
    with make_client(default_session()) as client:
        result = client.audit_claim(CONTRADICTED, scope="all", reporting_entity=ENTITY)
        check(result.verdict is Verdict.CONTRADICTED, f"90% contradicted ({result.rationale})")
        check("40.2" in result.rationale, "rationale names the reported value")
        check(bool(result.citations), "contradiction is cited")


def check_vague_claim_is_insufficient(tmp: Path) -> None:
    with make_client(default_session()) as client:
        result = client.audit_claim(VAGUE, scope="all", reporting_entity=ENTITY)
        check(
            result.verdict is Verdict.INSUFFICIENT,
            f"vague scope is not forced into a contradiction ({result.verdict})",
        )
        check(bool(result.missing_qualifiers), "missing qualifiers reported")


def check_verdict_fails_closed_without_citable_evidence(tmp: Path) -> None:
    """An adjudicator that claims support without naming a passage is downgraded."""
    session = default_session(
        Adjudication=lambda _: {
            "verdict": "supported",
            "rationale": "It looks right to me.",
            "supporting_evidence_ids": [],
            "missing_qualifiers": [],
        }
    )
    with make_client(session) as client:
        # A claim version 1 accepts, about a figure the fixture never states:
        # the model is scripted to support it anyway, and must be refused.
        result = client.audit_claim(
            "Danone reported 12,345 tonnes of packaging waste in 2025.",
            scope="all",
            reporting_entity=ENTITY,
        )
        check(
            result.verdict is not Verdict.SUPPORTED,
            f"uncited support is refused ({result.verdict})",
        )
        check(not result.citations, "no citations invented")


def check_visual_evidence_needs_crop_verification(tmp: Path) -> None:
    root = build_root(tmp / "visual", with_visual=True)
    claim = "The chart shows emissions falling by 40.2%."

    refusing = default_session()
    with make_client(refusing) as client:
        client.ingest_document(root, source_uri="urn:visual", reporting_entity=ENTITY)
        result = client.audit_claim(claim, scope="all", reporting_entity=ENTITY)
        check(
            result.evidence_quality is not EvidenceQuality.VERIFIED_VISUAL,
            "an unverified crop cannot become verified visual evidence",
        )
        check(
            all(c.source_kind is not EvidenceKind.VISUAL for c in result.citations),
            "a failed crop check drops the visual candidate entirely",
        )

    accepting = default_session(
        VisualVerification=lambda _: {
            "result": "support",
            "visible_text": "-40.2% vs 2020",
            "reason_code": "value_and_metric_visible",
        },
        Adjudication=lambda payload: {
            "verdict": "supported",
            "rationale": "The verified crop shows the figure.",
            "supporting_evidence_ids": _visual_ids(payload),
            "missing_qualifiers": [],
        },
    )
    with make_client(accepting) as client:
        result = client.audit_claim(claim, scope="all", reporting_entity=ENTITY)
        visual = [c for c in result.citations if c.source_kind is EvidenceKind.VISUAL]
        check(bool(visual), "an accepted crop can be cited")
        check(
            visual[0].quality is EvidenceQuality.VERIFIED_VISUAL,
            f"and is labelled verified_visual ({visual[0].quality})",
        )
        check(
            result.evidence_quality is not EvidenceQuality.COARSE_REGION,
            "no verdict rests on a coarse region",
        )
        recorded = client.conn.execute(
            "SELECT evidence_id, result FROM visual_verification WHERE audit_id = %s",
            (result.audit_id,),
        ).fetchall()
        check(
            any(r["result"] == "support" for r in recorded),
            f"and the crop check is on the record ({[r['result'] for r in recorded]})",
        )


def check_a_conflicting_crop_is_evidence_against_the_claim(tmp: Path) -> None:
    """A legible chart showing a *different* figure is the strongest refutation.

    Dropping it as "not supporting" would discard exactly that, so it goes
    forward as verified evidence and the comparison decides which way it counts.
    """
    claim = "The chart shows emissions falling by 40.2%."
    offered: list[int] = []

    def adjudicate(payload: dict[str, Any]) -> dict[str, Any]:
        offered.extend(_visual_ids(payload))
        return adjudication_reply(payload)

    conflicting = default_session(
        VisualVerification=lambda _: {
            "result": "conflict",
            "visible_text": "-34.5% vs 2020",
            "reason_code": "value_differs",
        },
        Adjudication=adjudicate,
    )
    with make_client(conflicting) as client:
        result = client.audit_claim(claim, scope="all", reporting_entity=ENTITY)
        rows = client.conn.execute(
            "SELECT evidence_id, result FROM visual_verification WHERE audit_id = %s",
            (result.audit_id,),
        ).fetchall()
        check(
            any(r["result"] == "conflict" for r in rows),
            f"the conflicting crop is recorded ({[r['result'] for r in rows]})",
        )
        statuses = client.conn.execute(
            "SELECT ac.visual_status, ac.reason FROM audit_candidate ac"
            " WHERE ac.audit_id = %s AND ac.evidence_id = ANY(%s)",
            (result.audit_id, [int(r["evidence_id"]) for r in rows]),
        ).fetchall()
        check(
            all(s["visual_status"] == "verified" for s in statuses),
            f"and counts as verified, not rejected ({[s['visual_status'] for s in statuses]})",
        )
    check(
        set(offered) == {int(r["evidence_id"]) for r in rows},
        f"and it is offered to the adjudicator as evidence ({offered})",
    )
    check(
        all(c.source_kind is not EvidenceKind.VISUAL for c in result.citations),
        "this adjudicator cited nothing, so nothing is cited",
    )


# --- H-1: a selected page range keeps its original PDF page numbers ---------


def check_middle_range_keeps_pdf_pages(tmp: Path) -> None:
    root = write_output_root(
        tmp / "middle",
        page_numbers=[10, 11],
        blocks=[
            block(10, 1, "4.8.2 NATURE INDICATORS", kind="section_header",
                  heading_path=["4.8.2 NATURE INDICATORS"]),
            block(11, 1, "Water withdrawal intensity improved during the year.",
                  heading_path=["4.8.3 WATER"]),
        ],
        tables={10: [kpi_table()]},
    )
    with make_client(default_session()) as client:
        report = client.ingest_document(root, source_uri="urn:middle", reporting_entity=ENTITY)
        check(report.status is VersionStatus.READY, "a range starting after page 1 indexes")

        matches = client.search_evidence(SUPPORTED, document_ids=[report.document_id])
        check(bool(matches), "the middle-range document is searchable")
        pages = {m.citation.pdf_page for m in matches}
        check(pages <= {10, 11}, f"search citations keep the pdf page numbers ({pages})")

        result = client.audit_claim(SUPPORTED, scope=[report.document_id], reporting_entity=ENTITY)
        check(result.verdict is Verdict.SUPPORTED, f"claim supported ({result.rationale})")
        check(
            all(c.pdf_page in (10, 11) for c in result.citations),
            f"audit citations keep the pdf page numbers "
            f"({[c.pdf_page for c in result.citations]})",
        )
        detail = client.get_evidence(result.citations[0].evidence_id)
        check(detail.pdf_page in (10, 11), "get_evidence reports the pdf page")
        check(detail.page_image_path is not None, "the page image still resolves")


# --- H-3: document identity -------------------------------------------------


def check_source_less_documents_stay_separate(tmp: Path) -> None:
    """Two unrelated roots with the same basename, no PDF, and no URI."""
    first = build_root(tmp / "left" / "report")
    second = build_root(tmp / "right" / "report")
    with make_client(default_session()) as client:
        a = client.ingest_document(first, reporting_entity=ENTITY)
        b = client.ingest_document(second, reporting_entity=ENTITY)
        check(a.document_id != b.document_id, "same basename, different documents")

        ready = {int(r["id"]) for r in client.documents()}
        check({a.document_id, b.document_id} <= ready, "both are ready at once")
        for report in (a, b):
            hits = client.search_evidence(SUPPORTED, document_ids=[report.document_id])
            check(bool(hits), f"document {report.document_id} is queryable on its own")

        again = client.ingest_document(second, reporting_entity=ENTITY)
        check(
            again.document_id == b.document_id and again.version_id == b.version_id,
            "re-ingesting the same root reuses its document and ready version",
        )
        check(again.reused_existing, "and is still a no-op")

        client.remove_document(a.document_id, confirm_document_id=a.document_id)
        survivors = {int(r["id"]) for r in client.documents()}
        check(a.document_id not in survivors, "the removed document is gone")
        check(b.document_id in survivors, "removing one leaves the other ready")
        check(
            bool(client.search_evidence(SUPPORTED, document_ids=[b.document_id])),
            "the surviving document still returns evidence",
        )


def check_shared_source_uri_is_one_document(tmp: Path) -> None:
    copies = [build_root(tmp / "copy_a" / "run"), build_root(tmp / "copy_b" / "run")]
    with make_client(default_session()) as client:
        ids = {client.ingest_document(root, source_uri="urn:same", reporting_entity=ENTITY).document_id
               for root in copies}
        check(len(ids) == 1, "an explicit logical source_uri makes them one document")


def check_a_legacy_schema_is_refused_not_migrated(tmp: Path) -> None:
    """PD-03: a mismatched database is rebuilt, never upgraded in place.

    The old behaviour here was a backfill that gave pre-identity rows a
    deterministic key. It is gone on purpose: index data is disposable, and a
    half-migrated database that still answers queries is a worse outcome than
    one that refuses and says how to rebuild.
    """
    from claim_evidence.db import SchemaMismatchError

    with make_client(default_session()) as client:
        conn = client.conn
        conn.execute("DROP INDEX IF EXISTS document_identity_idx")
        conn.execute("ALTER TABLE document DROP COLUMN IF EXISTS identity_key")
        legacy = int(
            conn.execute(
                "INSERT INTO document (name, sha256, source_uri)"
                " VALUES ('legacy.pdf', 'cafe1234', NULL) RETURNING id"
            ).fetchone()["id"]
        )
        conn.commit()

        try:
            client.init_db()
        except SchemaMismatchError as exc:
            check(
                "document.identity_key" in str(exc),
                "the refusal names the column the database is missing",
            )
            check(
                "reset-dev" in str(exc),
                "the refusal points at the guarded reset, not at a migration",
            )
        else:
            raise AssertionError("a legacy schema must be refused, not migrated")

        row = conn.execute(
            "SELECT id FROM document WHERE id = %s", (legacy,)
        ).fetchone()
        check(row is not None, "the refused init left the existing row alone")
        # Put the schema back for the checks that follow. This is test-harness
        # repair, not a migration path: the product's answer to a mismatched
        # database is `db reset-dev`, which is what the check above proves.
        conn.execute("DELETE FROM document WHERE id = %s", (legacy,))
        conn.execute("ALTER TABLE document ADD COLUMN identity_key text")
        conn.execute(
            "UPDATE document SET identity_key ="
            " encode(sha256(convert_to('restored:' || id::text, 'UTF8')), 'hex')"
        )
        conn.execute("ALTER TABLE document ALTER COLUMN identity_key SET NOT NULL")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS document_identity_idx"
            " ON document (identity_key)"
        )
        conn.commit()
        check(
            client.init_db() == "unchanged",
            "the restored database is recognized as current again",
        )


# --- H-4: an interrupted attempt is reconciled, not merged into ------------


def check_resume_reconciles_the_building_version(tmp: Path) -> None:
    from claim_evidence.db import (
        set_embeddings,
        start_version,
        upsert_document,
        upsert_evidence,
        upsert_fact,
        upsert_page,
    )
    from claim_evidence.ingest import build_fingerprint, identity_key
    from claim_evidence.models import Fact
    from claim_evidence.source import OutputReader, page_units

    root = build_root(tmp / "resume")
    config = settings()
    reader = OutputReader(root)
    pages = reader.validate()
    page = pages[0]
    units = list(page_units(reader, page, reader.blocks_by_page().get(1, [])))
    changed, unchanged = units[0], units[1]
    marker = [1.0] + [0.0] * (DIMENSIONS - 1)

    embedded_texts: list[str] = []
    session = default_session()
    session.embed_hook = lambda texts: embedded_texts.extend(texts) and None

    conn = connect(config.database_url)
    with make_client(session, conn) as client:
        # Seed an attempt that died before activation, holding stale text, a
        # unit the source no longer produces, a page that is gone, and a fact
        # built from all of it.
        fingerprint = build_fingerprint(
            reader, config, client.ollama,
            source_sha256=None, reporting_entity=ENTITY,
            extract_narrative_facts=True,
        )
        document_id = upsert_document(
            conn, reader.root.name, None, "urn:resume",
            identity_key(reader.root, None, "urn:resume"),
        )
        version_id = start_version(
            conn, document_id, fingerprint, embed_model=config.embed_model,
            embed_dim=DIMENSIONS, output_root=str(reader.root), source_pdf=None,
        )
        stale = changed.model_copy(
            update={"text": "OBSOLETE TEXT", "normalized_text": "obsolete text"}
        )
        ghost = changed.model_copy(
            update={
                "unit_key": "p0001:narrative:ghost",
                "text": "A paragraph the source no longer contains.",
                "normalized_text": "a paragraph the source no longer contains",
            }
        )
        page_id = upsert_page(
            conn, version_id, page.page, page.width, page.height, page.rel
        )
        seeded = upsert_evidence(conn, version_id, page_id, [stale, unchanged, ghost])
        gone_page = upsert_page(conn, version_id, 99, page.width, page.height, "page_0099")
        upsert_evidence(
            conn, version_id, gone_page,
            [changed.model_copy(update={"unit_key": "p0099:narrative:1", "page": 99})],
        )
        set_embeddings(
            conn,
            [
                (seeded[stale.unit_key], session.vector("obsolete text")),
                (seeded[unchanged.unit_key], marker),
                (seeded[ghost.unit_key], session.vector(ghost.normalized_text)),
            ],
        )
        upsert_fact(
            conn, version_id,
            Fact(subject="Danone", metric="obsolete metric", value_text="obsolete",
                 quote="OBSOLETE TEXT", evidence_keys=[stale.unit_key]),
            "obsolete metric", None, [seeded[stale.unit_key]],
        )
        conn.commit()
        stale_fact_id = conn.execute(
            "SELECT id FROM fact WHERE version_id = %s", (version_id,)
        ).fetchone()["id"]

        embedded_texts.clear()
        report = client.ingest_document(root, source_uri="urn:resume", reporting_entity=ENTITY)
        check(report.version_id == version_id, "the interrupted attempt is resumed")
        check(report.status is VersionStatus.READY, "it activates once checks pass")

        row = conn.execute(
            "SELECT id, source_text, embedding FROM evidence_unit"
            " WHERE version_id = %s AND unit_key = %s",
            (version_id, changed.unit_key),
        ).fetchone()
        check(row["source_text"] == changed.text, "changed text is overwritten")
        check(
            changed.normalized_text in embedded_texts,
            "the changed unit was re-embedded",
        )
        distance = conn.execute(
            "SELECT embedding <#> %s AS d FROM evidence_unit WHERE id = %s",
            (normalize_embedding(session.vector(changed.normalized_text)), row["id"]),
        ).fetchone()["d"]
        check(
            abs(float(distance) + 1.0) < 1e-6,
            f"its stored embedding represents the new text ({distance})",
        )

        kept = conn.execute(
            "SELECT embedding <#> %s AS d FROM evidence_unit"
            " WHERE version_id = %s AND unit_key = %s",
            (normalize_embedding(marker), version_id, unchanged.unit_key),
        ).fetchone()["d"]
        check(
            abs(float(kept) + 1.0) < 1e-6, "an unchanged unit keeps its embedding"
        )
        check(
            unchanged.normalized_text not in embedded_texts,
            "and was not re-embedded",
        )

        leftovers = conn.execute(
            "SELECT count(*) AS n FROM evidence_unit"
            " WHERE version_id = %s AND unit_key = ANY(%s)",
            (version_id, [ghost.unit_key, "p0099:narrative:1"]),
        ).fetchone()["n"]
        check(leftovers == 0, "removed units do not survive the resume")
        stale_pages = conn.execute(
            "SELECT count(*) AS n FROM page WHERE version_id = %s AND pdf_page = 99",
            (version_id,),
        ).fetchone()["n"]
        check(stale_pages == 0, "a page the manifest no longer lists is deleted")

        old_fact = conn.execute(
            "SELECT count(*) AS n FROM fact WHERE id = %s", (stale_fact_id,)
        ).fetchone()["n"]
        check(old_fact == 0, "facts from the earlier attempt are rebuilt, not kept")
        links = conn.execute(
            "SELECT count(*) AS n FROM fact_evidence fe"
            " LEFT JOIN fact f ON f.id = fe.fact_id WHERE f.id IS NULL"
        ).fetchone()["n"]
        check(links == 0, "no orphaned fact_evidence links remain")
        quotes = conn.execute(
            "SELECT count(*) AS n FROM fact WHERE version_id = %s AND quote = %s",
            (version_id, "OBSOLETE TEXT"),
        ).fetchone()["n"]
        check(quotes == 0, "no rebuilt fact carries the obsolete quote")


# --- H-5: an invalid scope is an error, never a verdict ---------------------


def _expect(error: type[Exception], call, message: str) -> None:
    try:
        call()
    except error as exc:
        check(True, f"{message} ({type(exc).__name__}: {exc})")
        return
    except Exception as exc:  # noqa: BLE001 - the wrong type is the failure
        raise AssertionError(f"{message}: raised {type(exc).__name__} instead") from exc
    raise AssertionError(f"{message}: nothing was raised")


def check_document_scope_is_validated(tmp: Path) -> None:
    root = build_root(tmp / "scoped")
    with make_client(default_session()) as client:
        indexed = client.ingest_document(root, source_uri="urn:scoped", reporting_entity=ENTITY)
        ready = indexed.document_id

        check(bool(client.search_evidence(SUPPORTED)), "None searches every ready document")
        check(
            bool(client.search_evidence(SUPPORTED, document_ids=[])),
            "an empty list searches every ready document",
        )
        scoped = client.search_evidence(SUPPORTED, document_ids=[ready])
        check(
            bool(scoped) and {m.citation.document_id for m in scoped} == {ready},
            "an explicit document restricts the results to itself",
        )
        check(
            len(client.search_evidence(SUPPORTED, document_ids=[ready, ready])) == len(scoped),
            "a duplicated id is harmless",
        )

        unknown = 10_000_000
        for call, label in (
            (lambda: client.search_evidence(SUPPORTED, document_ids=[unknown]), "search"),
            (lambda: client.audit_claim(SUPPORTED, scope=[unknown], reporting_entity=ENTITY), "audit"),
        ):
            _expect(NotFoundError, call, f"{label}: an unknown document is not found")
        _expect(
            NotFoundError,
            lambda: client.audit_claim(SUPPORTED, scope=[ready, unknown], reporting_entity=ENTITY),
            "a mixed selection fails as a whole rather than searching the valid part",
        )
        for value in (True, "abc", 1.5):
            _expect(
                ValidationError,
                lambda v=value: client.search_evidence(SUPPORTED, document_ids=[v]),
                f"{value!r} is a validation error",
            )

        # A document whose only version is still building has nothing to query.
        building = client.conn.execute(
            "INSERT INTO document (name, identity_key) VALUES ('halfbuilt', 'k-halfbuilt')"
            " RETURNING id"
        ).fetchone()["id"]
        client.conn.execute(
            "INSERT INTO document_version"
            " (document_id, fingerprint, embed_model, embed_dim, output_root)"
            " VALUES (%s, 'fp-halfbuilt', 'fake', %s, %s)",
            (building, DIMENSIONS, str(root)),
        )
        client.conn.commit()
        _expect(
            IndexNotReadyError,
            lambda: client.audit_claim(SUPPORTED, scope=[int(building)], reporting_entity=ENTITY),
            "a building-only document is not ready",
        )
        _expect(
            IndexNotReadyError,
            lambda: client.search_evidence(SUPPORTED, document_ids=[int(building)]),
            "and search says so too",
        )

        removed = client.ingest_document(build_root(tmp / "removed"), source_uri="urn:gone", reporting_entity=ENTITY)
        client.remove_document(removed.document_id, confirm_document_id=removed.document_id)
        _expect(
            NotFoundError,
            lambda: client.audit_claim(SUPPORTED, scope=[removed.document_id], reporting_entity=ENTITY),
            "a removed document is not found",
        )
        client.conn.execute("DELETE FROM document WHERE id = %s", (building,))
        client.conn.commit()


def check_invalid_scope_costs_nothing(tmp: Path) -> None:
    """Validation runs before Ollama and before the audit row is written."""
    session = default_session()
    with make_client(session) as client:
        before = client.conn.execute("SELECT count(*) AS n FROM audit_run").fetchone()["n"]
        calls = len(session.requests)
        _expect(
            NotFoundError,
            lambda: client.audit_claim(SUPPORTED, scope=[10_000_001], reporting_entity=ENTITY),
            "an unknown scope is rejected",
        )
        check(len(session.requests) == calls, "the model was never called")
        after = client.conn.execute("SELECT count(*) AS n FROM audit_run").fetchone()["n"]
        check(after == before, "no audit row was created")


def check_backend_mismatch_costs_nothing(tmp: Path) -> None:
    """A backend switch is rejected before model calls or database writes."""
    from dataclasses import replace

    root = build_root(tmp / "backend-mismatch")
    with make_client(default_session()) as source:
        indexed = source.ingest_document(
            root, source_uri="urn:backend-mismatch", reporting_entity=ENTITY
        )
        session = default_session()
        config = replace(settings(), model_backend="llamacpp")
        with ClaimEvidence(
            config, connect(config.database_url), OllamaClient(config, session)
        ) as client:
            before_audits = client.conn.execute(
                "SELECT count(*) AS n FROM audit_run"
            ).fetchone()["n"]
            before_version = client.conn.execute(
                "SELECT status, fact_candidates_total, fact_candidates_succeeded"
                " FROM document_version WHERE id = %s",
                (indexed.version_id,),
            ).fetchone()
            before_failures = client.conn.execute(
                "SELECT count(*) AS n FROM fact_candidate_failure WHERE version_id = %s",
                (indexed.version_id,),
            ).fetchone()["n"]

            _expect(
                IndexNotReadyError,
                lambda: client.search_evidence(
                    SUPPORTED, document_ids=[indexed.document_id]
                ),
                "search rejects an index from another backend",
            )
            _expect(
                IndexNotReadyError,
                lambda: client.audit_claim(
                    SUPPORTED,
                    scope=[indexed.document_id],
                    reporting_entity=ENTITY,
                ),
                "audit rejects an index from another backend",
            )
            _expect(
                IndexNotReadyError,
                lambda: client.retry_facts(indexed.document_id),
                "fact retry rejects an index from another backend",
            )
            _expect(
                IndexNotReadyError,
                lambda: client.search_evidence(SUPPORTED),
                "an unscoped mixed-backend search is rejected",
            )
            _expect(
                IndexNotReadyError,
                lambda: client.audit_claim(
                    SUPPORTED, scope="all", reporting_entity=ENTITY
                ),
                "an unscoped mixed-backend audit is rejected",
            )

            check(not session.requests, "the incompatible index made no model request")
            after_audits = client.conn.execute(
                "SELECT count(*) AS n FROM audit_run"
            ).fetchone()["n"]
            after_version = client.conn.execute(
                "SELECT status, fact_candidates_total, fact_candidates_succeeded"
                " FROM document_version WHERE id = %s",
                (indexed.version_id,),
            ).fetchone()
            after_failures = client.conn.execute(
                "SELECT count(*) AS n FROM fact_candidate_failure WHERE version_id = %s",
                (indexed.version_id,),
            ).fetchone()["n"]
            check(after_audits == before_audits, "no audit row was created")
            check(
                after_version == before_version and after_failures == before_failures,
                "fact retry did not mutate the version",
            )

            replacement = client.ingest_document(
                root,
                source_uri="urn:backend-mismatch",
                reporting_entity=ENTITY,
            )
            check(
                replacement.document_id == indexed.document_id
                and replacement.version_id != indexed.version_id
                and replacement.status is VersionStatus.READY,
                "re-ingestion activates a compatible replacement",
            )
            check(
                bool(
                    client.search_evidence(
                        SUPPORTED, document_ids=[replacement.document_id]
                    )
                ),
                "the replacement is queryable through llama.cpp",
            )

        source.remove_document(
            indexed.document_id, confirm_document_id=indexed.document_id
        )


def check_llamacpp_runs_the_public_pipeline(tmp: Path) -> None:
    """The public pipeline works through llama.cpp's wire format."""
    from dataclasses import replace

    source_text = f"{SUPPORTED} This statement is from the annual report."
    chart_claim = "The chart shows emissions falling by 40.2%."

    def adjudicate(payload: dict[str, Any]) -> dict[str, Any]:
        if chart_claim in payload["messages"][1]["content"]:
            return {
                "verdict": "supported",
                "rationale": "The verified crop shows the figure.",
                "supporting_evidence_ids": _visual_ids(payload),
                "missing_qualifiers": [],
            }
        return adjudication_reply(payload)

    session = default_session(
        ClaimSplit=lambda _: {"claims": [{"text": SUPPORTED}]},
        EntailmentBatch=lambda _: {
            "checks": [
                {"claim_index": 0, "outcome": "entailed", "reason_code": "scripted"}
            ]
        },
        Adjudication=adjudicate,
        VisualVerification=lambda _: {
            "result": "support",
            "visible_text": "-40.2% vs 2020",
            "reason_code": "value_and_metric_visible",
        },
    )
    config = replace(settings(), model_backend="llamacpp")
    with ClaimEvidence(
        config, connect(config.database_url), OllamaClient(config, session)
    ) as client:
        indexed = client.ingest_document(
            build_root(tmp / "llamacpp-pipeline", with_visual=True),
            source_uri="urn:llamacpp-pipeline",
            reporting_entity=ENTITY,
        )
        check(
            indexed.status is VersionStatus.READY,
            "llama.cpp ingestion reaches ready",
        )
        check(
            bool(client.search_evidence(SUPPORTED, document_ids=[indexed.document_id])),
            "llama.cpp search returns evidence",
        )
        decomposition = client.decompose_claims(
            source_text, reporting_entity=ENTITY
        )
        check(
            decomposition.claims[0].text == SUPPORTED,
            "llama.cpp decomposition is grounded",
        )
        verification = client.verify_claims(
            source_text, [SUPPORTED], reporting_entity=ENTITY
        )
        check(verification.ok, "llama.cpp claim verification passes")
        deterministic = client.audit_claim(
            SUPPORTED, scope=[indexed.document_id], reporting_entity=ENTITY
        )
        check(
            deterministic.verdict is Verdict.SUPPORTED,
            "llama.cpp deterministic audit is supported",
        )
        semantic = client.audit_claim(
            VAGUE, scope=[indexed.document_id], reporting_entity=ENTITY
        )
        check(
            semantic.verdict is Verdict.INSUFFICIENT,
            "llama.cpp semantic audit is validated",
        )
        identities = client.conn.execute(
            "SELECT dv.embed_model, ar.chat_model, ar.embed_model AS audit_embed_model"
            " FROM document_version dv JOIN audit_run ar ON ar.id = %s"
            " WHERE dv.id = %s",
            (semantic.audit_id, indexed.version_id),
        ).fetchone()
        check(
            all(str(value).startswith("llamacpp:") for value in identities.values()),
            "stored model identifiers name llama.cpp",
        )
        visual = client.audit_claim(
            chart_claim, scope=[indexed.document_id], reporting_entity=ENTITY
        )
        check(
            any(
                citation.quality is EvidenceQuality.VERIFIED_VISUAL
                for citation in visual.citations
            ),
            "llama.cpp vision verifies a real crop",
        )
        check(
            all(url.endswith(("/v1/embeddings", "/v1/chat/completions")) for url, _ in session.requests),
            "the pipeline used only llama.cpp inference endpoints",
        )
        client.remove_document(
            indexed.document_id, confirm_document_id=indexed.document_id
        )


def check_empty_index_is_not_an_insufficient_verdict(tmp: Path) -> None:
    with make_client(default_session()) as client:
        client.conn.execute("DELETE FROM document")
        client.conn.commit()
        _expect(
            IndexNotReadyError,
            lambda: client.audit_claim(SUPPORTED, scope="all", reporting_entity=ENTITY),
            "an empty index raises rather than returning insufficient",
        )
        _expect(
            IndexNotReadyError,
            lambda: client.search_evidence(SUPPORTED),
            "and search raises rather than returning nothing",
        )


# --- H-7: one authoritative structured explanation --------------------------


def _qualifiers(comparison) -> dict[str, Any]:
    return {q.qualifier: q for q in comparison.qualifiers}


def check_supported_claim_explains_itself(tmp: Path) -> None:
    with make_client(default_session()) as client:
        result = client.audit_claim(SUPPORTED, scope="all", reporting_entity=ENTITY)
        explanation = result.decision_explanation
        check(explanation is not None, "a verdict carries a structured explanation")
        check(
            explanation.decided_by == "deterministic_comparison",
            f"decided by arithmetic ({explanation.decided_by})",
        )
        check(
            explanation.verdict_rule == "exact_numeric_match",
            f"named rule ({explanation.verdict_rule})",
        )

        matching = [
            c for c in explanation.evidence_comparisons if c.numeric.outcome == "match"
        ]
        check(bool(matching), "the matching comparison is reported")
        target = matching[0]
        quals = _qualifiers(target)
        for name in ("scope", "unit", "reporting_period", "baseline_period"):
            check(
                quals[name].status == "match",
                f"{name} matched ({quals[name].status}: {quals[name].source_value})",
            )
        check("40.2" in (target.numeric.source_value or ""), "the source value is quoted")
        check(target.numeric.claim_value == "40.2", "the claim value is quoted")
        check(
            target.evidence_id in {c.evidence_id for c in result.citations},
            "the comparison points at cited evidence",
        )
        check(target.pdf_page in (1, 2), f"and at its page ({target.pdf_page})")

        check(
            result.timings.get("total") is not None
            and result.timings["retrieval"] is not None,
            f"timings are measured ({result.timings})",
        )
        check(
            set(result.timings) == {
                "parsing", "retrieval", "fusion_context", "visual_verification",
                "verdict", "persistence", "total",
            },
            f"every public timing group is present ({sorted(result.timings)})",
        )
        check(bool(result.index_references), "the audited index versions are reported")
        reference = result.index_references[0]
        check(
            reference.embedding_dimensions == DIMENSIONS
            and reference.embedding_model == client.settings.embed_model,
            "the reference names the embedding model and width",
        )


def check_contradicted_claim_explains_the_conflict(tmp: Path) -> None:
    with make_client(default_session()) as client:
        result = client.audit_claim(CONTRADICTED, scope="all", reporting_entity=ENTITY)
        explanation = result.decision_explanation
        check(
            explanation.verdict_rule == "comparable_numeric_conflict",
            f"named rule ({explanation.verdict_rule})",
        )
        conflicting = [
            c for c in explanation.evidence_comparisons if c.numeric.outcome == "conflict"
        ]
        check(bool(conflicting), "the conflicting comparison is reported")
        target = conflicting[0]
        check(target.numeric.claim_value == "90", "the claim's 90 is shown")
        check("40.2" in (target.numeric.source_value or ""), "against the source's 40.2")
        quals = _qualifiers(target)
        check(
            all(quals[n].status == "match" for n in ("scope", "unit", "reporting_period")),
            "every other material qualifier still matches",
        )


def check_vague_claim_explains_the_scope_gap(tmp: Path) -> None:
    with make_client(default_session()) as client:
        result = client.audit_claim(VAGUE, scope="all", reporting_entity=ENTITY)
        explanation = result.decision_explanation
        check(
            explanation.verdict_rule in ("scope_not_comparable", "missing_material_qualifier"),
            f"named rule ({explanation.verdict_rule})",
        )
        check(
            explanation.decided_by == "semantic_adjudication",
            "arithmetic refused, so the semantic path decided",
        )
        scoped = [
            _qualifiers(c)["scope"]
            for c in explanation.evidence_comparisons
            if "scope" in _qualifiers(c)
        ]
        check(bool(scoped), "scope comparisons are reported")
        check(
            all(q.status in ("mismatch", "missing") for q in scoped),
            "no scope is claimed to match",
        )
        check(
            all(
                c.numeric.outcome != "conflict"
                for c in explanation.evidence_comparisons
                if _qualifiers(c)["scope"].status != "match"
            ),
            "a scope that does not line up is never a numeric contradiction",
        )


def check_explanation_round_trips_through_the_trace(tmp: Path) -> None:
    with make_client(default_session()) as client:
        result = client.audit_claim(SUPPORTED, scope="all", reporting_entity=ENTITY)
        trace = client.get_audit_trace(result.audit_id)
        check(
            trace.decision_explanation.verdict_rule
            == result.decision_explanation.verdict_rule,
            "the stored rule matches the returned one",
        )
        check(
            len(trace.decision_explanation.evidence_comparisons)
            == len(result.decision_explanation.evidence_comparisons),
            "every comparison survives persistence",
        )
        check(trace.timings.get("total") is not None, "timings persist")
        check(
            [r.document_version_id for r in trace.index_references]
            == [r.document_version_id for r in result.index_references],
            "index references persist",
        )

        insufficient = client.audit_claim(VAGUE, scope="all", reporting_entity=ENTITY)
        empty_trace = client.get_audit_trace(insufficient.audit_id)
        check(not insufficient.citations, "the vague claim cites nothing")
        check(
            bool(empty_trace.index_references),
            "an insufficient audit still records what it searched",
        )
        check(
            bool(empty_trace.document_ids),
            "and reports the audited document ids without citations",
        )


def check_explanation_carries_no_model_text(tmp: Path) -> None:
    with make_client(default_session()) as client:
        result = client.audit_claim(VAGUE, scope="all", reporting_entity=ENTITY)
        stored = client.conn.execute(
            "SELECT decision_explanation, timings, index_references"
            " FROM audit_run WHERE id = %s",
            (result.audit_id,),
        ).fetchone()
        blob = json.dumps(
            [stored["decision_explanation"], stored["timings"], stored["index_references"]]
        )
        for secret in (
            "You decide whether",  # the adjudication system prompt
            "You decompose one atomic claim",  # the claim-parse prompt
            "does not state a comparable scope",  # the model's own rationale
            "postgresql://",
            str(tmp),
        ):
            check(secret not in blob, f"persisted explanation omits {secret[:32]!r}")


# --- M-9: context follows the page, not the insert order --------------------


def check_expansion_follows_source_order_not_ids(tmp: Path) -> None:
    """Insert the page's units backwards; expansion must ignore that entirely."""
    from claim_evidence.db import (
        start_version,
        upsert_document,
        upsert_evidence,
        upsert_page,
        neighbours,
    )
    from claim_evidence.ingest import identity_key
    from claim_evidence.source import OutputReader, page_units

    root = build_root(tmp / "ordered")
    config = settings()
    conn = connect(config.database_url)
    with make_client(default_session(), conn) as client:
        reader = OutputReader(root)
        page = reader.validate()[0]
        units = list(page_units(reader, page, reader.blocks_by_page().get(1, [])))
        check(
            [u.source_order for u in units] == list(range(len(units))),
            "every unit carries a source order",
        )

        document_id = upsert_document(
            conn, "ordered", None, "urn:ordered",
            identity_key(reader.root, None, "urn:ordered"),
        )
        version_id = start_version(
            conn, document_id, "fp-ordered", embed_model="fake",
            embed_dim=DIMENSIONS, output_root=str(reader.root), source_pdf=None,
        )
        page_id = upsert_page(
            conn, version_id, page.page, page.width, page.height, page.rel
        )
        # Reversed: the highest evidence id is now the first unit on the page.
        ids = upsert_evidence(conn, version_id, page_id, list(reversed(units)))
        conn.execute(
            "UPDATE document_version SET status = 'ready' WHERE id = %s", (version_id,)
        )
        conn.commit()

        by_key = {u.unit_key: u for u in units}
        value = next(
            u for u in units
            if u.kind is EvidenceKind.TABLE_VALUE and u.table_context.get("value") == "(40.2) %"
        )
        expanded = neighbours(conn, ids[value.unit_key])
        returned = [row["unit_key"] for row in expanded]
        check(bool(returned), "the value cell expands to something")

        siblings = [
            k for k in returned if by_key[k].context_key == value.context_key
        ]
        check(
            returned[0] in siblings,
            f"its own table row context comes first ({returned[0]})",
        )
        check(
            any(by_key[k].kind is EvidenceKind.TABLE_ROW for k in siblings),
            "and the row itself is among them",
        )
        check(
            all(by_key[k].kind is not EvidenceKind.PAGE_MARKDOWN for k in returned),
            "generated page markdown is never offered as citable support",
        )
        check(
            value.unit_key not in returned, "the candidate is not its own neighbour"
        )

        check(
            all(int(row["pdf_page"]) == page.page for row in expanded),
            "every neighbour is on the candidate's own page",
        )
        check(
            all(int(row["document_id"]) == document_id for row in expanded),
            "and in the candidate's own document",
        )

        conn.execute("DELETE FROM document WHERE id = %s", (document_id,))
        conn.commit()


def check_narrative_blocks_expand_in_page_order(tmp: Path) -> None:
    from claim_evidence.db import neighbours

    root = build_root(tmp / "prose")
    with make_client(default_session()) as client:
        report = client.ingest_document(root, source_uri="urn:prose", reporting_entity=ENTITY)
        rows = client.conn.execute(
            "SELECT e.id, e.unit_key, e.source_order, e.context_key FROM evidence_unit e"
            " JOIN page p ON p.id = e.page_id"
            " WHERE e.version_id = %s AND p.pdf_page = 1 AND e.kind = 'narrative'"
            " ORDER BY e.source_order",
            (report.version_id,),
        ).fetchall()
        check(len(rows) >= 2, f"the page has adjacent narrative blocks ({len(rows)})")
        check(
            all(r["context_key"].startswith("p0001:block:") for r in rows),
            "each block carries its own context key",
        )

        first = rows[0]
        returned = [r["unit_key"] for r in neighbours(client.conn, int(first["id"]))]
        check(
            rows[1]["unit_key"] in returned,
            "the next block on the page is a neighbour of the first",
        )


def check_expansion_uses_its_indexes(tmp: Path) -> None:
    with make_client(default_session()) as client:
        conn = client.conn
        evidence_id = conn.execute(
            "SELECT e.id FROM evidence_unit e JOIN document_version v ON v.id = e.version_id"
            " WHERE v.status = 'ready' AND e.source_order IS NOT NULL LIMIT 1"
        ).fetchone()["id"]
        conn.execute("SET enable_seqscan = off")
        plan = "\n".join(
            str(r)
            for r in conn.execute(
                "EXPLAIN SELECT id FROM evidence_unit"
                " WHERE page_id = (SELECT page_id FROM evidence_unit WHERE id = %s)"
                " ORDER BY source_order LIMIT 4",
                (evidence_id,),
            ).fetchall()
        )
        check(
            "evidence_unit_source_order_idx" in plan or "evidence_unit_page_idx" in plan,
            f"the page/source-order lookup uses an index ({plan[:120]})",
        )
        conn.execute("SET enable_seqscan = on")


# --- M-4: an audit records its corpus and its outcome -----------------------


def check_audit_persists_its_scope_and_status(tmp: Path) -> None:
    with make_client(default_session()) as client:
        ready = {int(r["id"]) for r in client.documents()}
        chosen = sorted(ready)[0]

        scoped = client.audit_claim(VAGUE, scope=[chosen], reporting_entity=ENTITY)
        trace = client.get_audit_trace(scoped.audit_id)
        check(not scoped.citations, "the vague claim cites nothing")
        check(
            trace.document_ids == [chosen],
            f"the requested scope is retained without citations ({trace.document_ids})",
        )
        check(trace.status == "completed", "a finished audit is explicitly completed")
        check(trace.completed_at is not None, "and carries its completion time")
        check(trace.failure_code is None, "with no failure metadata")

        everything = client.audit_claim(SUPPORTED, scope="all", reporting_entity=ENTITY)
        all_trace = client.get_audit_trace(everything.audit_id)
        check(
            set(all_trace.document_ids) == ready,
            f"an unscoped audit records the ids ready at the time ({all_trace.document_ids})",
        )
        stored = client.conn.execute(
            "SELECT requested_document_ids FROM audit_run WHERE id = %s",
            (everything.audit_id,),
        ).fetchone()["requested_document_ids"]
        check(
            set(int(i) for i in stored) == ready,
            "the exact ids are stored, not an empty list meaning 'whatever exists'",
        )


def check_trace_survives_document_removal(tmp: Path) -> None:
    root = build_root(tmp / "doomed")
    with make_client(default_session()) as client:
        report = client.ingest_document(root, source_uri="urn:doomed", reporting_entity=ENTITY)
        result = client.audit_claim(SUPPORTED, scope=[report.document_id], reporting_entity=ENTITY)
        client.remove_document(report.document_id, confirm_document_id=report.document_id)

        trace = client.get_audit_trace(result.audit_id)
        check(
            report.document_id in trace.document_ids,
            "the trace still names the document that was searched",
        )
        check(trace.status == "completed", "and still reads as a completed audit")
        check(trace.claim == SUPPORTED, "with its claim intact")


def check_failed_audit_is_explicit_and_safe(tmp: Path) -> None:
    import claim_evidence.audit as audit_module

    original = audit_module._deterministic
    audit_module._deterministic = lambda *a, **k: (_ for _ in ()).throw(
        audit_module.AuditError("adjudication failed: secret-sentinel-model-reply")
    )
    try:
        with make_client(default_session()) as client:
            before = client.conn.execute(
                "SELECT count(*) AS n FROM audit_run"
            ).fetchone()["n"]
            try:
                client.audit_claim(SUPPORTED, scope="all", reporting_entity=ENTITY)
            except audit_module.AuditError:
                pass
            else:
                raise AssertionError("the audit should have raised")

            row = client.conn.execute(
                "SELECT * FROM audit_run ORDER BY id DESC LIMIT 1"
            ).fetchone()
            check(
                client.conn.execute("SELECT count(*) AS n FROM audit_run").fetchone()["n"]
                == before + 1,
                "the failed audit is on the record, not discarded",
            )
            check(row["status"] == "failed", "and is explicitly failed")
            check(row["verdict"] is None, "with no verdict")
            check(row["failure_code"] == "internal_error", "a safe failure code")
            check(bool(row["failure_phase"]), "the failing phase")
            check(row["retryable"] is True, "and its retryability")
            check(bool(row["requested_document_ids"]), "the searched corpus is kept")
            blob = json.dumps(dict(row), default=str)
            for secret in ("secret-sentinel-model-reply", "You decide whether", "postgresql://"):
                check(secret not in blob, f"the row omits {secret[:28]!r}")

            trace = client.get_audit_trace(int(row["id"]))
            check(trace.status == "failed", "the trace reports the failure")
            check(trace.failure_code == "internal_error", "with the safe code")
            check(trace.retryable is True, "and its retryability")
    finally:
        audit_module._deterministic = original


def check_scope_failure_creates_no_audit_row(tmp: Path) -> None:
    with make_client(default_session()) as client:
        before = client.conn.execute("SELECT count(*) AS n FROM audit_run").fetchone()["n"]
        _expect(
            NotFoundError,
            lambda: client.audit_claim(SUPPORTED, scope=[10_000_002], reporting_entity=ENTITY),
            "an invalid scope is rejected",
        )
        after = client.conn.execute("SELECT count(*) AS n FROM audit_run").fetchone()["n"]
        check(after == before, "and no audit row was opened for it")


# --- M-3: three identities, none of them standing in for another ------------


def check_source_hash_is_the_real_pdf_digest(tmp: Path) -> None:
    import hashlib

    from claim_evidence.source import sha256_file

    root = build_root(tmp / "hashed")
    pdf = tmp / "hashed.pdf"
    pdf.write_bytes(b"%PDF-1.7 " + b"content" * 100)
    expected = hashlib.sha256(pdf.read_bytes()).hexdigest()

    with make_client(default_session()) as client:
        report = client.ingest_document(root, source_pdf=pdf, reporting_entity=ENTITY)
        stored = client.conn.execute(
            "SELECT sha256, identity_key FROM document WHERE id = %s",
            (report.document_id,),
        ).fetchone()
        check(
            stored["sha256"] == expected == sha256_file(pdf),
            "source_sha256 is the digest sha256sum reports",
        )
        summary = client.get_document(report.document_id)
        check(summary.source_sha256 == expected, "and it is what the public API returns")

        # Identity is keyed on the PDF's own bytes (PD-04), so the public hash
        # and the internal key are derived from one fact about the document.
        from claim_evidence.ingest import identity_key

        check(
            stored["identity_key"] == identity_key(Path(root), expected, None),
            "identity_key is derived from the PDF's SHA-256",
        )
        again = client.ingest_document(root, source_pdf=pdf, reporting_entity=ENTITY)
        check(
            again.document_id == report.document_id and again.reused_existing,
            "re-ingesting the same PDF is the same document and a no-op",
        )


def check_improved_extraction_replaces_the_version(tmp: Path) -> None:
    """Same PDF, better output: a new version of one document, not a new one."""
    root = build_root(tmp / "improved")
    pdf = tmp / "improved.pdf"
    pdf.write_bytes(b"%PDF-1.7 stable")

    with make_client(default_session()) as client:
        first = client.ingest_document(root, source_pdf=pdf, reporting_entity=ENTITY)

        table = Path(root) / "page_0001" / "table_candidates.json"
        original = table.read_text(encoding="utf-8")
        table.write_text(original.replace("(40.2) %", "(41.2) %"), encoding="utf-8")
        check(len(table.read_text(encoding="utf-8")) == len(original), "same file length")

        second = client.ingest_document(root, source_pdf=pdf, reporting_entity=ENTITY)
        check(
            second.document_id == first.document_id,
            "an improved extraction is the same logical document",
        )
        check(
            second.version_id != first.version_id and not second.reused_existing,
            "but a new version, not silently treated as unchanged",
        )
        retired = client.conn.execute(
            "SELECT status FROM document_version WHERE id = %s", (first.version_id,)
        ).fetchone()["status"]
        check(retired == "inactive", "the previous version is retired, not deleted")


def check_page_image_change_rebuilds_the_version(tmp: Path) -> None:
    from PIL import Image

    root = build_root(tmp / "repainted")
    with make_client(default_session()) as client:
        first = client.ingest_document(root, source_uri="urn:repainted", reporting_entity=ENTITY)
        Image.new("RGB", (600, 800), "black").save(Path(root) / "page_0001" / "page.png")
        second = client.ingest_document(root, source_uri="urn:repainted", reporting_entity=ENTITY)
        check(
            second.version_id != first.version_id,
            "a redrawn page image invalidates the version that cites crops of it",
        )


# --- M-2: a failed build says so; health counts what can be queried ---------


def check_failed_build_is_recorded_and_isolated(tmp: Path) -> None:
    root = build_root(tmp / "failing")
    with make_client(default_session()) as client:
        ready = client.ingest_document(root, source_uri="urn:failing", reporting_entity=ENTITY)
        check(ready.status is VersionStatus.READY, "the first build is ready")

    import claim_evidence.ingest as ingest_module

    original = ingest_module._verify
    ingest_module._verify = lambda *a, **k: (_ for _ in ()).throw(
        ingest_module.IngestionError("simulated failure")
    )
    try:
        with make_client(default_session()) as client:
            try:
                client.ingest_document(root, source_uri="urn:failing", force=True, reporting_entity=ENTITY)
            except ingest_module.IngestionError:
                pass
            else:
                raise AssertionError("the forced rebuild should have failed")
    finally:
        ingest_module._verify = original

    with make_client(default_session()) as client:
        rows = client.conn.execute(
            "SELECT id, status, failure_code, failure_phase, failed_at"
            " FROM document_version WHERE document_id = %s ORDER BY id",
            (ready.document_id,),
        ).fetchall()
        failed = [r for r in rows if r["status"] == "failed"]
        check(len(failed) == 1, f"exactly one version is marked failed ({len(failed)})")
        check(failed[0]["failure_code"] == "internal_error", "a safe failure code is stored")
        check(bool(failed[0]["failure_phase"]), "the failing phase is stored")
        check(failed[0]["failed_at"] is not None, "the failure is timestamped")
        check(
            "simulated failure" not in json.dumps(dict(failed[0]), default=str),
            "the raw exception text is not stored",
        )

        still = client.conn.execute(
            "SELECT status FROM document_version WHERE id = %s", (ready.version_id,)
        ).fetchone()["status"]
        check(still == "ready", "the previous ready version is untouched")
        hits = client.search_evidence(SUPPORTED, document_ids=[ready.document_id])
        check(bool(hits), "and still answers queries")
        check(
            all(m.citation.document_id == ready.document_id for m in hits),
            "a failed version contributes no evidence to retrieval",
        )

        # The document is still ready, so a plain re-run is a no-op and the
        # failed forced attempt stays on the record as what it was.
        again = client.ingest_document(root, source_uri="urn:failing", force=False, reporting_entity=ENTITY)
        check(again.reused_existing, "a ready document is still a no-op after a failed rebuild")


def check_retrying_a_failed_first_build_clears_its_failure(tmp: Path) -> None:
    """No ready version to fall back on: the retry reopens the failed attempt."""
    root = build_root(tmp / "retry")
    import claim_evidence.ingest as ingest_module

    original = ingest_module._verify
    ingest_module._verify = lambda *a, **k: (_ for _ in ()).throw(
        ingest_module.IngestionError("simulated failure")
    )
    try:
        with make_client(default_session()) as client:
            try:
                client.ingest_document(root, source_uri="urn:retry", reporting_entity=ENTITY)
            except ingest_module.IngestionError:
                pass
            else:
                raise AssertionError("the first build should have failed")
    finally:
        ingest_module._verify = original

    with make_client(default_session()) as client:
        before = client.conn.execute(
            "SELECT id, status, failure_code FROM document_version"
            " WHERE fingerprint IN (SELECT fingerprint FROM document_version"
            "   WHERE output_root = %s) ORDER BY id DESC LIMIT 1",
            (str(Path(root).resolve()),),
        ).fetchone()
        check(before["status"] == "failed", "the first attempt is marked failed")

        report = client.ingest_document(root, source_uri="urn:retry", reporting_entity=ENTITY)
        check(report.status is VersionStatus.READY, "the retry reaches ready")
        check(
            report.version_id == int(before["id"]),
            "the retry reopens the same attempt rather than orphaning it",
        )
        after = client.conn.execute(
            "SELECT status, failure_code, failure_phase, failed_at"
            " FROM document_version WHERE id = %s",
            (before["id"],),
        ).fetchone()
        check(after["status"] == "ready", "and it is ready")
        check(
            after["failure_code"] is None
            and after["failure_phase"] is None
            and after["failed_at"] is None,
            "the previous failure metadata was cleared, not left to contradict it",
        )


def check_health_counts_only_the_queryable_index(tmp: Path) -> None:
    with make_client(default_session()) as client:
        report = client.health()
        check(report.database_reachable, "database reachable")
        check(report.schema_current, f"schema is current ({report.problems})")
        check(
            report.schema_embedding_dimensions == report.configured_embedding_dimensions
            == DIMENSIONS,
            "health reports both embedding dimensions",
        )

        ready_units = client.conn.execute(
            "SELECT count(*) AS n FROM evidence_unit e"
            " JOIN document_version v ON v.id = e.version_id WHERE v.status = 'ready'"
        ).fetchone()["n"]
        stored = client.conn.execute(
            "SELECT count(*) AS n FROM evidence_unit"
        ).fetchone()["n"]
        check(report.evidence_units == ready_units, "evidence total is the ready total")
        check(report.stored_evidence_units == stored, "stored total is reported separately")
        check(
            stored > ready_units,
            f"the two differ once a build failed ({stored} stored, {ready_units} ready)",
        )


def check_stale_build_reports_as_interrupted(tmp: Path) -> None:
    root = build_root(tmp / "stale")
    with make_client(default_session()) as client:
        document_id = client.conn.execute(
            "INSERT INTO document (name, identity_key) VALUES ('stale', 'k-stale')"
            " RETURNING id"
        ).fetchone()["id"]
        version_id = client.conn.execute(
            "INSERT INTO document_version"
            " (document_id, fingerprint, embed_model, embed_dim, output_root)"
            " VALUES (%s, 'fp-stale', 'fake', %s, %s) RETURNING id",
            (document_id, DIMENSIONS, str(root)),
        ).fetchone()["id"]
        client.conn.commit()

        fresh = client.health()
        check(fresh.documents_building >= 1, "a just-started build counts as building")
        check(fresh.documents_interrupted == 0, "and is not called interrupted")

        client.conn.execute(
            "UPDATE document_version SET last_progress_at = now() - interval '2 hours'"
            " WHERE id = %s",
            (version_id,),
        )
        client.conn.commit()

        stale = client.health()
        check(stale.documents_interrupted >= 1, "a silent build counts as interrupted")
        check(
            stale.documents_building == fresh.documents_building - 1,
            "and no longer counts as actively building",
        )
        status = client.conn.execute(
            "SELECT status FROM document_version WHERE id = %s", (version_id,)
        ).fetchone()["status"]
        check(status == "building", "health classified it without mutating the row")

        client.conn.execute("DELETE FROM document WHERE id = %s", (document_id,))
        client.conn.commit()


# --- M-5: a mismatched vector column is not a healthy database --------------


def check_embedding_dimension_mismatch_is_detected(tmp: Path) -> None:
    from dataclasses import replace

    config = settings()
    widened = replace(config, embed_dimensions=DIMENSIONS + 4)
    with ClaimEvidence(widened, connect(config.database_url), OllamaClient(widened, default_session())) as client:
        try:
            client.init_db()
        except IndexNotReadyError as exc:
            message = str(exc)
            check(str(DIMENSIONS) in message and str(DIMENSIONS + 4) in message,
                  f"both dimensions are named ({message})")
            check("postgresql://" not in message, "no connection string is exposed")
        else:
            raise AssertionError("a dimension mismatch was accepted")

        report = client.health()
        check(report.database_reachable, "the database is still reported reachable")
        check(not report.schema_current, "but the schema is not current")
        check(
            report.schema_embedding_dimensions == DIMENSIONS
            and report.configured_embedding_dimensions == DIMENSIONS + 4,
            "health reports the declared and configured dimensions",
        )
        check(
            any("does not match configured" in p for p in report.problems),
            f"and explains the mismatch ({report.problems})",
        )

    with make_client(default_session()) as client:
        client.init_db()
        check(client.health().schema_current, "the configured dimension still initializes")


def _visual_ids(payload: dict[str, Any]) -> list[int]:
    """The ids of the visual passages actually in the prompt.

    Read off the `<evidence>` tag, which is where an id lives -- the older
    bracketed form this matched stopped being emitted, so it matched nothing and
    quietly turned every assertion built on it into a no-op.
    """
    import re

    prompt = payload["messages"][1]["content"]
    return [
        int(m)
        for m in re.findall(r'<evidence id="(\d+)"[^>]*kind="visual"', prompt)
    ]


def main() -> int:
    try:
        reset_database()
    except psycopg.OperationalError as exc:
        print(f"[skip] postgres unavailable ({exc.__class__.__name__}); run: docker compose up -d")
        return 0

    import tempfile

    random.seed(0)
    checks = [
        check_schema_is_repeatable,
        check_ingestion_is_idempotent,
        check_indexes_are_used,
        check_hybrid_search_finds_exact_and_semantic,
        check_vector_recall,
        check_supported_claim_cites_the_table,
        check_direct_evidence_never_reaches_the_fallback,
        check_adjudicator_prompt_is_bounded,
        check_contradicted_claim,
        check_vague_claim_is_insufficient,
        check_verdict_fails_closed_without_citable_evidence,
        check_interrupted_build_is_invisible,
        check_visual_evidence_needs_crop_verification,
        check_a_conflicting_crop_is_evidence_against_the_claim,
        check_middle_range_keeps_pdf_pages,
        check_source_less_documents_stay_separate,
        check_shared_source_uri_is_one_document,
        check_a_legacy_schema_is_refused_not_migrated,
        check_resume_reconciles_the_building_version,
        check_supported_claim_explains_itself,
        check_contradicted_claim_explains_the_conflict,
        check_vague_claim_explains_the_scope_gap,
        check_explanation_round_trips_through_the_trace,
        check_explanation_carries_no_model_text,
        check_expansion_follows_source_order_not_ids,
        check_narrative_blocks_expand_in_page_order,
        check_expansion_uses_its_indexes,
        check_audit_persists_its_scope_and_status,
        check_trace_survives_document_removal,
        check_failed_audit_is_explicit_and_safe,
        check_scope_failure_creates_no_audit_row,
        check_source_hash_is_the_real_pdf_digest,
        check_improved_extraction_replaces_the_version,
        check_page_image_change_rebuilds_the_version,
        check_failed_build_is_recorded_and_isolated,
        check_retrying_a_failed_first_build_clears_its_failure,
        check_health_counts_only_the_queryable_index,
        check_stale_build_reports_as_interrupted,
        check_embedding_dimension_mismatch_is_detected,
        check_document_scope_is_validated,
        check_invalid_scope_costs_nothing,
        check_backend_mismatch_costs_nothing,
        check_llamacpp_runs_the_public_pipeline,
        # Near the end: these add their own documents, and the checks above
        # audit with scope="all".
        check_mapped_markdown_bridges_two_blocks,
        check_unmapped_markdown_cannot_answer,
        check_repaired_table_row_is_offered_only_as_fallback,
        check_a_row_with_an_unmapped_cell_offers_no_context,
        # Last: it empties the index the checks above rely on.
        check_empty_index_is_not_an_insufficient_verdict,
    ]
    with tempfile.TemporaryDirectory() as temp:
        tmp = Path(temp)
        for func in checks:
            print(f"\n-- {func.__name__}")
            func(tmp)
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
