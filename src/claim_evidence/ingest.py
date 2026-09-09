"""Ingestion: completed output root -> queryable, ready document version.

Resumable by construction. Evidence, embeddings, and facts all upsert on stable
keys, so a re-run picks up where an interrupted one stopped instead of
duplicating rows. The version only flips to ``ready`` after the integrity
checks pass, so a half-built index is never visible to a query.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Sequence

import psycopg
from document_extract.contracts import PAGE_ARTIFACT_ROLES

from .config import Settings
from .db import (
    QUERYABLE_STATUSES,
    activate_version,
    add_alias,
    clear_fact_failures,
    delete_facts,
    delete_stale_pages,
    failed_fact_candidates,
    find_version,
    init_schema,
    mark_version_failed,
    note_progress,
    record_fact_failure,
    set_embeddings,
    set_fact_coverage,
    start_version,
    units_missing_embeddings,
    upsert_document,
    upsert_entity,
    upsert_evidence,
    upsert_fact,
    upsert_page,
    vector_dimension,
)
from .facts import (
    FACT_EXTRACTION_SYSTEM,
    FACT_PROMPT_VERSION,
    FACT_SCHEMA_VERSION,
    accept_llm_facts,
    fact_extraction_prompt,
    is_claim_like,
    normalized_name,
    organization_name,
    table_facts,
)
from .models import (
    EvidenceKind,
    EvidenceUnit,
    Fact,
    FactExtraction,
    IngestReport,
    VersionStatus,
)
from .progress import ProgressCallback, ProgressReporter, classify_error
from .normalize import NORMALIZATION_VERSION, normalize_for_match
from .errors import (
    ClaimEvidenceError,
    IndexNotReadyError,
    NotFoundError,
    ValidationError,
)
from .ollama import OllamaClient, OllamaError
from .source import OutputReader, canonical_digest, page_units, sha256_file

logger = logging.getLogger(__name__)

# Table values are reached through their row, their fact, and lexical search.
# Embedding 11k near-identical "(40.2) %" strings buys nothing semantically and
# costs the bulk of ingestion time.
# ponytail: skip value-cell embeddings; embed them too if vector recall on
# bare cell text ever proves necessary.
EMBEDDED_KINDS = (
    str(EvidenceKind.NARRATIVE),
    str(EvidenceKind.TABLE_ROW),
    str(EvidenceKind.VISUAL),
    str(EvidenceKind.PAGE_MARKDOWN),
)


class IngestionError(ClaimEvidenceError):
    """Ingestion could not complete; the version stays invisible to queries."""


# The integrity gate runs a fixed set of checks, which is what makes
# `building_indexes` a measurable phase rather than a spinner.
VERIFY_STEPS = (
    "evidence count",
    "page coverage",
    "page links",
    "citation paths",
    "artifact containment",
    "source order",
    "markdown mapping",
    "embedding coverage",
    "embedding dimension",
    "fact links",
)


# Bumped when the *meaning* of an identity basis changes, so a redefinition
# cannot silently equate old and new keys.
IDENTITY_VERSION = 1
# 2: page Markdown is indexed per mapped segment carrying the source keys it
# was generated from, instead of one whole-page blob. Same extraction output,
# different stored evidence, so every existing index has to be rebuilt.
# 3: only typed, unhedged chart summaries become visual units, an ambiguous
# literal fails its Markdown segment, and a repaired table row maps cell by
# cell. A v2 index holds units these rules would refuse, and no query-time
# filter can remove a row that should never have been stored.
FINGERPRINT_VERSION = 3


def normalize_source_uri(uri: str) -> str:
    """Normalize a logical URI to the bounded extent PD-04 supports.

    Scheme and host are case-insensitive by RFC; the path is not, and lowering
    it would merge two genuinely different documents. Only the file-URI and
    local-Windows forms this prototype is tested against are supported --
    exhaustive canonicalization is deferred (PV-010).
    """
    text = uri.strip()
    scheme, separator, rest = text.partition("://")
    if not separator:
        return text
    host, slash, path = rest.partition("/")
    return f"{scheme.lower()}://{host.lower()}{slash}{path}".rstrip("/")


def canonical_local_path(root: Path) -> str:
    """A local Windows extraction-output path in one settled spelling.

    ``resolve(strict=True)`` requires the directory to exist, which is the
    point: a path-based identity for something that is not there identifies
    nothing. ``normcase`` settles Windows' case-insensitivity and separator
    choice. Moving a path-only source therefore creates a new logical document,
    which is the accepted limitation (GR-063), not an oversight.
    """
    try:
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValidationError(
            "the extraction output root does not exist, so it cannot identify a "
            "document; pass the source PDF or a logical source URI"
        ) from exc
    return os.path.normcase(str(resolved))


def identity_basis(
    root: Path, source_sha256: str | None, source_uri: str | None
) -> tuple[str, str]:
    """The strongest available basis for this document's identity, and its kind.

    In PD-04's order: the PDF's own bytes, then a logical URI, then the local
    output path. The kind travels with the value so a URI and a path that
    happen to read the same are still two documents.
    """
    if source_sha256:
        return "pdf_sha256", source_sha256.lower()
    if source_uri and source_uri.strip():
        return "source_uri", normalize_source_uri(source_uri)
    return "local_output_path", canonical_local_path(root)


def identity_key(
    root: Path, source_sha256: str | None, source_uri: str | None
) -> str:
    """The document's internal identity: a digest of a versioned, tagged basis.

    Never a public field. The basis kind is inside the digest, so the same text
    in two roles cannot collide, and the version is inside it so redefining a
    basis later cannot silently equate the old and new meanings.
    """
    kind, value = identity_basis(root, source_sha256, source_uri)
    return canonical_digest(
        {"identity_version": IDENTITY_VERSION, "basis": kind, "value": value}
    )


def index_fingerprint(
    source_sha256: str | None,
    artifacts_sha256: str,
    settings: Settings,
    *,
    extraction_contract_version: str,
    extraction_settings: dict[str, Any],
    reporting_entity: str,
    fact_mode: str,
    embed_model_identity: dict[str, Any],
    fact_model_identity: dict[str, Any],
) -> str:
    """Everything this version would be built from, as one canonical value.

    The rule for what belongs here is simple and worth stating: if changing it
    changes a stored evidence unit, embedding, or fact, it is in. If it only
    changes how long the build takes or where it connects, it is out --
    timeouts, batch sizes, URLs, and UI settings would otherwise invalidate a
    perfectly good index every time someone tuned them. Audit-time model
    settings are out for the same reason: they cannot alter what was stored.
    """
    payload = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "source_pdf_sha256": (source_sha256 or "").lower(),
        "extraction": {
            "contract_version": extraction_contract_version,
            # The settings the extractor declares as evidence-affecting: the
            # visual-value mode, the models, the prompt digests. The artifact
            # hashes alone would miss a re-extraction that produced identical
            # files under different rules, and would also make every re-run
            # differ if run.json were hashed whole -- it carries a timestamp.
            "settings": extraction_settings,
            "artifacts_sha256": artifacts_sha256,
        },
        "evidence_normalization_version": NORMALIZATION_VERSION,
        # Every stored fact is attributed to it, so a different entity is a
        # different build.
        "reporting_entity": reporting_entity,
        "embedding": {
            **embed_model_identity,
            "dimensions": settings.embed_dimensions,
        },
        "facts": {
            "mode": fact_mode,
            "model": fact_model_identity,
            "prompt_version": FACT_PROMPT_VERSION,
            "schema_version": FACT_SCHEMA_VERSION,
            "num_ctx": settings.num_ctx,
            # The only generation options this package sets, stated rather than
            # assumed: a future change to them changes what facts come out.
            "options": {"temperature": 0},
        },
    }
    return canonical_digest(payload)


def build_fingerprint(
    reader: OutputReader,
    settings: Settings,
    client: OllamaClient,
    *,
    source_sha256: str | None,
    reporting_entity: str,
    extract_narrative_facts: bool,
) -> str:
    """The fingerprint for the build these inputs describe.

    One place assembles it, so the value ingestion stores and the value a
    caller computes to ask "would this rebuild?" cannot drift apart.
    """
    return index_fingerprint(
        source_sha256,
        reader.extraction_fingerprint(),
        settings,
        extraction_contract_version=reader.run["contract_version"],
        extraction_settings=reader.run["settings"],
        reporting_entity=reporting_entity,
        fact_mode="table_and_narrative" if extract_narrative_facts else "table_only",
        embed_model_identity=client.model_identity(settings.embed_model),
        fact_model_identity=(
            client.model_identity(settings.chat_model)
            if extract_narrative_facts
            # Not consulted when no narrative facts are extracted, and naming a
            # model that took no part in the build would make the fingerprint
            # depend on something that did not affect it.
            else {"name": None, "digest": None, "reproducibility": "not_used"}
        ),
    )


def ingest_document(
    conn: psycopg.Connection,
    client: OllamaClient,
    settings: Settings,
    output_root: str | Path,
    *,
    reporting_entity: str,
    source_pdf: str | Path | None = None,
    source_uri: str | None = None,
    force: bool = False,
    extract_narrative_facts: bool = True,
    progress: ProgressCallback | None = None,
) -> IngestReport:
    entity = (reporting_entity or "").strip()
    if not entity:
        raise ValidationError(
            "reporting_entity is required: the facts this build stores are"
            " attributed to it, and a document's filename is not an entity"
        )
    reporter = ProgressReporter(progress, "ingest")
    logger.info(
        "ingestion started",
        extra={
            "event": "ingestion_started", "operation": "ingest",
            "document_id": None, "force": force,
            "extract_narrative_facts": extract_narrative_facts,
        },
    )
    # Set once a version exists, so a failure before that point has nothing to
    # mark and a failure after it marks exactly one row.
    building: list[int] = []
    try:
        return _ingest(
            conn,
            client,
            settings,
            output_root,
            reporting_entity=entity,
            source_pdf=source_pdf,
            source_uri=source_uri,
            force=force,
            extract_narrative_facts=extract_narrative_facts,
            reporter=reporter,
            building=building,
        )
    except BaseException as exc:
        # One safe terminal event, then the original error reaches the caller
        # unchanged. A failed version is never activated.
        reporter.fail(exc)
        if building:
            _record_failure(conn, building[0], exc, reporter.phase)
        raise


def _record_failure(
    conn: psycopg.Connection, version_id: int, exc: BaseException, phase: str
) -> None:
    """Persist why the build stopped, without ever masking the original error.

    The exception on its way out is the caller's answer; a database that is
    itself unhappy must not replace it with a second, less useful one.
    """
    code, _retryable, _message = classify_error(exc)
    try:
        # The failing statement may have poisoned the transaction; a rollback
        # is what makes the status write possible at all.
        conn.rollback()
        mark_version_failed(conn, version_id, code, phase or "unknown")
    except Exception:  # noqa: BLE001 - never mask the original failure
        logger.error(
            "could not persist ingestion failure",
            extra={
                "event": "ingestion_failure_persistence_failed",
                "version_id": version_id, "error_type": "database_error",
            },
        )


def _ingest(
    conn: psycopg.Connection,
    client: OllamaClient,
    settings: Settings,
    output_root: str | Path,
    *,
    reporting_entity: str,
    source_pdf: str | Path | None,
    source_uri: str | None,
    force: bool,
    extract_narrative_facts: bool,
    reporter: ProgressReporter,
    building: list[int],
) -> IngestReport:
    reader = OutputReader(output_root)
    reporter.start("validating_input", "Validating the extraction output")
    pages = reader.validate()
    reporter.done(
        "validating_input",
        f"Validated {len(pages)} pages",
        completed=len(pages),
        total=len(pages),
    )

    source_pdf_path = Path(source_pdf) if source_pdf else None
    document_name = (source_pdf_path or reader.root).name
    # Two values doing two jobs, deliberately not one doing both: the public
    # source hash anyone can verify with `sha256sum`, and the fingerprint of
    # everything this version would be built from. Without a PDF the source
    # hash is null rather than quietly holding an extraction digest under a
    # field named for the source.
    source_sha256 = sha256_file(source_pdf_path) if source_pdf_path else None
    fingerprint = build_fingerprint(
        reader,
        settings,
        client,
        source_sha256=source_sha256,
        reporting_entity=reporting_entity,
        extract_narrative_facts=extract_narrative_facts,
    )

    document_id = upsert_document(
        conn,
        document_name,
        source_sha256,
        source_uri,
        identity_key(reader.root, source_sha256, source_uri),
    )
    reporter.document_id = document_id
    logger.info(
        "document resolved for ingestion",
        extra={
            "event": "ingestion_document_resolved", "operation": "ingest",
            "document_id": document_id, "page_count": len(pages),
        },
    )
    existing = find_version(conn, document_id, fingerprint)
    # Reuse requires an exact fingerprint *and* a queryable state. A degraded
    # version qualifies -- rebuilding it from scratch would re-derive identical
    # evidence to chase the same optional facts; `retry_facts` is the cheaper
    # and more targeted way to finish it.
    if existing and existing["status"] in QUERYABLE_STATUSES and not force:
        version_id = int(existing["id"])
        failed_keys = [
            row["unit_key"] for row in failed_fact_candidates(conn, version_id)
        ]
        report = IngestReport(
            document_id=document_id,
            version_id=version_id,
            status=VersionStatus(existing["status"]),
            fingerprint=fingerprint,
            reused_existing=True,
            pages=len(pages),
            fact_candidates_total=int(existing["fact_candidates_total"]),
            fact_candidates_succeeded=int(existing["fact_candidates_succeeded"]),
            failed_fact_candidates=failed_keys,
            warnings=reader.warnings,
            **_stored_counts(conn, version_id),
        )
        reporter.done(
            "completed",
            "Document is already indexed",
            completed=1,
            total=1,
            details=_completion_details(report, reporter),
        )
        logger.info(
            "ingestion completed using existing version",
            extra={
                "event": "ingestion_completed", "operation": "ingest",
                "document_id": document_id, "version_id": version_id,
                "status": str(report.status), "reused_existing": True,
                "page_count": report.pages, "evidence_count": report.evidence_units,
                "fact_count": report.facts,
                "duration_seconds": reporter.elapsed_seconds,
            },
        )
        return report

    version_id = start_version(
        conn,
        document_id,
        fingerprint,
        embed_model=settings.embed_model,
        embed_dim=settings.embed_dimensions,
        output_root=str(reader.root),
        source_pdf=str(source_pdf_path) if source_pdf_path else None,
        force=force,
    )
    building.append(version_id)
    logger.info(
        "document version opened",
        extra={
            "event": "ingestion_version_started", "operation": "ingest",
            "document_id": document_id, "version_id": version_id, "force": force,
        },
    )

    # The reporting entity is stated by the caller, not derived from the
    # document's basename. `organization_name("danoneurdaccessible.pdf")` made a
    # filename into a company and attributed every figure in the document to
    # that string; a later audit then compared a real entity against it.
    subject = reporting_entity
    subject_entity = upsert_entity(conn, "organization", subject, normalized_name(subject))
    # The filename stays as an alias -- useful for finding the entity, never
    # used as its name.
    add_alias(conn, subject_entity, document_name, normalized_name(document_name))
    conn.commit()

    blocks_by_page = reader.blocks_by_page()
    evidence_ids: dict[str, int] = {}
    all_units: list[EvidenceUnit] = []
    unit_count = 0
    warned = 0

    reporter.start("building_evidence", "Building evidence units", total=len(pages))
    for index, page in enumerate(pages, start=1):
        units = list(page_units(reader, page, blocks_by_page.get(page.page, [])))
        page_id = upsert_page(
            conn, version_id, page.page, page.width, page.height, page.rel
        )
        evidence_ids.update(upsert_evidence(conn, version_id, page_id, units))
        note_progress(conn, version_id)
        conn.commit()
        all_units.extend(units)
        unit_count += len(units)
        reporter.step(
            "building_evidence",
            f"Indexed page {page.page}",
            completed=index,
            total=len(pages),
            current_item=f"page {page.page}",
        )
        warned = _warn_new(reporter, "building_evidence", reader, warned)
    # A resumed attempt can still hold pages an earlier manifest listed.
    delete_stale_pages(conn, version_id, [p.page for p in pages])
    conn.commit()
    reporter.done(
        "building_evidence",
        f"Built {unit_count} evidence units",
        completed=len(pages),
        total=len(pages),
    )

    _embed_pending(conn, client, settings, version_id, reporter)
    # Drain the reader's warnings here, before fact extraction: _build_facts
    # emits its own, and draining afterwards would send each of those twice.
    _warn_new(reporter, "embedding_evidence", reader, warned)
    facts, rejected = _build_facts(
        conn,
        client,
        version_id,
        subject,
        subject_entity,
        all_units,
        evidence_ids,
        reader,
        extract_narrative_facts,
        reporter,
    )

    embedded = _verify(
        conn, version_id, unit_count, len(pages), settings, reporter, reader.root
    )

    # PD-06: evidence, provenance, and embeddings are required and have all
    # passed by now. Narrative facts are enrichment, so a partial failure among
    # them is `degraded` -- queryable, with the shortfall counted -- rather than
    # a failed build or, worse, a `ready` one that quietly holds fewer facts
    # than it should and gets reused as if it were complete.
    failed_keys = [row["unit_key"] for row in failed_fact_candidates(conn, version_id)]
    status = VersionStatus.DEGRADED if failed_keys else VersionStatus.READY

    reporter.start("activating_version", "Activating the new version")
    activate_version(conn, version_id, str(status))
    conn.commit()
    reporter.done(
        "activating_version",
        f"Version activated as {status}",
        completed=1,
        total=1,
    )

    coverage = conn.execute(
        "SELECT fact_candidates_total AS total, fact_candidates_succeeded AS ok"
        " FROM document_version WHERE id = %s",
        (version_id,),
    ).fetchone()

    report = IngestReport(
        document_id=document_id,
        version_id=version_id,
        status=status,
        fingerprint=fingerprint,
        fact_candidates_total=int(coverage["total"]),
        fact_candidates_succeeded=int(coverage["ok"]),
        failed_fact_candidates=failed_keys,
        pages=len(pages),
        evidence_units=unit_count,
        visual_evidence_units=sum(
            1 for unit in all_units if unit.kind is EvidenceKind.VISUAL
        ),
        embedded_units=embedded,
        facts=facts,
        warnings=reader.warnings,
        skipped_artifacts=sorted(set(reader.skipped)),
        rejected_facts=rejected,
    )
    reporter.done(
        "completed",
        f"Indexed {report.pages} pages into {report.evidence_units} evidence units",
        completed=1,
        total=1,
        details=_completion_details(report, reporter),
    )
    logger.info(
        "ingestion completed",
        extra={
            "event": "ingestion_completed", "operation": "ingest",
            "document_id": document_id, "version_id": version_id,
            "status": str(report.status), "reused_existing": False,
            "page_count": report.pages, "evidence_count": report.evidence_units,
            "embedded_count": report.embedded_units, "fact_count": report.facts,
            "warning_count": len(report.warnings),
            "duration_seconds": reporter.elapsed_seconds,
        },
    )
    return report


def _completion_details(report: IngestReport, reporter: ProgressReporter) -> dict[str, Any]:
    """Counts the caller already paid for, echoed onto the terminal event.

    Every value comes off the finished report, so showing completion statistics
    costs no extra queries.
    """
    return {
        "document_version_id": report.version_id,
        "page_count": report.pages,
        "evidence_count": report.evidence_units,
        "visual_evidence_count": report.visual_evidence_units,
        "embedding_count": report.embedded_units,
        "fact_count": report.facts,
        "warning_count": len(report.warnings),
        "elapsed_seconds": reporter.elapsed_seconds,
        "no_op": report.reused_existing,
    }


def _warn_new(
    reporter: ProgressReporter, phase: str, reader: OutputReader, already: int
) -> int:
    """Surface recoverable fallbacks the reader recorded, once each."""
    for index, warning in enumerate(reader.warnings[already:], start=already + 1):
        reporter.warn(phase, warning)
        logger.warning(
            "source artifact required a recoverable fallback",
            extra={
                "event": "source_warning", "operation": "ingest",
                "document_id": reporter.document_id, "phase": phase,
                "warning_index": index,
            },
        )
    return len(reader.warnings)


def _stored_counts(conn: psycopg.Connection, version_id: int) -> dict[str, int]:
    row = conn.execute(
        """
        SELECT
            (SELECT count(*) FROM evidence_unit WHERE version_id = %(v)s) AS units,
            (SELECT count(*) FROM evidence_unit
             WHERE version_id = %(v)s AND kind = 'visual') AS visual,
            (SELECT count(*) FROM evidence_unit
             WHERE version_id = %(v)s AND embedding IS NOT NULL) AS embedded,
            (SELECT count(*) FROM fact WHERE version_id = %(v)s) AS facts
        """,
        {"v": version_id},
    ).fetchone()
    return {
        "evidence_units": int(row["units"]),
        "visual_evidence_units": int(row["visual"]),
        "embedded_units": int(row["embedded"]),
        "facts": int(row["facts"]),
    }


def _embed_pending(
    conn: psycopg.Connection,
    client: OllamaClient,
    settings: Settings,
    version_id: int,
    reporter: ProgressReporter,
) -> int:
    """Embed everything not yet embedded, one batch per HTTP call.

    The database transaction is committed between batches, so a slow model
    never holds a write transaction open.
    """
    pending = units_missing_embeddings(conn, version_id, EMBEDDED_KINDS)
    embedded = 0
    size = max(1, settings.embed_batch_size)
    batches = -(-len(pending) // size)  # ceil, and 0 when nothing is pending
    reporter.start("embedding_evidence", "Embedding evidence", total=batches)
    logger.info(
        "evidence embedding started",
        extra={
            "event": "embedding_started", "operation": "ingest",
            "document_id": reporter.document_id, "version_id": version_id,
            "pending_units": len(pending), "batch_count": batches,
        },
    )
    for number, start in enumerate(range(0, len(pending), size), start=1):
        chunk = pending[start : start + size]
        vectors = client.embed([row["normalized_text"] or " " for row in chunk])
        set_embeddings(conn, [(int(r["id"]), v) for r, v in zip(chunk, vectors)])
        note_progress(conn, version_id)
        conn.commit()
        embedded += len(chunk)
        reporter.step(
            "embedding_evidence",
            f"Embedded batch {number} of {batches}",
            completed=number,
            total=batches,
            current_item=f"batch {number}",
        )
    reporter.done(
        "embedding_evidence",
        f"Embedded {embedded} evidence units",
        completed=batches,
        total=batches,
    )
    logger.info(
        "evidence embedding completed",
        extra={
            "event": "embedding_completed", "operation": "ingest",
            "document_id": reporter.document_id, "version_id": version_id,
            "embedded_units": embedded, "batch_count": batches,
            "duration_seconds": reporter.durations().get("embedding_evidence"),
        },
    )
    return embedded


def _build_facts(
    conn: psycopg.Connection,
    client: OllamaClient,
    version_id: int,
    subject: str,
    subject_entity: int,
    units: Sequence[EvidenceUnit],
    evidence_ids: dict[str, int],
    reader: OutputReader,
    extract_narrative_facts: bool,
    reporter: ProgressReporter,
) -> tuple[int, list[str]]:
    stored = 0
    rejected: list[str] = []

    # Rebuild rather than resume: every current unit is reprocessed below, so
    # keeping an earlier attempt's facts only risks stale values, qualifiers,
    # and evidence links surviving into a ready version.
    delete_facts(conn, version_id)
    conn.commit()

    candidates = (
        [
            unit
            for unit in units
            if unit.kind is EvidenceKind.NARRATIVE and is_claim_like(unit.text)
        ]
        if extract_narrative_facts
        else []
    )
    reporter.start("extracting_facts", "Extracting facts", total=len(candidates))
    logger.info(
        "fact extraction started",
        extra={
            "event": "fact_extraction_started", "operation": "ingest",
            "document_id": reporter.document_id, "version_id": version_id,
            "candidate_count": len(candidates),
            "narrative_enabled": extract_narrative_facts,
        },
    )

    for fact in table_facts(units, subject):
        stored += _store_fact(conn, version_id, fact, subject_entity, evidence_ids)
    conn.commit()

    if not extract_narrative_facts:
        reporter.done(
            "extracting_facts",
            f"Derived {stored} table facts",
            completed=0,
            total=0,
        )
        logger.info(
            "fact extraction completed",
            extra={
                "event": "fact_extraction_completed", "operation": "ingest",
                "document_id": reporter.document_id, "version_id": version_id,
                "candidate_count": 0, "successful_candidates": 0,
                "stored_facts": stored, "rejected_facts": 0,
                "duration_seconds": reporter.durations().get("extracting_facts"),
            },
        )
        return stored, rejected

    succeeded = 0
    for index, unit in enumerate(candidates, start=1):
        try:
            extraction = client.structured(
                FactExtraction,
                FACT_EXTRACTION_SYSTEM,
                fact_extraction_prompt(unit, subject),
            )
        except OllamaError as exc:
            # One passage the model could not process is not a failed index --
            # but it is not a silent success either. The key and a category are
            # recorded so a retry can process exactly this candidate again; the
            # passage and the model's error text are not stored.
            record_fact_failure(conn, version_id, unit.unit_key, "model_unavailable")
            conn.commit()
            reader.warnings.append(f"fact extraction failed for {unit.unit_key}")
            reporter.warn(
                "extracting_facts", "A passage could not be processed by the model"
            )
            logger.warning(
                "fact extraction deferred after model failure",
                extra={
                    "event": "fact_extraction_fallback", "operation": "ingest",
                    "document_id": reporter.document_id, "version_id": version_id,
                    "evidence_id": evidence_ids.get(unit.unit_key),
                    "unit_key": unit.unit_key, "error_type": type(exc).__name__,
                },
            )
            continue
        kept, dropped = accept_llm_facts(extraction.facts, unit, subject)
        rejected.extend(dropped)
        for fact in kept:
            stored += _store_fact(conn, version_id, fact, subject_entity, evidence_ids)
        succeeded += 1
        clear_fact_failures(conn, version_id, [unit.unit_key])
        note_progress(conn, version_id)
        conn.commit()
        reporter.step(
            "extracting_facts",
            f"Examined {index} of {len(candidates)} passages",
            completed=index,
            total=len(candidates),
            current_item=unit.unit_key,
        )

    set_fact_coverage(conn, version_id, total=len(candidates), succeeded=succeeded)
    conn.commit()
    reporter.done(
        "extracting_facts",
        f"Stored {stored} facts from {succeeded} of {len(candidates)} passages",
        completed=len(candidates),
        total=len(candidates),
    )
    logger.info(
        "fact extraction completed",
        extra={
            "event": "fact_extraction_completed", "operation": "ingest",
            "document_id": reporter.document_id, "version_id": version_id,
            "candidate_count": len(candidates), "successful_candidates": succeeded,
            "stored_facts": stored, "rejected_facts": len(rejected),
            "duration_seconds": reporter.durations().get("extracting_facts"),
        },
    )
    return stored, rejected


def _store_fact(
    conn: psycopg.Connection,
    version_id: int,
    fact: Fact,
    subject_entity: int,
    evidence_ids: dict[str, int],
) -> int:
    ids = [evidence_ids[key] for key in fact.evidence_keys if key in evidence_ids]
    if not ids:
        return 0
    metric_entity = upsert_entity(
        conn, "metric", fact.metric, normalized_name(fact.metric)
    )
    fact_id = upsert_fact(
        conn,
        version_id,
        fact,
        normalize_for_match(fact.metric),
        subject_entity,
        ids,
    )
    logger.debug(
        "fact stored",
        extra={
            "event": "fact_stored", "operation": "ingest",
            "version_id": version_id, "fact_id": fact_id,
            "evidence_ids": ids, "extraction_method": fact.extraction_method,
            "value": str(fact.value_decimal) if fact.value_decimal is not None else None,
            "value_text_present": fact.value_text is not None, "unit": fact.unit,
            "direction": fact.direction, "metric_chars": len(fact.metric),
            "scope_chars": len(fact.scope or ""),
            "reporting_period": fact.reporting_period,
            "baseline_period": fact.baseline_period,
        },
    )
    del metric_entity  # registered for graph traversal, not needed inline
    return 1


def _verify(
    conn: psycopg.Connection,
    version_id: int,
    expected_units: int,
    expected_pages: int,
    settings: Settings,
    reporter: ProgressReporter,
    root: Path,
) -> int:
    """The gate between a built version and a queryable one.

    Everything a resumed attempt could leave inconsistent is checked here:
    counts, stale pages, orphaned links, missing or wrong-width embeddings, and
    fact links that reach outside this version. Any failure raises, which
    leaves the version in ``building`` and whatever was ready still serving.

    Returns the version's embedding count, taken from the check query it
    already runs: a resumed build only writes the embeddings that were still
    missing, so "written this run" understates the finished index.
    """
    total = len(VERIFY_STEPS)
    reporter.start("building_indexes", "Checking index integrity", total=total)

    def passed(step: int, message: str) -> None:
        reporter.step(
            "building_indexes", message, completed=step, total=total,
            current_item=VERIFY_STEPS[step - 1],
        )

    counts = conn.execute(
        "SELECT count(*) AS n, count(embedding) AS embedded"
        " FROM evidence_unit WHERE version_id = %s",
        (version_id,),
    ).fetchone()
    stored = counts["n"]
    if stored != expected_units:
        raise IngestionError(f"stored {stored} evidence units, expected {expected_units}")
    passed(1, "Evidence count verified")

    pages = conn.execute(
        "SELECT count(*) AS n FROM page WHERE version_id = %s", (version_id,)
    ).fetchone()["n"]
    if pages != expected_pages:
        raise IngestionError(f"stored {pages} pages, expected {expected_pages}")
    passed(2, "Page coverage verified")

    orphans = conn.execute(
        """
        SELECT count(*) AS n FROM evidence_unit e
        LEFT JOIN page p ON p.id = e.page_id
        WHERE e.version_id = %s AND p.id IS NULL
        """,
        (version_id,),
    ).fetchone()["n"]
    if orphans:
        raise IngestionError(f"{orphans} evidence units have no page")
    passed(3, "Page links verified")

    uncited = conn.execute(
        """
        SELECT count(*) AS n FROM evidence_unit e
        WHERE e.version_id = %s AND e.citable
          AND NOT EXISTS (SELECT 1 FROM evidence_region r WHERE r.evidence_id = e.id)
        """,
        (version_id,),
    ).fetchone()["n"]
    if uncited:
        raise IngestionError(f"{uncited} citable units have no region to highlight")
    passed(4, "Citation paths verified")

    _verify_artifacts(conn, version_id, root)
    passed(5, "Artifact containment verified")

    unordered = conn.execute(
        """
        SELECT count(*) AS n FROM evidence_unit
        WHERE version_id = %s AND (source_order IS NULL
              OR (citable AND kind IN ('table_row', 'table_value')
                  AND context_key IS NULL))
        """,
        (version_id,),
    ).fetchone()["n"]
    if unordered:
        # Without a complete source order, context expansion silently falls
        # back to insertion order, which is the bug this replaces rather than a
        # graceful degradation of it.
        raise IngestionError(
            f"{unordered} units have no source order or no context key"
        )
    passed(6, "Source order verified")

    # Generated Markdown is only in the index because it maps onto real
    # evidence. A segment whose mapping is empty, or reaches a unit that is not
    # citable, not on its page, or not in this version, is context with no
    # provenance -- which is the one thing it was never allowed to be.
    unmapped = conn.execute(
        """
        SELECT count(*) AS n FROM evidence_unit m
        WHERE m.version_id = %s AND m.kind = %s
          AND (jsonb_array_length(
                   COALESCE(m.table_context -> 'sources', '[]'::jsonb)) = 0
               OR EXISTS (
                   SELECT 1
                   FROM jsonb_array_elements_text(m.table_context -> 'sources') AS k(key)
                   WHERE NOT EXISTS (
                       SELECT 1 FROM evidence_unit s
                       WHERE s.version_id = m.version_id AND s.page_id = m.page_id
                         AND s.unit_key = k.key AND s.citable)))
        """,
        (version_id, str(EvidenceKind.PAGE_MARKDOWN)),
    ).fetchone()["n"]
    if unmapped:
        raise IngestionError(
            f"{unmapped} page Markdown units do not resolve to citable evidence"
        )
    passed(7, "Markdown mapping verified")

    unembedded = conn.execute(
        """
        SELECT count(*) AS n FROM evidence_unit
        WHERE version_id = %s AND kind = ANY(%s) AND embedding IS NULL
        """,
        (version_id, list(EMBEDDED_KINDS)),
    ).fetchone()["n"]
    if unembedded:
        raise IngestionError(f"{unembedded} embeddable units have no embedding")
    passed(8, "Embedding coverage verified")

    wrong = conn.execute(
        """
        SELECT count(*) AS n FROM evidence_unit
        WHERE version_id = %s AND embedding IS NOT NULL
          AND vector_dims(embedding) <> %s
        """,
        (version_id, settings.embed_dimensions),
    ).fetchone()["n"]
    if wrong:
        raise IngestionError(
            f"{wrong} stored embeddings are not {settings.embed_dimensions}-dimensional"
        )
    passed(9, "Embedding dimension verified")

    crossed = conn.execute(
        """
        SELECT count(*) AS n FROM fact f
        JOIN fact_evidence fe ON fe.fact_id = f.id
        JOIN evidence_unit e ON e.id = fe.evidence_id
        WHERE f.version_id = %s AND e.version_id <> f.version_id
        """,
        (version_id,),
    ).fetchone()["n"]
    if crossed:
        raise IngestionError(f"{crossed} fact links point outside this version")

    reporter.done(
        "building_indexes", "Index checks passed", completed=total, total=total
    )
    return int(counts["embedded"])


def retry_failed_facts(
    conn: psycopg.Connection,
    client: OllamaClient,
    settings: Settings,
    version_id: int,
    *,
    progress: ProgressCallback | None = None,
) -> IngestReport:
    """Re-run fact extraction for the candidates that failed, and only those.

    Evidence, provenance, and embeddings are already correct and expensive; a
    retry that rebuilt them would spend minutes re-deriving identical rows to
    fix a handful of missing facts. The version is read out of the database
    rather than re-derived from the source, so nothing here depends on the
    extraction output still being byte-identical.
    """
    reporter = ProgressReporter(progress, "ingest")
    version = conn.execute(
        "SELECT v.*, d.name FROM document_version v"
        " JOIN document d ON d.id = v.document_id WHERE v.id = %s",
        (version_id,),
    ).fetchone()
    if version is None:
        raise NotFoundError(f"no document version with id {version_id}")
    reporter.document_id = int(version["document_id"])
    logger.info(
        "fact retry started",
        extra={
            "event": "fact_retry_started", "operation": "ingest",
            "document_id": reporter.document_id, "version_id": version_id,
        },
    )

    pending = failed_fact_candidates(conn, version_id)
    reporter.start("extracting_facts", "Retrying failed passages", total=len(pending))
    subject = organization_name(version["name"], None)
    subject_entity = upsert_entity(
        conn, "organization", subject, normalized_name(subject)
    )

    stored = succeeded = 0
    for index, row in enumerate(pending, start=1):
        unit = conn.execute(
            "SELECT id, unit_key, source_text, heading_path FROM evidence_unit"
            " WHERE version_id = %s AND unit_key = %s",
            (version_id, row["unit_key"]),
        ).fetchone()
        if unit is None:
            # The unit no longer exists, so neither does the candidate. Nothing
            # to retry and nothing to keep reporting as outstanding.
            clear_fact_failures(conn, version_id, [row["unit_key"]])
            conn.commit()
            continue
        candidate = EvidenceUnit(
            unit_key=unit["unit_key"],
            page=0,
            kind=EvidenceKind.NARRATIVE,
            text=unit["source_text"],
            normalized_text=normalize_for_match(unit["source_text"]),
            heading_path=list(unit["heading_path"] or []),
        )
        try:
            extraction = client.structured(
                FactExtraction,
                FACT_EXTRACTION_SYSTEM,
                fact_extraction_prompt(candidate, subject),
            )
        except OllamaError as exc:
            record_fact_failure(
                conn, version_id, row["unit_key"], "model_unavailable"
            )
            conn.commit()
            reporter.warn(
                "extracting_facts", "A passage could not be processed by the model"
            )
            logger.warning(
                "fact retry deferred after model failure",
                extra={
                    "event": "fact_extraction_fallback", "operation": "ingest",
                    "document_id": reporter.document_id, "version_id": version_id,
                    "evidence_id": int(unit["id"]), "unit_key": row["unit_key"],
                    "error_type": type(exc).__name__,
                },
            )
            continue
        kept, _dropped = accept_llm_facts(extraction.facts, candidate, subject)
        for fact in kept:
            stored += _store_fact(
                conn, version_id, fact, subject_entity,
                {candidate.unit_key: int(unit["id"])},
            )
        succeeded += 1
        clear_fact_failures(conn, version_id, [row["unit_key"]])
        conn.commit()
        reporter.step(
            "extracting_facts",
            f"Retried {index} of {len(pending)} passages",
            completed=index,
            total=len(pending),
            current_item=row["unit_key"],
        )

    remaining = [r["unit_key"] for r in failed_fact_candidates(conn, version_id)]
    conn.execute(
        "UPDATE document_version"
        "   SET fact_candidates_succeeded = fact_candidates_succeeded + %s"
        " WHERE id = %s",
        (succeeded, version_id),
    )
    # Promotion is the point of the retry: a version whose last failure is gone
    # is complete, and must stop reporting itself as degraded.
    status = VersionStatus.DEGRADED if remaining else VersionStatus.READY
    if str(version["status"]) in QUERYABLE_STATUSES:
        activate_version(conn, version_id, str(status))
    conn.commit()

    coverage = conn.execute(
        "SELECT fact_candidates_total AS total, fact_candidates_succeeded AS ok"
        " FROM document_version WHERE id = %s",
        (version_id,),
    ).fetchone()
    counts = _stored_counts(conn, version_id)
    report = IngestReport(
        document_id=int(version["document_id"]),
        version_id=version_id,
        status=status,
        fingerprint=version["fingerprint"],
        pages=conn.execute(
            "SELECT count(*) AS n FROM page WHERE version_id = %s", (version_id,)
        ).fetchone()["n"],
        fact_candidates_total=int(coverage["total"]),
        fact_candidates_succeeded=int(coverage["ok"]),
        failed_fact_candidates=remaining,
        **counts,
    )
    reporter.done(
        "completed",
        f"Retried {len(pending)} passages; {len(remaining)} still failing",
        completed=1,
        total=1,
        details=_completion_details(report, reporter),
    )
    logger.info(
        "fact retry completed",
        extra={
            "event": "fact_retry_completed", "operation": "ingest",
            "document_id": reporter.document_id, "version_id": version_id,
            "attempted_candidates": len(pending),
            "successful_candidates": succeeded, "stored_facts": stored,
            "remaining_candidates": len(remaining), "status": str(status),
            "duration_seconds": reporter.elapsed_seconds,
        },
    )
    return report


def _verify_artifacts(conn: psycopg.Connection, version_id: int, root: Path) -> None:
    """Every citable unit must point at a real file inside the extraction root.

    Run before activation, not at render time. A citation whose artifact is
    missing or escapes the root is not a degraded citation -- it is one the
    product cannot show the user, and shipping it means the verdict claims
    provenance it cannot produce on request.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT e.artifact_path, p.page_dir
        FROM evidence_unit e JOIN page p ON p.id = e.page_id
        WHERE e.version_id = %s AND e.citable
        """,
        (version_id,),
    ).fetchall()

    missing: list[str] = []
    escaping: list[str] = []
    for row in rows:
        relative = str(row["artifact_path"] or "")
        if not relative:
            missing.append("(unset)")
            continue
        candidate = Path(relative)
        if candidate.is_absolute() or candidate.anchor:
            escaping.append(relative)
            continue
        resolved = (root / candidate).resolve()
        if root != resolved and root not in resolved.parents:
            escaping.append(relative)
        elif not resolved.is_file():
            missing.append(relative)
    if escaping:
        # The offending value is not echoed: it is attacker-controlled input in
        # the case that matters, and naming it hands back a filesystem probe.
        raise IngestionError(
            f"{len(escaping)} evidence artifact path(s) escape the extraction root"
        )
    if missing:
        raise IngestionError(
            f"{len(missing)} evidence artifact(s) do not exist: {sorted(set(missing))[:5]}"
        )

    # Page images are what a visual citation is cropped from, so a page whose
    # image is gone cannot support one.
    for page_dir in {str(row["page_dir"]) for row in rows}:
        image = (root / page_dir / PAGE_ARTIFACT_ROLES["page_image"]).resolve()
        if root not in image.parents or not image.is_file():
            raise IngestionError(f"page directory {page_dir} has no usable page image")


def ensure_schema(conn: psycopg.Connection, settings: Settings) -> str:
    """Apply the schema, then refuse to proceed on a dimension mismatch.

    ``vector(N)`` is templated in only when the table is first created, so
    re-initializing an existing database with a different configured dimension
    silently leaves the old column in place. Caught here, before the first
    embedding write, rather than at query time on a half-built index.
    """
    outcome = init_schema(conn, settings.embed_dimensions)
    declared = vector_dimension(conn)
    if declared is not None and declared != settings.embed_dimensions:
        # Not altered automatically: rewriting a populated vector column
        # discards every embedding in the database.
        raise IndexNotReadyError(
            f"database vector dimension {declared} does not match configured "
            f"dimension {settings.embed_dimensions}; use a fresh database or "
            f"reset the development one with `claim-evidence db reset-dev`"
        )
    return outcome


__all__ = [
    "EMBEDDED_KINDS",
    "VERIFY_STEPS",
    "IngestionError",
    "ensure_schema",
    "FINGERPRINT_VERSION",
    "IDENTITY_VERSION",
    "canonical_local_path",
    "identity_basis",
    "identity_key",
    "index_fingerprint",
    "normalize_source_uri",
    "ingest_document",
    "retry_failed_facts",
]
