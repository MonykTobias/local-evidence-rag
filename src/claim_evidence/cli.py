"""argparse CLI: db init, ingest, search, audit."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from typing import Sequence

from .audit import AuditError
from .client import ClaimEvidence
from .errors import ClaimEvidenceError, ValidationError
from .ingest import IngestionError
from .models import ClaimResult, EvidenceMatch, HealthReport, IngestReport
from .model_client import ModelError
from .progress import classify_error
from .reset import CONFIRM_PHRASE, reset_dev
from .source import OutputValidationError

USER_ERRORS = (
    ClaimEvidenceError,
    OutputValidationError,
    IngestionError,
    AuditError,
    ModelError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claim-evidence",
        description="Audit claims against source-bound evidence from a document extraction.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    db = sub.add_parser("db", help="database maintenance").add_subparsers(
        dest="db_command", required=True
    )
    db.add_parser("init", help="install the current schema, or confirm it is current")
    db.add_parser("documents", help="list ready documents")

    reset = db.add_parser(
        "reset-dev",
        help="drop and reinstall the schema of a disposable _dev/_test database",
        description=(
            "Destroys this database's index rows, citations, and audit history. "
            "Source PDFs, extraction output, archives, and configuration are "
            "never touched. Requires CE_ENVIRONMENT=development, a database name "
            "ending in _dev or _test, and both confirmations. Shows what would "
            "be lost and changes nothing unless --execute is given."
        ),
    )
    reset.add_argument(
        "--confirm-database",
        default="",
        help="the exact database name from the configured URL",
    )
    reset.add_argument(
        "--confirm-phrase",
        default="",
        help=f"exactly {CONFIRM_PHRASE}",
    )
    reset.add_argument(
        "--execute",
        action="store_true",
        help="actually reset; without it this is a dry run",
    )

    ingest = sub.add_parser("ingest", help="index a completed output root")
    ingest.add_argument("output_root")
    ingest.add_argument(
        "--entity",
        dest="reporting_entity",
        required=True,
        help="the reporting entity every stored fact is attributed to; a "
        "document filename is not one",
    )
    ingest.add_argument("--pdf", dest="source_pdf", help="source PDF for SHA-256 identity")
    ingest.add_argument("--source-uri", dest="source_uri")
    ingest.add_argument(
        "--force",
        action="store_true",
        help="rebuild even when the source is unchanged; the current version "
        "keeps serving queries until the replacement passes its checks",
    )
    ingest.add_argument(
        "--no-narrative-facts",
        action="store_true",
        help="index evidence only; skip the LLM narrative fact extractor",
    )

    search = sub.add_parser("search", help="hybrid evidence search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--document-id", type=int, action="append", dest="document_ids")
    search.add_argument("--json", action="store_true")

    sub.add_parser("health", help="report database, schema, and model status")

    sub.add_parser("documents", help="list indexed documents")

    remove = sub.add_parser("remove", help="delete one document's index rows")
    remove.add_argument("document_id")
    remove.add_argument(
        "--confirm", required=True, help="repeat DOCUMENT_ID to confirm"
    )

    trace = sub.add_parser("trace", help="show one audit's retrieval trace")
    trace.add_argument("audit_id")
    trace.add_argument("--json", action="store_true")

    evidence = sub.add_parser("evidence", help="show one evidence unit's regions")
    evidence.add_argument("evidence_id")
    evidence.add_argument("--json", action="store_true")

    audit = sub.add_parser("audit", help="audit one atomic claim")
    audit.add_argument("claim")
    audit.add_argument(
        "--entity",
        dest="reporting_entity",
        required=True,
        help="who the claim is about; version 1 audits one named entity",
    )
    audit.add_argument("--limit", type=int, default=20)
    audit.add_argument(
        "--document-id",
        type=int,
        action="append",
        dest="document_ids",
        help="restrict the audit to these documents; omit to search all of "
        "them, which must be said rather than defaulted to",
    )
    audit.add_argument(
        "--all-documents",
        action="store_true",
        help="search every queryable document",
    )
    audit.add_argument("--json", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with ClaimEvidence.from_env() as client:
        if args.command == "db":
            if args.db_command == "init":
                print(f"schema {client.init_db()}")
            elif args.db_command == "reset-dev":
                plan = reset_dev(
                    client.settings,
                    confirm_database=args.confirm_database,
                    confirm_phrase=args.confirm_phrase,
                    dry_run=not args.execute,
                    conn=client.conn,
                )
                print(plan.describe())
                print(
                    "reset complete; run `claim-evidence db init` is not needed, "
                    "the current schema was reinstalled"
                    if plan.performed
                    else "dry run: nothing was changed. Re-run with --execute to reset."
                )
            else:
                for row in client.documents():
                    print(f"{row['id']}\t{row['name']}\tversion {row['version_id']}")
            return 0

        if args.command == "health":
            print(_format_health(client.health()))
            return 0

        if args.command == "documents":
            for doc in client.list_documents():
                print(
                    f"{doc.document_id}\t{doc.name}\t{doc.status}\t"
                    f"{doc.page_count} pages\t{doc.evidence_count} evidence\t"
                    f"{doc.fact_count} facts\t{doc.embedding_model}"
                )
            return 0

        if args.command == "remove":
            report = client.remove_document(
                args.document_id, confirm_document_id=args.confirm
            )
            print(f"removed document {report.document_id} ({report.name})")
            for table, count in sorted(report.deleted.items()):
                print(f"  {table:<18} {count}")
            print("source files and output directories were not touched")
            return 0

        if args.command == "trace":
            trace = client.get_audit_trace(args.audit_id)
            print(
                json.dumps(trace.model_dump(mode="json"), indent=2)
                if args.json
                else _format_trace(trace)
            )
            return 0

        if args.command == "evidence":
            detail = client.get_evidence(args.evidence_id)
            print(
                json.dumps(detail.model_dump(mode="json"), indent=2)
                if args.json
                else _format_evidence(detail)
            )
            return 0

        if args.command == "ingest":
            report = client.ingest_document(
                args.output_root,
                reporting_entity=args.reporting_entity,
                source_pdf=args.source_pdf,
                source_uri=args.source_uri,
                force=args.force,
                extract_narrative_facts=not args.no_narrative_facts,
            )
            print(_format_ingest(report))
            return 0

        if args.command == "search":
            matches = client.search_evidence(
                args.query, document_ids=args.document_ids, limit=args.limit
            )
            print(
                json.dumps([m.model_dump(mode="json") for m in matches], indent=2)
                if args.json
                else _format_matches(matches)
            )
            return 0

        # Scope is stated, never defaulted: one of the two flags is required,
        # so an audit can never quietly widen to the whole corpus.
        if args.all_documents == bool(args.document_ids):
            raise ValidationError(
                "pass either --all-documents or one or more --document-id"
            )
        result = client.audit_claim(
            args.claim,
            scope="all" if args.all_documents else args.document_ids,
            reporting_entity=args.reporting_entity,
            limit=args.limit,
        )
        print(
            json.dumps(result.model_dump(mode="json"), indent=2)
            if args.json
            else _format_result(result)
        )
        return 0


def _format_ingest(report: IngestReport) -> str:
    lines = [
        f"document {report.document_id} version {report.version_id} [{report.status}]",
        f"  fingerprint    {report.fingerprint[:16]}...",
        f"  pages          {report.pages}",
        f"  evidence units {report.evidence_units} ({report.embedded_units} embedded)",
        f"  facts          {report.facts}",
    ]
    if report.reused_existing:
        lines.append("  unchanged: existing ready version reused")
    for warning in report.warnings[:10]:
        lines.append(f"  warning: {warning}")
    if report.rejected_facts:
        lines.append(f"  rejected facts: {len(report.rejected_facts)}")
    return "\n".join(lines)


def _format_matches(matches: Sequence[EvidenceMatch]) -> str:
    if not matches:
        return "no matches"
    lines = []
    for match in matches:
        cite = match.citation
        lines.append(
            f"[{cite.evidence_id}] p.{cite.pdf_page} {cite.source_kind} "
            f"({cite.quality}, {cite.geometry_precision}) score={match.combined_score:.4f}"
        )
        lines.append(f"      {match.text[:180]}")
    return "\n".join(lines)


def _format_result(result: ClaimResult) -> str:
    lines = [
        f"verdict:  {result.verdict}",
        f"quality:  {result.evidence_quality}",
        f"rationale: {result.rationale}",
    ]
    if result.missing_qualifiers:
        lines.append(f"missing:  {', '.join(result.missing_qualifiers)}")
    for cite in result.citations:
        regions = ", ".join(
            f"{r.role}[{r.bbox[0]:.3f},{r.bbox[1]:.3f},{r.bbox[2]:.3f},{r.bbox[3]:.3f}]"
            for r in cite.regions
        )
        lines.append(
            f"  - {cite.document_name} p.{cite.pdf_page} ({cite.quality}, "
            f"{cite.geometry_precision})"
        )
        if cite.table_cells:
            lines.append(f"      cells: {' | '.join(cite.table_cells)}")
        elif cite.quote:
            lines.append(f"      quote: {cite.quote[:180]}")
        lines.append(f"      regions: {regions}")
    return "\n".join(lines)


def _format_health(report: HealthReport) -> str:
    lines = [
        f"database   {'reachable' if report.database_reachable else 'UNREACHABLE'}",
        f"schema     v{report.schema_version} "
        f"({'current' if report.schema_current else 'OUT OF DATE'})",
        f"pgvector   {report.pgvector_version or 'not installed'}",
        f"{report.model_backend:<11}"
        f"{'reachable' if report.model_server_reachable else 'UNREACHABLE'}",
    ]
    for model in report.models:
        state = (
            "available"
            if model.available
            else "NOT PULLED" if report.model_backend == "ollama" else "UNAVAILABLE"
        )
        lines.append(f"  {model.role:<7}{model.name} [{state}]")
    lines += [
        f"documents  {report.documents_ready} ready, "
        f"{report.documents_building} building, {report.documents_inactive} inactive",
        f"index      {report.evidence_units} evidence, "
        f"{report.embeddings} embeddings, {report.facts} facts",
    ]
    for problem in report.problems:
        lines.append(f"problem:   {problem}")
    return "\n".join(lines)


def _format_trace(trace) -> str:
    lines = [
        f"audit {trace.audit_id}: {trace.claim}",
        f"  verdict   {trace.verdict} ({trace.evidence_quality})",
        f"  rationale {trace.rationale}",
        f"  channels  {len(trace.lexical_candidates)} lexical, "
        f"{len(trace.vector_candidates)} vector, {len(trace.graph_candidates)} graph",
        f"  citations {trace.citation_ids}",
    ]
    for candidate in trace.candidates:
        mark = "*" if candidate.selected else " "
        lines.append(
            f" {mark}[{candidate.evidence_id}] p.{candidate.pdf_page} "
            f"{candidate.source_kind} rrf#{candidate.combined_rank} "
            f"score={candidate.combined_score:.4f} "
            f"L{candidate.lexical_rank} V{candidate.vector_rank} G{candidate.graph_rank}"
        )
        lines.append(f"      {candidate.reason}")
    return "\n".join(lines)


def _format_evidence(detail) -> str:
    lines = [
        f"evidence {detail.evidence_id} in {detail.document_name} p.{detail.pdf_page}",
        f"  kind      {detail.source_kind} ({detail.evidence_quality}, "
        f"{detail.geometry_precision})",
        f"  page      {detail.page_width} x {detail.page_height}",
        f"  image     {detail.page_image_path or 'none'}",
        f"  text      {detail.text[:200]}",
    ]
    for region in detail.regions:
        left, top, right, bottom = region.bbox
        lines.append(
            f"  region    {region.role:<20}{region.precision:<8}"
            f"[{left:.4f}, {top:.4f}, {right:.4f}, {bottom:.4f}] "
            f"({region.coordinate_space})"
        )
    for path in detail.artifact_paths:
        lines.append(f"  artifact  {path}")
    return "\n".join(lines)


DEBUG_VARIABLE = "CLAIM_EVIDENCE_DEBUG"


def cli() -> None:
    """Run one command, reporting failures the way a public surface must.

    Every failure is classified before it is printed. Only the package's own
    typed errors carry a message written for a person; a model error can embed
    the model's reply, a driver error embeds the host and sometimes the
    password, and an unexpected bug embeds whatever it happened to be holding.
    Those are reported by category.

    The raw cause is available, but it has to be asked for: set
    ``CLAIM_EVIDENCE_DEBUG=1`` and it goes to stderr as a traceback. That is a
    local debugging aid, not the default, because the default output is what
    ends up pasted into a chat window or a bug report.
    """
    debug = os.environ.get(DEBUG_VARIABLE, "").strip().lower() in ("1", "true", "yes")
    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stderr,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - one classification point
        code, retryable, message = classify_error(exc)
        print(f"error [{code}]: {message}", file=sys.stderr)
        if retryable:
            print("this may succeed on a retry", file=sys.stderr)
        if debug:
            traceback.print_exc()
        else:
            print(
                f"set {DEBUG_VARIABLE}=1 for the underlying cause",
                file=sys.stderr,
            )
        raise SystemExit(1) from None


if __name__ == "__main__":
    cli()
