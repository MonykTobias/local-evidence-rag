"""Hybrid retrieval: facts, full-text, and vectors merged by rank fusion.

The three retrievers run independently and are merged with reciprocal-rank
fusion, so a candidate that only one of them finds still surfaces. Exact
numbers, years, units, and scope tokens get an explicit bonus on top, because
a claim audit lives or dies on "40.2" versus "40.3" and embedding similarity
is indifferent to that difference.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Sequence

import psycopg

from .db import (
    graph_search,
    lexical_search,
    neighbours,
    regions_for,
    vector_search,
)
from .models import (
    Citation,
    EvidenceKind,
    EvidenceMatch,
    EvidenceQuality,
    GeometryPrecision,
    ParsedClaim,
    Region,
)
from .normalize import all_years, content_tokens, normalize_for_match, scope_markers
from .progress import ProgressReporter

logger = logging.getLogger(__name__)

RRF_K = 60
# Scaled to the fusion signal, not to 1.0: a candidate ranked first by all
# three retrievers scores 3/(RRF_K+1) ~= 0.049, so a full exact-token match is
# worth about as much as that, and a scope match about half. Bonuses on a 0-1
# scale would swamp the ranks entirely and leave ties broken by row id.
EXACT_TOKEN_BONUS = 1.0 / RRF_K
SCOPE_TOKEN_BONUS = 0.5 / RRF_K
_NUMBER_TOKEN = re.compile(r"\d[\d.,]*")


def lexical_query(text: str, key_terms: Sequence[str] = ()) -> str:
    """Build a recall-oriented websearch query.

    ``websearch_to_tsquery`` ANDs bare terms, which makes a whole sentence
    match nothing. Terms are OR'd instead and precision comes from ranking.
    """
    terms = list(dict.fromkeys([*key_terms, *sorted(content_tokens(text))]))
    # A multi-word term must be a quoted phrase; bare words next to OR are
    # parsed as an AND group and quietly drop the whole clause's recall.
    quoted = [f'"{t}"' if " " in t else t for t in terms if t]
    numbers = [f'"{value}"' for value in dict.fromkeys(_exact_numbers(text))]
    parts = [*numbers, *quoted]
    return " OR ".join(parts) if parts else text


def _exact_numbers(text: str) -> list[str]:
    """Numeric tokens with sentence punctuation stripped ("2020." -> "2020")."""
    return [match.strip(".,") for match in _NUMBER_TOKEN.findall(text) if match.strip(".,")]


def exact_tokens(claim: ParsedClaim, claim_text: str) -> set[str]:
    """Tokens whose literal presence is strong evidence of relevance."""
    tokens = set(_exact_numbers(claim_text))
    tokens |= set(all_years(claim_text))
    for period in (claim.reporting_period, claim.baseline_period):
        if period:
            tokens.add(period)
    if claim.value_decimal is not None:
        tokens.add(str(claim.value_decimal))
        tokens.add(str(abs(claim.value_decimal)))
    if claim.unit:
        tokens.add(claim.unit)
    return {t for t in tokens if t}


def fuse(
    ranked: dict[str, list[dict[str, Any]]],
    *,
    claim: ParsedClaim | None = None,
    claim_text: str = "",
) -> list[dict[str, Any]]:
    """Reciprocal-rank fusion plus an exact-token bonus."""
    tokens = exact_tokens(claim, claim_text) if claim else set()
    markers = scope_markers(claim_text) if claim_text else frozenset()

    merged: dict[int, dict[str, Any]] = {}
    for source, rows in ranked.items():
        for position, row in enumerate(rows, start=1):
            evidence_id = int(row["id"])
            entry = merged.setdefault(
                evidence_id,
                {"row": row, "score": 0.0, "lexical_rank": None,
                 "vector_rank": None, "graph_rank": None, "lexical_score": None,
                 "vector_score": None, "graph_score": None},
            )
            entry["score"] += 1.0 / (RRF_K + position)
            entry[f"{source}_rank"] = position
            # The retriever's own score, kept for the audit trace.
            if (channel_score := row.get(f"{source}_score")) is not None:
                entry[f"{source}_score"] = float(channel_score)

    for entry in merged.values():
        text = normalize_for_match(entry["row"]["source_text"])
        if tokens:
            hits = sum(1 for token in tokens if normalize_for_match(token) in text)
            entry["score"] += EXACT_TOKEN_BONUS * hits / len(tokens)
        if markers and scope_markers(entry["row"]["source_text"]) & markers:
            entry["score"] += SCOPE_TOKEN_BONUS

    # Ties break on where the evidence sits in the document, not on when its
    # row happened to be inserted. Evidence ids record ingestion and resume
    # history, so an id tiebreak makes the order of two equally-scored
    # candidates depend on which attempt built them.
    ordered = sorted(merged.values(), key=_document_position)
    for position, entry in enumerate(ordered, start=1):
        entry["combined_rank"] = position
    return ordered


def _document_position(entry: dict[str, Any]) -> tuple[float, int, int, int]:
    """Sort key: best score first, then reading order, then id as a last resort."""
    row = entry["row"]
    order = row.get("source_order")
    return (
        -entry["score"],
        int(row.get("pdf_page") or 0),
        int(order) if order is not None else 1 << 30,
        int(row["id"]),
    )


def _admitted(
    rows: Sequence[dict[str, Any]], allowed_kinds: Sequence[EvidenceKind] | None
) -> list[dict[str, Any]]:
    """Rows this retrieval pass is allowed to rank.

    Filtered here rather than in the queries: `db.py` returns what the index
    holds, and every caller of a search helper -- recall measurement included --
    would otherwise have to agree with the audit about what is admissible.
    """
    if allowed_kinds is None:
        return [row for row in rows if row.get("citable", True)]
    wanted = {str(kind) for kind in allowed_kinds}
    return [row for row in rows if str(row["kind"]) in wanted]


def retrieve(
    conn: psycopg.Connection,
    query_embedding: Sequence[float] | None,
    claim: ParsedClaim,
    claim_text: str,
    *,
    document_ids: Sequence[int] | None = None,
    limit: int = 20,
    pool: int = 60,
    reporter: ProgressReporter | None = None,
    allowed_kinds: Sequence[EvidenceKind] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Candidates for one claim, best first, plus the per-channel results.

    The raw channel lists come back with the fused list so a caller can report
    what each retriever contributed without querying anything again.

    ``allowed_kinds`` is what separates the two passes an audit makes. Left
    unset, only citable rows are ranked at all: generated Markdown repeats the
    text of the units it was made from, so it competes with them on every token
    they match while being unable to carry a citation, and one that outranked
    them would spend the caller's limit on a row no verdict can rest on. Set to
    ``(PAGE_MARKDOWN,)``, only that generated text comes back -- the second
    pass, run once direct evidence has already come back with nothing.
    """
    report = reporter or ProgressReporter(None, "audit")
    ranked: dict[str, list[dict[str, Any]]] = {}
    retrieval_pass = "mapped_context" if allowed_kinds else "direct"
    lexical = lexical_query(claim_text, claim.key_terms)
    metric_terms = sorted(content_tokens(claim.metric or claim_text))[:8]
    common = {
        "operation": report.operation, "audit_id": report.audit_id,
        "document_id": report.document_id, "retrieval_pass": retrieval_pass,
    }
    logger.info(
        "retrieval started",
        extra={
            "event": "retrieval_started", **common,
            "document_ids": list(document_ids or ()), "limit": limit, "pool": pool,
            "vector_enabled": query_embedding is not None,
        },
    )
    logger.debug(
        "retrieval queries prepared",
        extra={
            "event": "retrieval_queries", **common,
            "graph_term_count": len(metric_terms),
            "lexical_query_chars": len(lexical),
            "lexical_query_sha256": hashlib.sha256(
                lexical.encode("utf-8")
            ).hexdigest(),
            "exact_query_tokens": sorted(exact_tokens(claim, claim_text)),
            "reporting_period": claim.reporting_period,
            "baseline_period": claim.baseline_period,
        },
    )

    report.start("retrieving_graph", "Searching claim facts", total=pool)
    ranked["graph"] = _admitted(
        graph_search(
            conn,
            metric_terms=metric_terms,
            reporting_period=claim.reporting_period,
            baseline_period=claim.baseline_period,
            document_ids=document_ids,
            limit=pool,
        ),
        allowed_kinds,
    )
    report.done(
        "retrieving_graph",
        f"{len(ranked['graph'])} fact candidates",
        completed=len(ranked["graph"]),
        total=pool,
    )

    report.start("retrieving_full_text", "Searching full text", total=pool)
    ranked["lexical"] = _admitted(
        lexical_search(
            conn, lexical, document_ids, pool
        ),
        allowed_kinds,
    )
    report.done(
        "retrieving_full_text",
        f"{len(ranked['lexical'])} full-text candidates",
        completed=len(ranked["lexical"]),
        total=pool,
    )

    if query_embedding is not None:
        report.start("retrieving_vectors", "Searching embeddings", total=pool)
        ranked["vector"] = _admitted(
            vector_search(conn, query_embedding, document_ids, pool), allowed_kinds
        )
        report.done(
            "retrieving_vectors",
            f"{len(ranked['vector'])} vector candidates",
            completed=len(ranked["vector"]),
            total=pool,
        )

    logger.debug(
        "retrieval stages produced candidates",
        extra={
            "event": "retrieval_stage_candidates", **common,
            "candidates_by_stage": {
                stage: [
                    {
                        "evidence_id": int(row["id"]), "rank": rank,
                        "score": row.get(f"{stage}_score"),
                    }
                    for rank, row in enumerate(rows, start=1)
                ]
                for stage, rows in ranked.items()
            },
        },
    )

    report.start("fusing_candidates", "Merging candidate ranks")
    fused = fuse(ranked, claim=claim, claim_text=claim_text)
    # completed == total: every merged candidate was processed. The limit that
    # follows is a cut, not incomplete work, so it belongs in the message.
    report.done(
        "fusing_candidates",
        f"Merged {len(fused)} candidates, keeping {min(len(fused), limit)}",
        completed=len(fused),
        total=len(fused),
    )
    kept = fused[:limit]
    for entry in kept:
        logger.debug(
            "candidate retained after fusion",
            extra={
                "event": "candidate_fused", **common,
                "evidence_id": int(entry["row"]["id"]),
                "combined_rank": entry.get("combined_rank"),
                "combined_score": entry.get("score"),
                "graph_rank": entry.get("graph_rank"),
                "graph_score": entry.get("graph_score"),
                "lexical_rank": entry.get("lexical_rank"),
                "lexical_score": entry.get("lexical_score"),
                "vector_rank": entry.get("vector_rank"),
                "vector_score": entry.get("vector_score"),
            },
        )
    logger.info(
        "retrieval completed",
        extra={
            "event": "retrieval_completed", **common,
            "graph_candidates": len(ranked.get("graph", ())),
            "lexical_candidates": len(ranked.get("lexical", ())),
            "vector_candidates": len(ranked.get("vector", ())),
            "fused_candidates": len(fused), "retained_candidates": len(kept),
        },
    )
    logger.debug(
        "retrieval candidates left after limit",
        extra={
            "event": "retrieval_candidates_dropped", **common,
            "evidence_ids": [int(entry["row"]["id"]) for entry in fused[limit:]],
        },
    )
    return kept, ranked


def expand(
    conn: psycopg.Connection,
    candidates: Sequence[dict[str, Any]],
    top: int = 5,
    reporter: ProgressReporter | None = None,
) -> list[dict[str, Any]]:
    """Pull in each top candidate's page neighbours: table rows, headers, prose.

    A value cell alone rarely proves a claim; the row it sits in and the
    paragraph beside it are what make the qualifiers checkable.
    """
    report = reporter or ProgressReporter(None, "audit")
    considered = min(top, len(candidates))
    logger.debug(
        "context expansion started",
        extra={
            "event": "context_expansion_started", "operation": report.operation,
            "audit_id": report.audit_id, "document_id": report.document_id,
            "input_candidates": len(candidates), "seeds": considered,
        },
    )
    report.start("expanding_context", "Expanding context", total=considered)
    seen = {int(c["row"]["id"]) for c in candidates}
    extra: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates[:top], start=1):
        for row in neighbours(conn, int(candidate["row"]["id"])):
            evidence_id = int(row["id"])
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            extra.append(
                {
                    "row": row,
                    "score": candidate["score"] * 0.25,
                    "lexical_rank": None,
                    "vector_rank": None,
                    "graph_rank": None,
                    "combined_rank": None,
                    "expanded_from": int(candidate["row"]["id"]),
                }
            )
            logger.debug(
                "candidate entered through context expansion",
                extra={
                    "event": "candidate_expanded", "operation": report.operation,
                    "audit_id": report.audit_id, "document_id": report.document_id,
                    "evidence_id": evidence_id,
                    "expanded_from": int(candidate["row"]["id"]),
                    "combined_score": candidate["score"] * 0.25,
                },
            )
        report.step(
            "expanding_context",
            f"Expanded {position} of {considered}",
            completed=position,
            total=considered,
        )
    report.done(
        "expanding_context",
        f"Added {len(extra)} neighbouring units",
        completed=considered,
        total=considered,
    )
    expanded = [*candidates, *extra]
    logger.info(
        "context expansion completed",
        extra={
            "event": "context_expansion_completed", "operation": report.operation,
            "audit_id": report.audit_id, "document_id": report.document_id,
            "input_candidates": len(candidates), "added_candidates": len(extra),
            "output_candidates": len(expanded),
        },
    )
    return expanded


def to_citation(
    row: dict[str, Any],
    regions: Sequence[dict[str, Any]],
    *,
    quality: EvidenceQuality | None = None,
) -> Citation:
    kind = EvidenceKind(row["kind"])
    context = row.get("table_context") or {}
    cells = [str(v) for v in (context.get("cells") or []) if v]
    if kind is EvidenceKind.TABLE_VALUE:
        cells = [
            str(context.get("descriptor") or ""),
            " ".join(context.get("header_path") or []),
            str(context.get("unit") or ""),
            str(context.get("value") or ""),
        ]
    return Citation(
        evidence_id=int(row["id"]),
        document_id=int(row["document_id"]),
        document_name=row["document_name"],
        document_sha256=row.get("sha256"),
        source_uri=row.get("source_uri"),
        pdf_page=int(row["pdf_page"]),
        printed_page_label=row.get("printed_page_label"),
        source_kind=kind,
        quality=quality or EvidenceQuality(row["quality"]),
        quote=row["source_text"] if kind is not EvidenceKind.TABLE_VALUE else None,
        table_cells=[c for c in cells if c],
        heading_path=list(row.get("heading_path") or []),
        artifact_path=f"{row['page_dir']}/{row['artifact_path'].split('/')[-1]}",
        regions=[
            Region(
                bbox=(r["left_norm"], r["top_norm"], r["right_norm"], r["bottom_norm"]),
                role=r["role"],
                precision=GeometryPrecision(r["precision"]),
                source_bbox=tuple(r["source_bbox"]) if r.get("source_bbox") else None,
                source_origin=r.get("source_origin"),
            )
            for r in regions
        ],
        geometry_precision=GeometryPrecision(row["geometry_precision"]),
    )


def to_matches(
    conn: psycopg.Connection, candidates: Sequence[dict[str, Any]]
) -> list[EvidenceMatch]:
    ids = [int(c["row"]["id"]) for c in candidates]
    grouped = regions_for(conn, ids)
    return [
        EvidenceMatch(
            citation=to_citation(c["row"], grouped.get(int(c["row"]["id"]), [])),
            text=c["row"]["source_text"],
            lexical_rank=c.get("lexical_rank"),
            vector_rank=c.get("vector_rank"),
            graph_rank=c.get("graph_rank"),
            combined_score=round(c["score"], 6),
        )
        for c in candidates
    ]


__all__ = [
    "EXACT_TOKEN_BONUS",
    "RRF_K",
    "exact_tokens",
    "expand",
    "fuse",
    "lexical_query",
    "retrieve",
    "to_citation",
    "to_matches",
]
