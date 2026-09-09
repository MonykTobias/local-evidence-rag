"""Public result types and the internal evidence/fact records they wrap.

Everything crossing the package boundary is a Pydantic model so an external
agent gets validated data and `model_dump()` JSON for free. Model responses are
parsed through the same models, which is what makes a hallucinated field a
validation error instead of a verdict.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import ValidationError

logger = logging.getLogger(__name__)


class PublicModel(BaseModel):
    """Base for every type that crosses the package boundary.

    ``extra="forbid"`` is the contract rule, not a style choice: a caller that
    sends a key this package does not know about has misunderstood the request,
    and quietly dropping it means answering a question nobody asked. The same
    strictness catches a producer and a consumer drifting apart while both
    still report success.

    LLM response models deliberately do *not* inherit this: a chatty model
    adding a field is handled by the quote gate and the evidence-id allowlist,
    not by refusing to parse the reply at all.
    """

    model_config = ConfigDict(extra="forbid")


def _coerce_displayed_value(value: Any) -> Any:
    """Read a displayed value string into a Decimal.

    Models answer with the figure as the page prints it -- "-40.2%",
    "(40.2) %", "1,044" -- none of which is valid JSON number syntax. Parsing
    it exactly the way a table cell is parsed is faithful to the source; it is
    the quote check, not the number format, that guards against invention.
    """
    if isinstance(value, str):
        from .normalize import parse_value  # local: `normalize` imports this module

        parsed, _ = parse_value(value)
        logger.debug(
            "displayed model value coerced",
            extra={
                "event": "displayed_value_coerced", "raw_chars": len(value),
                "normalized_value": str(parsed) if parsed is not None else None,
            },
        )
        return parsed
    return value


class EvidenceKind(StrEnum):
    NARRATIVE = "narrative"
    TABLE_ROW = "table_row"
    TABLE_VALUE = "table_value"
    VISUAL = "visual"
    PAGE_MARKDOWN = "page_markdown"


class EvidenceQuality(StrEnum):
    """What a citation actually proves. Deliberately not a confidence score."""

    DIRECT_TEXT = "direct_text"
    DIRECT_TABLE = "direct_table"
    VERIFIED_VISUAL = "verified_visual"
    COARSE_REGION = "coarse_region"
    NONE = "none"


class RegionRole(StrEnum):
    """What a region encloses. One closed vocabulary, so a renderer can style
    a value cell differently from the descriptor without guessing."""

    CLAIM_TEXT = "claim_text"
    DESCRIPTOR = "descriptor"
    HEADER = "header"
    UNIT = "unit"
    VALUE = "value"
    SUPPORTING_CONTEXT = "supporting_context"
    VISUAL_REGION = "visual_region"
    UNKNOWN = "unknown"


# Roles written by earlier builds, so an index created before the vocabulary
# was closed still reads back without a re-ingest.
_LEGACY_ROLES = {
    "block": RegionRole.CLAIM_TEXT,
    "cell": RegionRole.SUPPORTING_CONTEXT,
    "row": RegionRole.SUPPORTING_CONTEXT,
    "table": RegionRole.SUPPORTING_CONTEXT,
    "page": RegionRole.SUPPORTING_CONTEXT,
    "content": RegionRole.SUPPORTING_CONTEXT,
    "visual": RegionRole.VISUAL_REGION,
}


def _coerce_role(value: Any) -> Any:
    if isinstance(value, str) and value not in set(RegionRole):
        return _LEGACY_ROLES.get(value, RegionRole.UNKNOWN)
    return value


class GeometryPrecision(StrEnum):
    """How tightly a region encloses the cited content."""

    BLOCK = "block"
    CELL = "cell"
    ROW = "row"
    TABLE = "table"
    CROP = "crop"
    PAGE = "page"


class Verdict(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    MIXED = "mixed"
    INSUFFICIENT = "insufficient"


class VersionStatus(StrEnum):
    """Where one build got to.

    ``degraded`` is queryable: evidence, provenance, and embeddings are all
    complete and only optional narrative fact enrichment is missing, which
    makes for a smaller index rather than a wrong one. ``interrupted`` is what
    a build that was still running when its process died becomes -- it is not
    ``failed``, because nobody observed it fail.
    """

    BUILDING = "building"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"
    INACTIVE = "inactive"
    INTERRUPTED = "interrupted"


class Region(PublicModel):
    """One highlightable rectangle on a page.

    ``bbox`` is always ``[left, top, right, bottom]`` normalized to 0-1 with a
    top-left origin. The pre-normalization coordinates stay in ``source_bbox``
    and ``source_origin`` so a caller can re-derive the original geometry.
    """

    model_config = ConfigDict(frozen=True)

    bbox: tuple[float, float, float, float]
    role: RegionRole = RegionRole.SUPPORTING_CONTEXT
    precision: GeometryPrecision = GeometryPrecision.BLOCK
    source_bbox: tuple[float, float, float, float] | None = None
    source_origin: str | None = None
    coordinate_space: Literal["normalized_top_left"] = "normalized_top_left"

    _coerce_role = field_validator("role", mode="before")(_coerce_role)


class Citation(PublicModel):
    evidence_id: int
    document_id: int
    document_name: str
    document_sha256: str | None = None
    source_uri: str | None = None
    pdf_page: int
    printed_page_label: str | None = None
    source_kind: EvidenceKind
    quality: EvidenceQuality
    quote: str | None = None
    table_cells: list[str] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)
    artifact_path: str
    regions: list[Region] = Field(default_factory=list)
    geometry_precision: GeometryPrecision = GeometryPrecision.BLOCK


class EvidenceMatch(PublicModel):
    citation: Citation
    text: str
    lexical_rank: int | None = None
    vector_rank: int | None = None
    graph_rank: int | None = None
    combined_score: float = 0.0


QualifierName = Literal[
    "subject", "metric", "scope", "unit",
    "reporting_period", "baseline_period", "geography",
]


class QualifierComparison(PublicModel):
    """How one material qualifier of the claim lined up with one stored fact.

    ``match`` means the package's own comparison established comparability --
    never that a value was merely parsed. A qualifier the source omits is
    ``missing``, and ``mismatch`` requires both sides present and disagreeing.
    """

    qualifier: QualifierName
    claim_value: str | None = None
    source_value: str | None = None
    status: Literal["match", "mismatch", "missing"]
    reason: str | None = None


class NumericComparison(PublicModel):
    """The arithmetic, in the terms the page and the claim each stated it."""

    claim_value: str | None = None
    claim_operator: str | None = None
    claim_direction: str | None = None
    source_value: str | None = None
    source_operator: str | None = None
    source_unit: str | None = None
    outcome: Literal["match", "conflict", "incomparable", "not_applicable"]
    reason: str | None = None


class EvidenceComparison(PublicModel):
    """One claim-versus-fact comparison, bound to the evidence it came from."""

    evidence_id: int
    fact_id: int | None = None
    pdf_page: int | None = None
    qualifiers: list[QualifierComparison] = Field(default_factory=list)
    numeric: NumericComparison


class DecisionExplanation(PublicModel):
    """Why this verdict, in operational terms.

    ``verdict_rule`` is a stable machine-readable name for the rule that fired,
    not model reasoning: the user-facing prose stays in ``rationale``. Nothing
    here is a prompt, a raw model reply, or hidden chain-of-thought.
    """

    decided_by: Literal[
        "deterministic_comparison", "semantic_adjudication", "no_evidence"
    ]
    verdict_rule: str
    evidence_comparisons: list[EvidenceComparison] = Field(default_factory=list)


class IndexReference(PublicModel):
    """Exactly which ready version answered one audit.

    Pinned before retrieval, so a trace read back later says what was searched
    rather than what happens to be ready now.
    """

    document_id: int
    document_version_id: int
    embedding_model: str
    embedding_dimensions: int


class ClaimResult(PublicModel):
    claim: str
    verdict: Verdict
    rationale: str
    evidence_quality: EvidenceQuality
    citations: list[Citation] = Field(default_factory=list)
    missing_qualifiers: list[str] = Field(default_factory=list)
    audit_id: int | None = None
    decision_explanation: DecisionExplanation | None = None
    # Elapsed seconds per public phase group; a phase that did not run is null
    # rather than zero.
    timings: dict[str, float | None] = Field(default_factory=dict)
    index_references: list[IndexReference] = Field(default_factory=list)


class AuditRequest(PublicModel):
    """One audit request, with its corpus stated rather than inferred.

    Scope is explicit on purpose. An omitted, null, or empty selection used to
    mean "search everything", which turns a frontend bug — a document list that
    had not loaded yet — into a silently much larger query whose answer looks
    perfectly normal. There is no value that means "whatever you think best".
    """

    claim: str
    scope: Literal["all"] | list[int]
    limit: int = 20

    @model_validator(mode="before")
    @classmethod
    def _explicit_scope(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "scope" not in data or data["scope"] is None:
            raise ValidationError(
                "scope is required: pass 'all' or a non-empty list of document ids"
            )
        scope = data["scope"]
        if isinstance(scope, str):
            if scope != "all":
                raise ValidationError("scope must be 'all' or a list of document ids")
            return data
        if isinstance(scope, (list, tuple)):
            if not scope:
                raise ValidationError(
                    "scope is an empty list; pass 'all' to search every document"
                )
            for value in scope:
                # bool is an int in Python; True is not document 1.
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValidationError("scope must contain positive document ids")
            return data
        raise ValidationError("scope must be 'all' or a list of document ids")

    @field_validator("claim")
    @classmethod
    def _non_empty_claim(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValidationError("claim is required")
        if len(text) > 10_000:
            raise ValidationError("claim must be at most 10000 characters")
        return text

    @field_validator("limit")
    @classmethod
    def _bounded_limit(cls, value: int) -> int:
        if isinstance(value, bool) or not 1 <= value <= 50:
            raise ValidationError("limit must be between 1 and 50")
        return value

    @property
    def document_ids(self) -> list[int] | None:
        """The selection as retrieval wants it: ``None`` means every document."""
        return None if self.scope == "all" else sorted(set(self.scope))


class SourceToken(PublicModel):
    """One token of a proposed claim, at its position in the original text.

    Offsets are into the source exactly as it was submitted, so a caller
    highlights the real characters rather than a normalized copy of them.
    """

    model_config = ConfigDict(frozen=True)

    token: str
    start: int
    end: int


class GroundedClaim(PublicModel):
    """One proposed atomic claim, and the deterministic proof it came from the
    post -- or the category under which that proof failed.

    ``token_grounded`` is a statement about provenance only. It says every
    meaningful word was in the source in this order; it does not say the claim
    still means what the post meant, which is a separate check.
    """

    text: str
    source_tokens: list[SourceToken] = Field(default_factory=list)
    grounding: Literal["token_grounded", "not_grounded"]
    reason_code: str | None = None


class ClaimDecomposition(PublicModel):
    """What one post decomposed into, failures included.

    A proposal that failed grounding is reported rather than dropped: silently
    removing it hides that the splitter tried to introduce something.
    """

    source_text: str
    claims: list[GroundedClaim] = Field(default_factory=list)


class EntailmentCheck(BaseModel):
    """Whether the original post directly asserts one final subclaim.

    Not a verdict about the world and not evidence-based: the verifier sees the
    post and the subclaim only, and answers whether one says the other.
    """

    claim_index: int
    outcome: Literal["entailed", "ambiguous", "not_entailed", "contradicted"]
    reason_code: str = ""


class EntailmentBatch(BaseModel):
    """The verifier's reply for a whole batch, in one call.

    Required, not defaulted, for the reason ``Fact.quote`` is: a defaulted list
    turns a reply that answered nothing into an empty, *valid* answer, and an
    empty answer here would read as "no claim was contested".
    """

    checks: list[EntailmentCheck]


class ClaimVerification(PublicModel):
    """Both gates for one batch of final subclaims, and whether they passed.

    ``ok`` is decided here so a caller cannot arrive at a second, more generous
    definition of "may be audited". It is false whenever any claim failed
    grounding or is anything other than ``entailed``.
    """

    source_text: str
    claims: list[GroundedClaim] = Field(default_factory=list)
    entailment: list[EntailmentCheck] = Field(default_factory=list)
    ok: bool = False


class ProposedClaim(BaseModel):
    """One assertion the splitter proposes. Text only -- offsets and grounding
    are calculated here, never accepted from a model."""

    text: str


class ClaimSplit(BaseModel):
    """LLM claim-splitter response payload.

    ``claims`` is required so a reply that answered something else entirely is
    a schema failure -- and therefore a retry and then a hard error -- rather
    than an empty list indistinguishable from "this post asserts nothing".
    """

    claims: list[ProposedClaim]


class PublicError(PublicModel):
    """The one error shape every public surface returns.

    A stable code the caller branches on and a sentence written for a person.
    Never a driver message, a model reply, a prompt, a stack trace, or a path:
    those are local debug-log data and reach nobody through this type.
    """

    error_code: Literal[
        "validation_error",
        "unsupported_claim",
        "not_found",
        "index_not_ready",
        "dependency_unavailable",
        "internal_error",
    ]
    message: str
    retryable: bool = False


class IngestReport(PublicModel):
    document_id: int
    version_id: int
    status: VersionStatus
    fingerprint: str
    reused_existing: bool = False
    pages: int = 0
    evidence_units: int = 0
    visual_evidence_units: int = 0
    embedded_units: int = 0
    facts: int = 0
    # How much of the optional narrative enrichment landed. `degraded` is these
    # two numbers disagreeing, not a judgement call, and the failed keys are
    # what a targeted retry re-processes.
    fact_candidates_total: int = 0
    fact_candidates_succeeded: int = 0
    failed_fact_candidates: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    skipped_artifacts: list[str] = Field(default_factory=list)
    rejected_facts: list[str] = Field(default_factory=list)

    @property
    def fact_coverage(self) -> float | None:
        """Fraction of requested candidates that produced facts, or None."""
        if not self.fact_candidates_total:
            return None
        return round(self.fact_candidates_succeeded / self.fact_candidates_total, 4)


def phase_percent(completed: int | None, total: int | None) -> float | None:
    """Phase-local percentage, or None when the work is not measurable.

    A total of zero is a finished phase with nothing to do, not a division
    error: "0 of 0 crops verified" is 100% done.
    """
    if completed is None or total is None:
        return None
    if total <= 0:
        return 100.0
    return round(min(100.0, max(0.0, 100.0 * completed / total)), 2)


class ProgressEvent(PublicModel):
    """One phase update from a running ingestion or audit.

    Carries counts and identifiers only. Never a prompt, a model response,
    source text, a connection string, or a stack trace -- a frontend renders
    these directly, and an operator may paste one into a bug report.
    """

    operation: Literal["ingest", "audit"]
    phase: str
    status: Literal["start", "progress", "completed", "warning", "failed"]
    message: str
    # Null whenever the work is genuinely not measurable; an invented
    # percentage is worse than an honest spinner.
    completed: int | None = None
    total: int | None = None
    percent: float | None = None
    document_id: int | None = None
    audit_id: int | None = None
    current_item: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _derive_percent(self) -> Self:
        if self.percent is None:
            object.__setattr__(
                self, "percent", phase_percent(self.completed, self.total)
            )
        return self


# --- Frontend-support types -------------------------------------------------


class ModelHealth(PublicModel):
    role: Literal["embed", "chat", "vision"]
    name: str
    available: bool


class HealthReport(PublicModel):
    """System diagnostics. Carries no credentials, connection strings, raw
    driver exceptions, or prompts -- only categories and safe sentences."""

    database_reachable: bool = False
    schema_version: int | None = None
    schema_current: bool = False
    pgvector_version: str | None = None
    ollama_reachable: bool = False
    models: list[ModelHealth] = Field(default_factory=list)
    schema_embedding_dimensions: int | None = None
    configured_embedding_dimensions: int | None = None
    documents_ready: int = 0
    # Queryable, with some optional narrative fact coverage missing. Counted
    # separately from `ready` so "usable" and "complete" stay distinguishable.
    documents_degraded: int = 0
    # Recently active builds only. A build whose process died cannot write its
    # own failure, so it is classified by silence instead: still 'building'
    # with no progress for longer than the stale threshold is 'interrupted',
    # as is anything startup reconciliation has already marked so.
    documents_building: int = 0
    documents_failed: int = 0
    documents_interrupted: int = 0
    documents_inactive: int = 0
    audits_interrupted: int = 0
    # The queryable index: rows belonging to a ready version, which is exactly
    # what retrieval can reach. Historical rows from failed, superseded and
    # half-built versions are counted separately.
    evidence_units: int = 0
    embeddings: int = 0
    facts: int = 0
    stored_evidence_units: int = 0
    problems: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.database_reachable and self.schema_current and not self.problems


class DocumentSummary(PublicModel):
    document_id: int
    document_version_id: int
    name: str
    source_uri: str | None = None
    source_sha256: str | None = None
    status: VersionStatus
    page_count: int = 0
    evidence_count: int = 0
    fact_count: int = 0
    visual_evidence_count: int = 0
    embedding_model: str
    embedding_dimensions: int
    indexed_at: datetime | None = None
    # Server-side path metadata. A frontend backend may use these after its own
    # allowed-root validation; they are not for an untrusted browser.
    output_root: str
    source_pdf: str | None = None


class RemovalReport(PublicModel):
    document_id: int
    name: str
    deleted: dict[str, int] = Field(default_factory=dict)


class TraceCandidate(PublicModel):
    evidence_id: int
    pdf_page: int
    source_kind: EvidenceKind
    text: str
    lexical_rank: int | None = None
    lexical_score: float | None = None
    vector_rank: int | None = None
    vector_score: float | None = None
    graph_rank: int | None = None
    graph_score: float | None = None
    combined_rank: int | None = None
    combined_score: float = 0.0
    expanded_from: int | None = None
    visual_status: Literal[
        "not_applicable", "verified", "rejected", "unavailable"
    ] = "not_applicable"
    selected: bool = False
    reason: str | None = None


class AuditTrace(PublicModel):
    """Operational retrieval metadata for one audit.

    How candidates were found and ranked -- not model reasoning. Only the
    structured output and the concise rationale the product already shows are
    stored, so a trace can be handed to a UI without leaking a prompt.
    """

    audit_id: int
    claim: str
    # The corpus that was searched, recorded when the audit opened. Not derived
    # from citations: an insufficient verdict cites nothing and still searched
    # something, and a document removed afterwards must not erase the record.
    document_ids: list[int] = Field(default_factory=list)
    status: Literal["running", "completed", "failed"] = "completed"
    created_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    failure_code: str | None = None
    failure_phase: str | None = None
    retryable: bool | None = None
    parsed_claim: dict[str, Any] = Field(default_factory=dict)
    verdict: Verdict | None = None
    rationale: str | None = None
    evidence_quality: EvidenceQuality | None = None
    missing_qualifiers: list[str] = Field(default_factory=list)
    citation_ids: list[int] = Field(default_factory=list)
    candidates: list[TraceCandidate] = Field(default_factory=list)
    decision_explanation: DecisionExplanation | None = None
    timings: dict[str, float | None] = Field(default_factory=dict)
    index_references: list[IndexReference] = Field(default_factory=list)
    error: str | None = None

    @property
    def graph_candidates(self) -> list[TraceCandidate]:
        return [c for c in self.candidates if c.graph_rank is not None]

    @property
    def lexical_candidates(self) -> list[TraceCandidate]:
        return [c for c in self.candidates if c.lexical_rank is not None]

    @property
    def vector_candidates(self) -> list[TraceCandidate]:
        return [c for c in self.candidates if c.vector_rank is not None]


class EvidenceDetail(PublicModel):
    """Everything needed to render a cited region over its page image.

    The package renders nothing itself: it publishes one authoritative
    provenance representation and lets the caller draw.
    """

    evidence_id: int
    document_id: int
    document_version_id: int
    document_name: str
    pdf_page: int
    printed_page_label: str | None = None
    source_kind: EvidenceKind
    evidence_quality: EvidenceQuality
    geometry_precision: GeometryPrecision
    text: str
    quote: str | None = None
    table_context: dict[str, Any] = Field(default_factory=dict)
    heading_path: list[str] = Field(default_factory=list)
    page_width: float
    page_height: float
    page_image_path: str | None = None
    regions: list[Region] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)


# --- Internal records -------------------------------------------------------


class EvidenceUnit(BaseModel):
    """One immutable, addressable piece of source evidence."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    unit_key: str
    page: int
    kind: EvidenceKind
    text: str
    normalized_text: str
    citable: bool = True
    quality: EvidenceQuality = EvidenceQuality.DIRECT_TEXT
    heading_path: list[str] = Field(default_factory=list)
    table_context: dict[str, Any] = Field(default_factory=dict)
    artifact_path: str = ""
    regions: list[Region] = Field(default_factory=list)
    geometry_precision: GeometryPrecision = GeometryPrecision.BLOCK
    truncated_source: bool = False
    # Where this unit sits on its page, and what it belongs to. Context
    # expansion reads these instead of evidence ids, which record when a row
    # was inserted rather than what the page actually says.
    source_order: int | None = None
    context_key: str | None = None


class Fact(BaseModel):
    """A claim-shaped assertion pinned to the evidence that states it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    subject: str
    metric: str
    value_decimal: Decimal | None = None
    value_text: str | None = None
    unit: str | None = None
    direction: Literal["increase", "decrease", "level", "unknown"] = "unknown"
    comparison: Literal["=", ">=", "<=", ">", "<", "~"] = "="
    reporting_period: str | None = None
    baseline_period: str | None = None
    scope: str | None = None
    geography: str | None = None
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    extraction_method: Literal["table", "llm"] = "table"
    # Required, not defaulted: an optional field is omitted from the schema's
    # `required` list, and a model told a field is optional simply leaves it
    # out -- which made every extracted fact fail the quote check.
    quote: str = Field(
        description=(
            "Text copied verbatim from the passage, character for character, "
            "that states this fact."
        )
    )
    evidence_keys: list[str] = Field(default_factory=list)

    _parse_value = field_validator("value_decimal", mode="before")(_coerce_displayed_value)

    @property
    def fact_key(self) -> str:
        parts = (
            self.subject,
            self.metric,
            str(self.value_decimal),
            self.value_text or "",
            self.unit or "",
            self.reporting_period or "",
            self.baseline_period or "",
            self.scope or "",
            self.geography or "",
            "|".join(sorted(self.evidence_keys)),
        )
        return "\x1f".join(parts)


class ParsedClaim(BaseModel):
    """One atomic claim decomposed into comparable qualifiers."""

    subject: str = ""
    metric: str = ""
    value_decimal: Decimal | None = None
    value_text: str | None = None
    unit: str | None = None
    direction: Literal["increase", "decrease", "level", "unknown"] = "unknown"
    comparison: Literal["=", ">=", "<=", ">", "<", "~"] = "="
    reporting_period: str | None = None
    baseline_period: str | None = None
    scope: str | None = None
    geography: str | None = None
    approximate: bool = False
    key_terms: list[str] = Field(default_factory=list)

    _parse_value = field_validator("value_decimal", mode="before")(_coerce_displayed_value)

    @model_validator(mode="after")
    def _approximate_implies_tolerance(self) -> Self:
        if self.comparison == "~":
            object.__setattr__(self, "approximate", True)
        return self


class FactExtraction(BaseModel):
    """LLM fact-extraction response payload."""

    facts: list[Fact] = Field(default_factory=list)


class VisualResult(StrEnum):
    """What one re-verified crop turned out to be (PD-09).

    ``conflict`` is deliberately not "does not support": a crop showing a
    different figure is evidence *against* the claim, and folding it in with an
    unreadable one loses that.
    """

    SUPPORT = "support"
    CONFLICT = "conflict"
    ILLEGIBLE = "illegible"
    UNRELATED = "unrelated"


class VisualVerification(BaseModel):
    """Vision model response for one evidence crop.

    ``visible_text`` is what the model reports it can read, and is checked
    against the claim before a support or conflict is allowed to stand.
    ``reason`` is the model's own prose: kept for the local debug log, never
    persisted and never published -- ``reason_code`` is what a caller sees.
    """

    result: VisualResult
    visible_text: str = ""
    reason_code: str = ""
    reason: str = ""

    @property
    def supports_claim(self) -> bool:
        return self.result is VisualResult.SUPPORT

    @property
    def conflicts_with_claim(self) -> bool:
        return self.result is VisualResult.CONFLICT


class Adjudication(BaseModel):
    """Structured verdict from the semantic verifier."""

    verdict: Verdict
    rationale: str
    supporting_evidence_ids: list[int] = Field(default_factory=list)
    missing_qualifiers: list[str] = Field(default_factory=list)


__all__ = [
    "Adjudication",
    "AuditRequest",
    "AuditTrace",
    "Citation",
    "ClaimDecomposition",
    "ClaimResult",
    "ClaimSplit",
    "ClaimVerification",
    "DocumentSummary",
    "EntailmentBatch",
    "EntailmentCheck",
    "EvidenceDetail",
    "EvidenceKind",
    "EvidenceMatch",
    "EvidenceQuality",
    "EvidenceUnit",
    "Fact",
    "FactExtraction",
    "GeometryPrecision",
    "GroundedClaim",
    "HealthReport",
    "IngestReport",
    "ModelHealth",
    "ParsedClaim",
    "ProgressEvent",
    "ProposedClaim",
    "PublicError",
    "PublicModel",
    "phase_percent",
    "Region",
    "RegionRole",
    "RemovalReport",
    "SourceToken",
    "TraceCandidate",
    "Verdict",
    "VisualResult",
    "VersionStatus",
    "VisualVerification",
]
