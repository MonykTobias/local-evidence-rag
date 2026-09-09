"""Evidence-first claim auditing over completed ``document_extract`` output."""

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from .client import ClaimEvidence
from .config import Settings
from .errors import (
    ClaimEvidenceError,
    DependencyUnavailableError,
    IndexNotReadyError,
    NotFoundError,
    ValidationError,
)
from .progress import ProgressCallback
from .models import (
    AuditTrace,
    Citation,
    ClaimDecomposition,
    ClaimResult,
    ClaimVerification,
    DecisionExplanation,
    DocumentSummary,
    EntailmentCheck,
    EvidenceComparison,
    EvidenceDetail,
    EvidenceKind,
    EvidenceMatch,
    EvidenceQuality,
    GeometryPrecision,
    GroundedClaim,
    HealthReport,
    IndexReference,
    IngestReport,
    ModelHealth,
    NumericComparison,
    ProgressEvent,
    QualifierComparison,
    Region,
    RegionRole,
    RemovalReport,
    SourceToken,
    TraceCandidate,
    Verdict,
    VersionStatus,
)

__version__ = "0.2.0"

__all__ = [
    "AuditTrace",
    "Citation",
    "ClaimDecomposition",
    "ClaimEvidence",
    "ClaimEvidenceError",
    "ClaimResult",
    "ClaimVerification",
    "DecisionExplanation",
    "DependencyUnavailableError",
    "DocumentSummary",
    "EntailmentCheck",
    "EvidenceComparison",
    "EvidenceDetail",
    "EvidenceKind",
    "EvidenceMatch",
    "EvidenceQuality",
    "GeometryPrecision",
    "GroundedClaim",
    "HealthReport",
    "IndexNotReadyError",
    "IndexReference",
    "IngestReport",
    "ModelHealth",
    "NotFoundError",
    "NumericComparison",
    "ProgressCallback",
    "ProgressEvent",
    "QualifierComparison",
    "Region",
    "RegionRole",
    "RemovalReport",
    "Settings",
    "SourceToken",
    "TraceCandidate",
    "ValidationError",
    "Verdict",
    "VersionStatus",
    "__version__",
]
