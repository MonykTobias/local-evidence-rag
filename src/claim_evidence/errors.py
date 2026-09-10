"""Typed errors a caller can map without reading messages.

Driver exceptions and model response bodies stay inside the package. A caller
gets a category and a safe sentence, because a connection string or a prompt
echoed into an HTTP error is a leak, not a diagnostic.
"""

from __future__ import annotations


class ClaimEvidenceError(Exception):
    """Base class for every error this package raises on purpose."""


class NotFoundError(ClaimEvidenceError):
    """A requested document, audit, or evidence unit does not exist."""


class ValidationError(ClaimEvidenceError):
    """The caller's arguments are wrong; no state was changed."""


class UnsupportedClaimError(ValidationError):
    """The claim is well-formed but outside what version 1 audits.

    Separate from ``ValidationError`` because it is a different answer: the
    request was understood and is being declined, which is an HTTP 422 rather
    than a 400. It carries a stable ``reason_code`` so a caller branches on the
    category instead of matching the message text.

    Never an ``insufficient`` verdict. That verdict says the sources were
    searched and hold nothing comparable; using it here would report a limit of
    this tool as a finding about the document.
    """

    def __init__(self, message: str, *, reason_code: str = "unsupported_claim") -> None:
        super().__init__(message)
        self.reason_code = reason_code


class DependencyUnavailableError(ClaimEvidenceError):
    """PostgreSQL, pgvector, or the model server could not be reached or used."""


class IndexNotReadyError(ClaimEvidenceError):
    """The schema is missing, or no ready document version exists to query."""


__all__ = [
    "ClaimEvidenceError",
    "DependencyUnavailableError",
    "IndexNotReadyError",
    "NotFoundError",
    "UnsupportedClaimError",
    "ValidationError",
]
