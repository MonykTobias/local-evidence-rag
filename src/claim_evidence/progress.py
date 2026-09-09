"""Optional progress reporting for ingestion and auditing.

One callback, one emit path, no queue and no threads. A frontend passes a
callable and gets ordered phase events; passing nothing behaves exactly as
before.

A broken UI must never corrupt an index build, so a callback that raises is
disabled for the rest of the operation and the work continues.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from .errors import (
    DependencyUnavailableError,
    IndexNotReadyError,
    NotFoundError,
    ValidationError,
)
from .models import ProgressEvent

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[ProgressEvent], None]

# Stable codes a frontend can branch on, with the minimum retry classification.
ERROR_CODES: dict[str, bool] = {
    "validation_error": False,
    "dependency_unavailable": True,
    "index_not_ready": True,
    "not_found": False,
    "internal_error": True,
}
INTERNAL_ERROR_MESSAGE = "An internal error interrupted the operation."
DEPENDENCY_ERROR_MESSAGE = "A required service was unavailable."


def classify_error(exc: BaseException) -> tuple[str, bool, str]:
    """Map an exception to ``(error_code, retryable, safe_message)``.

    Only the four concrete public error types are quoted back to the caller;
    their messages are written for public consumption. Everything else -- a
    driver error, a model response body, an internal error of ours that wraps
    one, an unexpected bug -- is reported by category, because those messages
    carry hosts, credentials, prompts, and source text.
    """
    from .ollama import OllamaError

    if isinstance(exc, OllamaError):
        # Deliberately not str(exc): it embeds the model's response body to
        # make server-side debugging possible.
        return "dependency_unavailable", True, DEPENDENCY_ERROR_MESSAGE
    if isinstance(exc, ValidationError):
        return "validation_error", False, str(exc)
    if isinstance(exc, NotFoundError):
        return "not_found", False, str(exc)
    if isinstance(exc, IndexNotReadyError):
        return "index_not_ready", True, str(exc)
    if isinstance(exc, DependencyUnavailableError):
        return "dependency_unavailable", True, str(exc)
    # Other ClaimEvidenceError subclasses -- AuditError, IngestionError -- are
    # ours but their text is not vetted: they wrap driver and model failures
    # verbatim to make a server log useful. Category only.
    return "internal_error", True, INTERNAL_ERROR_MESSAGE


class ProgressReporter:
    """Funnels every event through one place so the rules hold everywhere.

    Construct one per operation. ``callback=None`` makes every method a no-op,
    which is what keeps the un-instrumented path free.
    """

    def __init__(
        self,
        callback: ProgressCallback | None,
        operation: str,
        *,
        document_id: int | None = None,
        audit_id: int | None = None,
    ) -> None:
        self._callback = callback
        self.operation = operation
        self.document_id = document_id
        self.audit_id = audit_id
        self.started = time.monotonic()
        self.broken = False
        self.phase = ""
        self._phase_started = self.started
        self._durations: dict[str, float] = {}

    @property
    def active(self) -> bool:
        return self._callback is not None and not self.broken

    @property
    def elapsed_seconds(self) -> float:
        return round(time.monotonic() - self.started, 3)

    def durations(self) -> dict[str, float]:
        """Seconds spent per phase so far, including the one still running.

        Measured whether or not a callback was supplied, because timings are
        part of the result rather than a side effect of instrumenting the UI.
        A phase that never ran is absent, not zero.
        """
        elapsed = dict(self._durations)
        if self.phase:
            elapsed[self.phase] = round(
                elapsed.get(self.phase, 0.0) + (time.monotonic() - self._phase_started), 3
            )
        return elapsed

    def _close_phase(self, phase: str) -> None:
        now = time.monotonic()
        self._durations[phase] = round(
            self._durations.get(phase, 0.0) + (now - self._phase_started), 3
        )
        self._phase_started = now

    def emit(
        self,
        phase: str,
        status: str,
        message: str,
        *,
        completed: int | None = None,
        total: int | None = None,
        current_item: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """The single emit path. Never raises."""
        if phase != self.phase:
            if self.phase:
                self._close_phase(self.phase)
            else:
                self._phase_started = time.monotonic()
            self.phase = phase
        elif status == "completed":
            self._close_phase(phase)
        logger.debug(
            "operation phase",
            extra={
                "event": "operation_phase", "operation": self.operation,
                "phase": phase, "status": status,
                "document_id": self.document_id, "audit_id": self.audit_id,
                "completed": completed, "total": total,
                "elapsed_seconds": self.elapsed_seconds,
            },
        )
        if not self.active:
            return
        event = ProgressEvent(
            operation=self.operation,
            phase=phase,
            status=status,
            message=message,
            completed=completed,
            total=total,
            document_id=self.document_id,
            audit_id=self.audit_id,
            current_item=current_item,
            details=details or {},
        )
        try:
            self._callback(event)  # type: ignore[misc]
        except Exception:
            # A UI that crashes is a UI problem. Stop talking to it and let the
            # index build finish.
            self.broken = True
            logger.warning(
                "progress callback disabled after failure",
                extra={
                    "event": "progress_callback_failed", "operation": self.operation,
                    "phase": phase, "document_id": self.document_id,
                    "audit_id": self.audit_id,
                },
            )

    # --- convenience wrappers over the one emit path ------------------------

    def start(self, phase: str, message: str, *, total: int | None = None) -> None:
        self.emit(phase, "start", message, completed=0 if total else None, total=total)

    def step(
        self,
        phase: str,
        message: str,
        *,
        completed: int,
        total: int,
        current_item: str | None = None,
    ) -> None:
        self.emit(
            phase,
            "progress",
            message,
            completed=completed,
            total=total,
            current_item=current_item,
        )

    def done(
        self,
        phase: str,
        message: str,
        *,
        completed: int | None = None,
        total: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.emit(
            phase,
            "completed",
            message,
            completed=completed,
            total=total,
            details=details,
        )

    def warn(self, phase: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        self.emit(phase, "warning", message, details=details)

    def fail(self, exc: BaseException, *, phase: str | None = None) -> None:
        """Emit one safe terminal event. The caller re-raises afterwards."""
        failed_phase = phase or self.phase or "unknown"
        code, retryable, message = classify_error(exc)
        logger.error(
            "operation failed",
            extra={
                "event": "operation_failed", "operation": self.operation,
                "phase": failed_phase, "document_id": self.document_id,
                "audit_id": self.audit_id, "error_type": type(exc).__name__,
                "error_code": code, "retryable": retryable,
                "elapsed_seconds": self.elapsed_seconds,
            },
        )
        self.emit(
            failed_phase,
            "failed",
            message,
            details={
                "failed_phase": failed_phase,
                "error_code": code,
                "retryable": retryable,
                "elapsed_seconds": self.elapsed_seconds,
            },
        )


# Detailed phases grouped into the durations a caller is shown. One mapping,
# so a renamed phase does not silently drop out of the reported timings.
AUDIT_TIMING_GROUPS: dict[str, tuple[str, ...]] = {
    "parsing": ("parsing_claim",),
    "retrieval": ("retrieving_graph", "retrieving_full_text", "retrieving_vectors"),
    "fusion_context": ("fusing_candidates", "expanding_context"),
    "visual_verification": ("verifying_visuals",),
    "verdict": ("deciding_verdict",),
    "persistence": ("persisting_trace",),
}


def audit_timings(reporter: ProgressReporter) -> dict[str, float | None]:
    """Public timing groups for one audit.

    A group whose phases never ran is ``None``, not ``0.0``: an audit that
    skipped visual verification did no zero-second work, it did none.
    """
    elapsed = reporter.durations()
    timings: dict[str, float | None] = {}
    for name, phases in AUDIT_TIMING_GROUPS.items():
        measured = [elapsed[phase] for phase in phases if phase in elapsed]
        timings[name] = round(sum(measured), 3) if measured else None
    timings["total"] = reporter.elapsed_seconds
    return timings


__all__ = [
    "AUDIT_TIMING_GROUPS",
    "audit_timings",
    "DEPENDENCY_ERROR_MESSAGE",
    "ERROR_CODES",
    "INTERNAL_ERROR_MESSAGE",
    "ProgressCallback",
    "ProgressReporter",
    "classify_error",
]
