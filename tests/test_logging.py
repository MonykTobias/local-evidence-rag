"""Logging is diagnostic only: opt-in, correlated, and free of source text."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from decimal import Decimal
from typing import Any

import pytest

import claim_evidence.audit as audit_module
from claim_evidence.audit import _adjudicate, _deterministic, parse_claim
from claim_evidence.config import Settings
from claim_evidence.facts import compare_detailed, heuristic_claim
from claim_evidence.models import (
    Adjudication,
    Citation,
    EvidenceKind,
    EvidenceQuality,
    ParsedClaim,
    Verdict,
)
from claim_evidence.normalize import parse_value
from claim_evidence.ollama import OllamaClient, OllamaError
from fake_ollama import FakeSession


def _fact(value: str = "-5.3") -> dict[str, Any]:
    return {
        "id": 901,
        "evidence_id": 7642,
        "subject": "IKEA",
        "metric": "IKEA retail sales change",
        "value_decimal": Decimal(value),
        "unit": "%",
        "direction": "decrease",
        "comparison": "=",
        "reporting_period": "FY2024",
        "baseline_period": "FY2023",
        "scope": "IKEA retail sales",
        "geography": None,
        "qualifiers": {"displayed_value": "-5.3%"},
    }


def _parsed(direction: str = "decrease") -> ParsedClaim:
    return ParsedClaim(
        subject="IKEA",
        metric="IKEA retail sales change",
        scope="IKEA retail sales",
        value_decimal=Decimal("5.3"),
        unit="%",
        direction=direction,
        comparison="=",
        reporting_period="FY2024",
        baseline_period="FY2023",
    )


def _candidate() -> dict[str, Any]:
    return {
        "row": {
            "id": 7642,
            "unit_key": "p0011:narrative:1",
            "kind": "narrative",
            "quality": "direct_text",
            "citable": True,
            "source_text": "Retail sales changed by 5.3 percent.",
            "heading_path": [],
            "table_context": {},
            "artifact_path": "page_0011/page.json",
            "geometry_precision": "line",
            "source_order": 1,
            "context_key": "p0011:narrative:1",
            "pdf_page": 11,
            "printed_page_label": None,
            "page_dir": "page_0011",
            "document_id": 1,
            "document_name": "report.pdf",
            "sha256": "a" * 64,
            "source_uri": None,
        }
    }


def _decide(monkeypatch: pytest.MonkeyPatch, audit_id: int = 418):
    monkeypatch.setattr(audit_module, "facts_for_evidence", lambda _conn, _ids: [_fact()])
    comparisons: list = []
    result = _deterministic(
        None,
        _parsed(),
        [_candidate()],
        {},
        {},
        comparisons,
        audit_id=audit_id,
    )
    return result, comparisons


def test_package_is_silent_by_default() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import logging, claim_evidence; "
            "logging.getLogger('claim_evidence.audit').warning('should be silent')",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_debug_can_be_enabled_without_duplicate_records(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="claim_evidence")
    assert parse_value("(40.2) %") == (Decimal("-40.2"), "%")
    records = [r for r in caplog.records if getattr(r, "event", None) == "value_normalized"]
    assert len(records) == 1
    assert records[0].normalized_value == "-40.2"


def test_info_omits_per_fact_noise(caplog) -> None:
    caplog.set_level(logging.INFO, logger="claim_evidence")
    for _ in range(25):
        compare_detailed(_parsed(), _fact())
    assert caplog.records == []


def test_audit_id_propagates_and_logging_does_not_change_result(
    caplog, monkeypatch: pytest.MonkeyPatch
) -> None:
    caplog.set_level(logging.CRITICAL, logger="claim_evidence")
    quiet, quiet_comparisons = _decide(monkeypatch)
    caplog.clear()
    caplog.set_level(logging.DEBUG, logger="claim_evidence")
    verbose, verbose_comparisons = _decide(monkeypatch)

    assert verbose == quiet
    assert verbose_comparisons == quiet_comparisons
    relevant = [
        r for r in caplog.records
        if getattr(r, "event", None)
        in {"deterministic_started", "fact_compared", "deterministic_completed"}
    ]
    assert relevant
    assert all(r.audit_id == 418 for r in relevant)


def test_fallback_logs_warning_and_preserves_fallback_result(caplog) -> None:
    class Unavailable:
        def structured(self, *_args, **_kwargs):
            raise OllamaError("model unavailable")

    claim = "IKEA retail sales decreased by 5.3% compared with FY23."
    caplog.set_level(logging.WARNING, logger="claim_evidence")
    result = parse_claim(Unavailable(), claim)  # type: ignore[arg-type]

    assert result == heuristic_claim(claim)
    records = [r for r in caplog.records if getattr(r, "event", None) == "claim_parse_fallback"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING


def test_prompts_secrets_and_schema_errors_are_not_logged(caplog) -> None:
    secret = "sk-test-DO-NOT-LOG-123456"
    client = OllamaClient(
        Settings(), FakeSession(chat_replies=["not json", "still not json"])
    )
    caplog.set_level(logging.DEBUG, logger="claim_evidence")

    with pytest.raises(OllamaError):
        client.structured(ParsedClaim, f"system {secret}", f"claim {secret}")

    assert secret not in caplog.text
    assert [
        r for r in caplog.records
        if getattr(r, "event", None) == "structured_response_invalid"
    ]


class _IkeaAdjudicator:
    settings = Settings(chat_model="test-adjudicator")

    def __init__(self, verdict: Verdict) -> None:
        self.verdict = verdict

    def structured(self, *_args, **_kwargs) -> Adjudication:
        return Adjudication(
            verdict=self.verdict,
            rationale="The cited sentence establishes the direction.",
            supporting_evidence_ids=[7642],
            missing_qualifiers=[],
        )


def test_ikea_polarity_trace_exposes_the_existing_policy_divergence(
    caplog, monkeypatch: pytest.MonkeyPatch
) -> None:
    citation = Citation(
        evidence_id=7642,
        document_id=1,
        document_name="IKEA Sustainability Report FY24",
        pdf_page=11,
        source_kind=EvidenceKind.NARRATIVE,
        quality=EvidenceQuality.DIRECT_TEXT,
        quote="Retail sales changed by 5.3 percent.",
        artifact_path="page_0011/page.json",
    )
    row = _candidate()["row"]
    monkeypatch.setattr(
        audit_module,
        "_verify_visuals",
        lambda *_args, **_kwargs: [(citation, row["source_text"], row)],
    )
    caplog.set_level(logging.DEBUG, logger="claim_evidence")

    decreased = _adjudicate(
        None,
        _IkeaAdjudicator(Verdict.SUPPORTED),  # type: ignore[arg-type]
        "IKEA retail sales decreased by 5.3% compared with FY23.",
        _parsed("decrease"),
        [_candidate()],
        {},
        {},
        {},
        scope_ambiguous=True,
        audit_id=501,
    )
    increased = _adjudicate(
        None,
        _IkeaAdjudicator(Verdict.CONTRADICTED),  # type: ignore[arg-type]
        "IKEA retail sales increased by 5.3% compared with FY23.",
        _parsed("increase"),
        [_candidate()],
        {},
        {},
        {},
        scope_ambiguous=True,
        audit_id=502,
    )

    assert decreased[0] is Verdict.SUPPORTED
    assert increased[0] is Verdict.INSUFFICIENT
    policies = {
        r.audit_id: r
        for r in caplog.records
        if getattr(r, "event", None) == "verdict_policy_applied"
    }
    assert policies[501].verdict_before == "supported"
    assert policies[501].verdict_after == "supported"
    assert policies[502].verdict_before == "contradicted"
    assert policies[502].verdict_after == "insufficient"
    assert policies[502].verdict_rule == "scope_not_comparable"
