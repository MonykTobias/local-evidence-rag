"""The build fingerprint: what must trigger a rebuild, and what must not.

No database and no model. Run from the repository root with
``python tests/test_fingerprint.py``.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

from fake_ollama import FakeSession
from fixtures import block, kpi_table, run_contract, write_output_root

from claim_evidence import Settings
from claim_evidence.ingest import build_fingerprint
from claim_evidence.ollama import OllamaClient
from claim_evidence.source import OutputReader, canonical_json

DIMENSIONS = 8


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def settings(**overrides) -> Settings:
    base = dict(
        database_url="postgresql://u:p@localhost:5433/claim_evidence_test",
        embed_dimensions=DIMENSIONS,
        chat_model="fact-model",
        vision_model="vision-model",
    )
    return Settings(**{**base, **overrides})


def client(session: FakeSession | None = None, config: Settings | None = None) -> OllamaClient:
    return OllamaClient(config or settings(), session or FakeSession(dimensions=DIMENSIONS))


def build_root(root: Path, **kwargs) -> Path:
    return write_output_root(
        root,
        pages=2,
        blocks=[block(1, 1, "Emissions fell against the 2020 baseline.")],
        tables={1: [kpi_table()]},
        **kwargs,
    )


def fingerprint_of(root: Path, *, config: Settings | None = None,
                   session: FakeSession | None = None,
                   source_sha256: str | None = "a" * 64,
                   entity: str = "Danone S.A.",
                   narrative: bool = True) -> str:
    config = config or settings()
    return build_fingerprint(
        OutputReader(root),
        config,
        client(session, config),
        source_sha256=source_sha256,
        reporting_entity=entity,
        extract_narrative_facts=narrative,
    )


# --- stability --------------------------------------------------------------


def test_the_same_inputs_give_the_same_value() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = build_root(Path(temp) / "run")
        check(
            fingerprint_of(root) == fingerprint_of(root),
            "an exact repeat is byte-for-byte the same fingerprint",
        )


def test_the_encoding_is_canonical() -> None:
    """Key order and spacing must not be part of the answer."""
    a = canonical_json({"b": 2, "a": {"z": 1, "y": [1, 2]}})
    b = canonical_json({"a": {"y": [1, 2], "z": 1}, "b": 2})
    check(a == b, "key order does not change the encoding")
    check(b" " not in a, "the encoding is compact")
    check(
        canonical_json({"k": "40,2 %"}).decode("utf-8") == '{"k":"40,2 %"}',
        "non-ASCII is encoded as UTF-8, not escaped",
    )


# --- what must change it ----------------------------------------------------


def test_a_changed_source_pdf_changes_it() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = build_root(Path(temp) / "run")
        check(
            fingerprint_of(root, source_sha256="a" * 64)
            != fingerprint_of(root, source_sha256="b" * 64),
            "a different source PDF is a different build",
        )


def test_every_evidence_bearing_artifact_changes_it() -> None:
    """One at a time, so a role that is silently ignored cannot hide."""
    from document_extract.contracts import PAGE_ARTIFACT_ROLES

    for role, filename in PAGE_ARTIFACT_ROLES.items():
        with tempfile.TemporaryDirectory() as temp:
            root = build_root(Path(temp) / "run")
            before = fingerprint_of(root)
            # Each artifact is changed in a way that keeps it *valid*: the
            # point is that a better extraction is a different build, not that
            # a corrupt file is rejected.
            target = root / "page_0001" / filename
            if filename.endswith(".png"):
                from PIL import Image

                Image.new("RGB", (600, 800), "black").save(target)
            elif filename.endswith(".jsonl"):
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps({"page": 1, "index": 9, "norm_rect": [0, 0, 1, 1]}) + "\n"
                    )
            elif filename.endswith(".json"):
                payload = json.loads(target.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    payload["revision"] = 2
                else:
                    payload = list(payload) + []
                    payload.append({"candidate_id": "tc999", "bbox": [0, 0, 1, 1],
                                    "markdown": "", "stats": {}})
                target.write_text(json.dumps(payload), encoding="utf-8")
            else:
                target.write_text(
                    target.read_text(encoding="utf-8") + "\nA corrected sentence.\n",
                    encoding="utf-8",
                )
            check(
                fingerprint_of(root) != before,
                f"a changed {role} ({filename}) invalidates the build",
            )


def test_an_appearing_optional_artifact_changes_it() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = build_root(Path(temp) / "run")
        before = fingerprint_of(root)
        (root / "page_0002" / "image_summaries.jsonl").write_text(
            json.dumps({"page": 2, "index": 1, "norm_rect": [0.1, 0.1, 0.5, 0.5]}) + "\n",
            encoding="utf-8",
        )
        check(
            fingerprint_of(root) != before,
            "a page that gains image summaries is a different build",
        )


def test_the_embedding_configuration_changes_it() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = build_root(Path(temp) / "run")
        before = fingerprint_of(root)
        check(
            fingerprint_of(root, config=settings(embed_model="other-embedder")) != before,
            "a different embedding model invalidates the build",
        )
        check(
            fingerprint_of(root, config=settings(embed_dimensions=16)) != before,
            "a different embedding dimension invalidates the build",
        )


def test_the_model_backend_changes_it_but_its_url_does_not() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = build_root(Path(temp) / "run")
        ollama = fingerprint_of(root)
        first = settings(
            model_backend="llamacpp",
            llamacpp_base_url="http://127.0.0.1:8080",
            llamacpp_embed_base_url="http://127.0.0.1:8081",
        )
        second = settings(
            model_backend="llamacpp",
            llamacpp_base_url="https://remote.example.test:8443",
            llamacpp_embed_base_url="https://remote.example.test:8444",
            llamacpp_api_key="secret",
        )
        check(fingerprint_of(root, config=first) != ollama, "the provider invalidates the build")
        check(
            fingerprint_of(root, config=first) == fingerprint_of(root, config=second),
            "llama.cpp URLs and credentials do not invalidate the build",
        )


def test_the_reporting_entity_changes_it() -> None:
    """Every stored fact is attributed to it, so it decides what gets stored."""
    with tempfile.TemporaryDirectory() as temp:
        root = build_root(Path(temp) / "run")
        check(
            fingerprint_of(root, entity="Danone S.A.")
            != fingerprint_of(root, entity="Nestle S.A."),
            "a different reporting entity is a different build",
        )


def test_the_fact_configuration_changes_it() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = build_root(Path(temp) / "run")
        before = fingerprint_of(root)
        check(
            fingerprint_of(root, narrative=False) != before,
            "turning narrative fact extraction off invalidates the build",
        )
        check(
            fingerprint_of(root, config=settings(chat_model="other-fact-model")) != before,
            "a different fact model invalidates the build",
        )
        check(
            fingerprint_of(root, config=settings(num_ctx=8192)) != before,
            "a different context window invalidates the build",
        )


def test_a_model_digest_changes_it() -> None:
    """A tag can be re-pulled; the weights behind it are what actually matter."""
    with tempfile.TemporaryDirectory() as temp:
        root = build_root(Path(temp) / "run")
        tag_only = fingerprint_of(root)
        pinned = fingerprint_of(
            root,
            session=FakeSession(
                dimensions=DIMENSIONS,
                model_digests={"qwen3-embedding:4b": "c" * 64},
            ),
        )
        check(tag_only != pinned, "learning the model's digest is a different build")


def test_model_identity_states_its_reproducibility() -> None:
    reachable = OllamaClient(
        settings(), FakeSession(dimensions=DIMENSIONS, model_digests={"m": "d" * 64})
    )
    identity = reachable.model_identity("m")
    check(identity["digest"] == "d" * 64, "a digest is recorded when the server has one")
    check(
        identity["reproducibility"] == "digest_pinned",
        "and the record says the build is reproducible",
    )
    identity = reachable.model_identity("no-digest-model")
    check(identity["digest"] is None, "a model without a digest records none")
    check(
        identity["reproducibility"] == "tag_only",
        "and says so, rather than implying a reproducibility it does not have",
    )


def test_a_new_mapping_version_cannot_reuse_the_old_index() -> None:
    """The rules changed, so the same artifacts are now a different index.

    Nothing in the output root moved: what changed is which of its records this
    package will store, and no query-time filter can remove a unit that should
    never have been written. The version is the only thing that says so.
    """
    from claim_evidence import ingest

    check(ingest.FINGERPRINT_VERSION == 3, "the current mapping rules are version 3")
    with tempfile.TemporaryDirectory() as temp:
        root = build_root(Path(temp) / "run")
        current = fingerprint_of(root)
        original = ingest.FINGERPRINT_VERSION
        try:
            ingest.FINGERPRINT_VERSION = 2
            previous = fingerprint_of(root)
        finally:
            ingest.FINGERPRINT_VERSION = original
        check(current != previous, "a v2-ready build is not reusable as v3")
        check(
            ingest.FINGERPRINT_VERSION == original,
            "and the module constant is back where it was",
        )


def test_the_extraction_contract_version_is_part_of_it() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = build_root(Path(temp) / "run")
        before = fingerprint_of(root)
        payload = json.loads((root / "run.json").read_text(encoding="utf-8"))
        payload["settings"]["visual_values_mode"] = "audit"
        (root / "run.json").write_text(json.dumps(payload), encoding="utf-8")
        check(
            fingerprint_of(root) != before,
            "a changed extraction setting reaches the fingerprint through run.json",
        )


# --- what must not change it ------------------------------------------------


def test_operational_settings_do_not_change_it() -> None:
    """Tuning throughput must not invalidate a perfectly good index."""
    with tempfile.TemporaryDirectory() as temp:
        root = build_root(Path(temp) / "run")
        before = fingerprint_of(root)
        for label, config in (
            ("request timeout", settings(request_timeout=30.0)),
            ("embed batch size", settings(embed_batch_size=4)),
            ("database URL", settings(database_url="postgresql://x:y@other:5433/ce_dev")),
            ("Ollama URL", settings(ollama_base_url="http://other-host:11434")),
            ("connect timeout", settings(database_connect_timeout=2.0)),
            ("stale threshold", settings(build_stale_minutes=5.0)),
        ):
            check(
                fingerprint_of(root, config=config) == before,
                f"changing the {label} does not invalidate the build",
            )


def test_audit_only_model_settings_do_not_change_it() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = build_root(Path(temp) / "run")
        check(
            fingerprint_of(root, config=settings(vision_model="other-vision"))
            == fingerprint_of(root),
            "the vision model runs at audit time and cannot alter what was stored",
        )


def test_a_stale_page_directory_does_not_change_it() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = build_root(Path(temp) / "run")
        before = fingerprint_of(root)
        stale = root / "page_0009"
        stale.mkdir()
        (stale / "docling_final.md").write_text("left over", encoding="utf-8")
        check(
            fingerprint_of(root) == before,
            "a directory the manifest does not list cannot move the fingerprint",
        )


def main() -> int:
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            print(f"\n--- {name} ---")
            function()
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
