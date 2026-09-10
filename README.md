# claim_evidence

Turns completed [`document_extract`](../document_extract) output folders into an
evidence-first RAG and a claim-focused knowledge graph.

One question, answered conservatively:

> Given one atomic claim, return `supported`, `contradicted`, `mixed`, or
> `insufficient`, backed by exact source evidence containing the 1-based PDF
> page and one or more bounding boxes.

A claim may be supported by text, tables, or a chart, but **visual evidence must
be re-checked from the cited crop** before it can support a verdict, and
generated Markdown never supports anything on its own.

This is the **canonical RAG/knowledge-graph backend**. The sibling
`document_knowledge` directory is a non-runtime scaffold and must not be
installed or imported.

## Architecture

```text
PDF
 │
 ▼
Document Extraction
 │
 ▼
Evidence Units
 ├── Structured Facts / Knowledge Graph
 ├── PostgreSQL Full-Text Search
 └── pgvector Semantic Search
              │
              ▼
       Hybrid Retrieval
              │
              ▼
     Evidence Verification
              │
              ▼
 Supported / Contradicted
 Mixed / Insufficient
```
## Install

Python 3.12. One environment holds all three prototype packages; install them
together from the directory containing the checkouts, so nothing needs a
`sys.path` insertion:

```bash
py -3.12 -m pip install -e claim_evidence -e document_extract -r gw_detector_v2/requirements.txt
```

For this package alone:

```bash
python -m pip install -e .
```

Start the development database (PostgreSQL 18 + pgvector 0.8.5):

```bash
docker compose up -d
```

Pull the models into Ollama:

```bash
ollama pull qwen3-embedding:4b
```

Ollama remains the default model backend. A native Windows llama.cpp setup is
described below.

## Configuration

| Variable | Default |
|---|---|
| `CLAIM_EVIDENCE_DATABASE_URL` | `postgresql://claim_evidence:claim_evidence@localhost:5433/claim_evidence` |
| `CLAIM_EVIDENCE_DATABASE_CONNECT_TIMEOUT` | `10` (seconds) |
| `CLAIM_EVIDENCE_BUILD_STALE_MINUTES` | `60` |
| `CLAIM_EVIDENCE_MODEL_BACKEND` | `ollama` (`ollama` or `llamacpp`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` |
| `CLAIM_EVIDENCE_LLAMACPP_BASE_URL` | `http://127.0.0.1:8080` (chat) |
| `CLAIM_EVIDENCE_LLAMACPP_EMBED_BASE_URL` | `http://127.0.0.1:8081` |
| `CLAIM_EVIDENCE_LLAMACPP_VISION_BASE_URL` | chat URL when unset |
| `CLAIM_EVIDENCE_LLAMACPP_API_KEY` | unset |
| `CLAIM_EVIDENCE_EMBED_MODEL` | `qwen3-embedding:4b` |
| `CLAIM_EVIDENCE_EMBED_DIMENSIONS` | `1024` |
| `CLAIM_EVIDENCE_CHAT_MODEL` | `hf.co/unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q8_K_XL` |
| `CLAIM_EVIDENCE_VISION_MODEL` | same as the chat model |
| `CLAIM_EVIDENCE_EMBED_BATCH_SIZE` | `32` |
| `CLAIM_EVIDENCE_REQUEST_TIMEOUT` | `600` |
| `CLAIM_EVIDENCE_NUM_CTX` | `16384` |

Point `CLAIM_EVIDENCE_DATABASE_URL` at a managed instance to skip Compose
entirely. The embedding dimension is templated into the schema at `db init`, so
changing dimensions means re-running `db init` on a fresh database. Changing
the embedding provider, model, quantization, or preprocessing while retaining
the same dimension requires re-ingestion. Give changed llama.cpp weights a new
server alias: llama.cpp does not expose a weight digest that this package can
pin, and an unchanged alias cannot prove reproducibility.

`CLAIM_EVIDENCE_NUM_CTX` is the context window for chat and vision requests.
Ollama receives it on every request; llama.cpp must be started with the same
value and health reports a mismatch. Embeddings are unaffected. Every
structured call this package makes is bounded — one evidence passage for fact
extraction, at most `MAX_PASSAGES` × `PASSAGE_CHARS` for adjudication — so the
model's 64k default buys nothing and spends KV-cache memory that would
otherwise hold model layers on the GPU. Raise it only for prompts measured to
need it.

### llama.cpp on Windows

Download a CUDA-enabled `llama-server.exe` from the official llama.cpp
releases, then download the GGUF files separately. The initial configuration
uses two manually managed servers: Qwen3-VL-4B-Instruct UD-Q8_K_XL plus its
matching `mmproj-F16.gguf` for chat and vision, and Qwen3-Embedding-4B Q8_0 for
embeddings. Replace the example paths with the real locations:

```powershell
& 'C:\path\to\llama-server.exe' -m 'C:\models\Qwen3-VL-4B-Instruct-UD-Q8_K_XL.gguf' --mmproj 'C:\models\mmproj-F16.gguf' --alias 'hf.co/unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q8_K_XL' --host 127.0.0.1 --port 8080 -c 16384 -np 1 -ngl auto --jinja
```

```powershell
& 'C:\path\to\llama-server.exe' -m 'C:\models\Qwen3-Embedding-4B-Q8_0.gguf' --alias 'qwen3-embedding:4b' --host 127.0.0.1 --port 8081 --embedding --pooling last -c 8192 -b 8192 -ub 8192 -ngl auto
```

Configure this package in the terminal where it runs:

```powershell
$env:CLAIM_EVIDENCE_MODEL_BACKEND = 'llamacpp'
$env:CLAIM_EVIDENCE_LLAMACPP_BASE_URL = 'http://127.0.0.1:8080'
$env:CLAIM_EVIDENCE_LLAMACPP_EMBED_BASE_URL = 'http://127.0.0.1:8081'
$env:CLAIM_EVIDENCE_NUM_CTX = '16384'
$env:CLAIM_EVIDENCE_EMBED_DIMENSIONS = '1024'
claim-evidence health
```

For a remote deployment, set both base URLs to absolute `https://` origins,
set the vision URL only when vision has a separate origin, and set
`CLAIM_EVIDENCE_LLAMACPP_API_KEY` when the server requires a bearer key. URLs
must be origins without a `/v1` suffix. TLS verification remains enabled.

After health succeeds, run the read-only live smoke check with a representative
image crop:

```powershell
python tests/smoke_llamacpp.py C:\path\to\crop.png
```

Switching backends never reuses an incompatible vector index. Re-run `ingest`
for each source using the selected backend. Changing only the fact-generation
model also calls for full re-ingestion rather than `retry-facts`.

An unreachable database raises `DependencyUnavailableError` after
`CLAIM_EVIDENCE_DATABASE_CONNECT_TIMEOUT` rather than blocking on the operating
system's network timeout. libpq takes whole seconds and treats anything below 2
as 2. A nonpositive or unparseable value is a `ValidationError` at startup, and
neither error quotes the connection string.

**On the 1024 default:** `qwen3-embedding:4b` natively returns 2560 dimensions
and Ollama's `/api/embed` exposes no output-dimension parameter. Qwen3-Embedding
is Matryoshka-trained, so the client keeps the leading 1024 values and
re-normalizes — the documented way to shorten an MRL embedding. A model that is
*not* MRL-trained must be configured with its native dimension; a vector shorter
than the configured width is a hard error rather than a silent pad.

## CLI

```bash
claim-evidence db init
```

```bash
claim-evidence ingest OUTPUT_ROOT --pdf SOURCE_PDF
```

```bash
claim-evidence search "Scope 1 and 2 emissions versus 2020" --json
```

```bash
claim-evidence audit "Danone reduced Scope 1 and 2 energy and industry emissions by 40.2% in 2025 versus 2020." --json
```

```bash
claim-evidence health
```

```bash
claim-evidence documents
```

```bash
claim-evidence trace AUDIT_ID --json
```

```bash
claim-evidence evidence EVIDENCE_ID --json
```

```bash
claim-evidence remove DOCUMENT_ID --confirm DOCUMENT_ID
```

`ingest --no-narrative-facts` indexes evidence and deterministic table facts but
skips the LLM narrative fact extractor, which is the slow part of a first run.
`ingest --force` rebuilds an unchanged source without taking the current
version out of service.

## Python API

```python
from claim_evidence import ClaimEvidence

client = ClaimEvidence.from_env()

ingest_report = client.ingest_document(
    output_root,
    source_pdf=source_pdf,
    source_uri=None,
    progress=lambda event: print(event.phase, event.percent),  # optional
)

result = client.audit_claim(
    "Danone reduced Scope 1 and 2 energy and industry emissions "
    "by 40.2% in 2025 versus 2020."
)

matches = client.search_evidence("Scope 1 and 2 emissions versus 2020", limit=20)
```

`document_ids` is validated before anything is embedded, parsed, or persisted.
`None` and `[]` both mean every ready document; an unknown id raises
`NotFoundError`, an id with no ready version raises `IndexNotReadyError`, and a
malformed one raises `ValidationError`. A mixed selection fails as a whole
rather than quietly searching the valid part, and an empty index raises rather
than returning an `insufficient` verdict — "nothing was retrieved" is a
statement about the request, not about the report.

`ClaimResult.evidence_quality` is one of `direct_text`, `direct_table`,
`verified_visual`, `coarse_region`, `none`. There is deliberately no numeric
model confidence: an uncalibrated number reads as precision the system does not
have.

`search_evidence` returns citable evidence only. Every `citation.source_kind` is
`narrative`, `table_row`, `table_value`, or `visual` — never `page_markdown`,
which exists to be read alongside its sources during an audit and is not a
result a caller can quote or point at on a page.

### Splitting a post into claims

`audit_claim` audits one assertion. A social-media post usually makes several,
so two calls turn one into the other — and neither of them audits anything.

```python
decomposition = client.decompose_claims(post, reporting_entity="IKEA")
# decomposition.claims[i].text          the proposed assertion
# decomposition.claims[i].grounding     token_grounded | not_grounded
# decomposition.claims[i].source_tokens [{token, start, end}, ...]
# decomposition.claims[i].reason_code   why grounding failed, when it did

verification = client.verify_claims(
    post, ["...", "..."], reporting_entity="IKEA"
)
# verification.ok            every claim grounded *and* entailed
# verification.entailment[i] entailed | ambiguous | not_entailed | contradicted
```

The splitter is a model, so what it returns is a proposal. Two independent
checks stand between a proposal and an audit:

* **Grounding** is deterministic and runs no model. Every meaningful token of a
  proposed claim must appear in the post, in the same order, and its character
  offsets are recorded. Numbers, signs, units, dates and fiscal periods,
  negation, modality, bounds, direction, and causal words are protected: they
  are never normalized away, and altering one is refused as
  `protected_token_changed`. Embedding similarity is deliberately not used —
  two sentences sit close in vector space while disagreeing on exactly those.
* **Entailment** is one separate structured call over the whole batch, made
  after the user has reviewed and edited the rows. It sees the post and the
  final subclaims only, never document evidence, and answers whether the post
  directly asserts each one. A single claim identical to the whole input skips
  it: its meaning cannot have changed.

Grounding proves provenance, not meaning. "Emissions fell 20% and water use
fell 10%" grounds "Emissions fell 10%" — every word is there, in order — and it
is the entailment check that refuses it.

`verify_claims` recomputes grounding from the text submitted to it, because the
review step lets a user edit a row and an edited row is a new proposal. `ok` is
decided here so no caller arrives at a second, more generous definition of
"may be audited": it is false if *any* claim failed either gate, and a caller
must then create no audit rows at all. A post with no checkable assertion is
`UnsupportedClaimError(reason_code="no_atomic_claim")`; more than 20 is
`too_many_claims`. An unreachable or unparseable model is
`DependencyUnavailableError` — retryable, and never a smaller batch.

### Why a verdict came out that way

`ClaimResult` and `AuditTrace` both carry `decision_explanation`, `timings`, and
`index_references`. The explanation is produced by the same comparison that
decided the verdict, so the two can never disagree, and a caller never has to
re-derive whether a qualifier matched.

```text
decision_explanation.decided_by    deterministic_comparison | semantic_adjudication | no_evidence
decision_explanation.verdict_rule  exact_numeric_match | bounded_numeric_match
                                   | comparable_numeric_conflict | mixed_comparable_facts
                                   | semantic_evidence_support | semantic_evidence_conflict
                                   | scope_not_comparable | no_citable_evidence
                                   | missing_material_qualifier
decision_explanation.evidence_comparisons[]
    evidence_id  fact_id  pdf_page
    qualifiers[] qualifier claim_value source_value status reason
    numeric      claim_value claim_operator claim_direction
                 source_value source_operator source_unit outcome reason
```

A qualifier is `match` only where the comparison actually established
comparability; a qualifier the source omits is `missing`, and `mismatch` needs
both sides present and disagreeing. `numeric.outcome` is `incomparable` when a
qualifier blocked the arithmetic and `not_applicable` when either side states no
number — a broad claim is never turned into a numeric conflict against narrower
evidence. There is one entry per fact examined, so multiple or disagreeing facts
keep their own comparisons.

`verdict_rule` is operational metadata, not model reasoning; the user-facing
prose stays in `rationale`. Nothing in the explanation is a prompt, a raw model
reply, or a local path.

`AuditTrace` also carries the audit's own lifecycle: `status` is `running`,
`completed`, or `failed`, alongside `completed_at`/`failed_at` and, on failure,
a safe `failure_code`, `failure_phase`, and `retryable`. A failed audit is an
explicitly failed row, not an ambiguous one with a null verdict, and it stores
no exception text.

`document_ids` is the corpus recorded when the audit opened — for an unscoped
audit, the exact ready ids at that moment. It is never reconstructed from
citations, so it survives an `insufficient` verdict, a failure, and a document
being removed afterwards. A scope that fails validation opens no audit row at
all.

`timings` reports elapsed seconds for `parsing`, `retrieval`, `fusion_context`,
`visual_verification`, `verdict`, `persistence`, and `total`. A group whose
phases never ran is `null` rather than `0.0`. `index_references` pins the exact
ready `document_version_id`, embedding model, and dimension that answered the
audit, and is recorded even for an `insufficient` verdict with no citations.

Each `Citation` carries the document identity, `pdf_page` (always the 1-based
PDF index, never a printed footer number), the quote or table cells, the
artifact it came from, and a list of `Region`s normalized to
`[left, top, right, bottom]` in a top-left 0-1 space with the original
coordinates preserved.

## What the package indexes

The run-level `manifest.json` is the authoritative page list. Stale `page_*`
directories are reported and ignored; any page that is not `completed`, any
duplicate or missing page, and any missing artifact aborts ingestion rather than
producing an index with quiet holes in it.

A run does not have to start at page 1. `page_state.total_pages` is the count of
*selected* pages, so the manifest must cover a contiguous run of that length
starting at its own first page — `1..494`, `1..20`, `10..20`, and a lone `359`
are all valid, and a hole inside the range still fails closed. Page numbers are
never renumbered: PDF page 359 is `359` in evidence, search results, and
citations. Every selected page's checkpoint must agree on the count.

Each manifest `page_dir` is resolved and required to stay strictly inside the
registered output root before any artifact is opened. An absolute path, a `..`
traversal, or a symlink pointing elsewhere is rejected, and the error names the
page rather than the path it tried to reach.

| Artifact | Used for | Citable |
|---|---|---|
| `blocks.jsonl` | headings, prose, block geometry | yes |
| `table_candidates.json` | rows, values, cell geometry | yes |
| `image_summaries.jsonl` | chart retrieval hints | only after crop verification |
| `docling_final.md` | mapped page context | no |
| `page.png` | crops for visual re-verification | via the crop |

### Which images become evidence

Only charts, and only ones the producer typed as such at both stages:
`triage_type` and `summary_type` must both be `chart`. `classification` is the
legacy field and is deliberately not consulted — reading it would re-admit every
untyped record. A record is also refused when the producer flagged it
(`summary_warnings`), marked it redundant with the page text
(`summary_redundant`), left the summary empty, or when the summary hedges its
own figures: `approximate`, `estimated`, `guess`, `unclear`, `illegible`, or
`not readable` as whole words. `Title: (none)` is an absence, not a hedge, and
does not disqualify anything.

The bar is deliberately "could re-cropping this settle a claim". A photo, an
infographic, a diagram, or a map yields a model-written description, and a
description is not a figure the audit can check. A rectangle that will not
convert to a crop region drops its own record and leaves the rest of the page
alone.

### Mapped page context

`docling_final.md` is generated: refined, repaired, and reordered, so a
sentence in it may be one the page never printed. It is indexed only where it
maps exactly back onto the units it was generated from, and the mapping is
built at ingestion by string matching — no embeddings, no similarity, no model.

Each blank-line paragraph, and each table row on its own, is a candidate
segment. Where a pipe table declares a header with a `|---|` separator, only
the rows below it are candidates: a column label is not a reading, and "2024"
as a header must not stand in for "2024" as a value. Tables that declare no
separator keep every row, because older runs emit them that way.

A segment maps in one of two ways, tried in that order.

**Literal anchors.** Every substantive character is accounted for by an anchor:
a narrative block's text, a table cell as printed, or an image's own path.
Markdown punctuation between anchors is fine; a clause no unit accounts for is
not, and drops the whole segment. An anchor two citable units both print is
ambiguous and drops the segment too — a citation chosen between two equal
candidates is a guess, not provenance.

**Cell by cell**, for a pipe row only, and only after literal anchoring failed.
The refinement rebuilds a table out of a chart it read, so no unit's text is the
row and the anchors never line up. Each non-empty cell — stripped of `**`, `__`,
backticks and `<br>` — must be a substring of exactly one citable unit *on the
same page*. A cell matching nothing, or two units, rejects the whole row. It is
a much stricter rule than literal anchoring because a cell is short enough to
turn up somewhere it did not come from.

Either way a stored segment is `citable = false`, `quality = none`, derives no
facts, is never returned as a neighbour, and carries the `unit_key`s of its
sources in reading order rather than any geometry of its own.

It is read only after direct evidence has failed. Direct retrieval ranks
citable rows only — generated Markdown is not in that pool at all, so it can
neither outrank the evidence it repeats nor spend one of the caller's results.
If the deterministic comparison or the adjudicator reaches `supported`,
`contradicted`, or `mixed`, no Markdown is retrieved at all, and the audit trace
holds no `page_markdown` candidate to show for it. On `insufficient` — including
when direct retrieval returned nothing — a second, Markdown-only retrieval runs.
Its rows are not context-expanded: a segment reaches its sources through the
keys it was written with, not through page neighbours. Those sources join the
pool and the arithmetic runs again over the wider set before the adjudicator is
asked a second time.

The value is the case direct retrieval structurally cannot serve: one sentence
Docling split across two boxes exists in neither block, so neither can be
retrieved by it. The segment offers both together — and the citations are still
those two blocks, with their own regions.

In the prompt a segment is a `<page_context>` block with no id, placed before
the `<evidence>` blocks it maps to. A group is atomic: the segment and every
unit it resolves to go in together or not at all, so context never appears
beside a subset of what it describes. A group that will not fit under
`MAX_PASSAGES`, or that contains a visual whose crop was rejected, is dropped
whole. The cap counts context blocks; nothing else about citations changes.

Each of those outcomes is named in the trace, on the segment's own candidate
row: `markdown group used as page context`, `markdown group rejected: visual
crop`, `markdown group rejected: passage cap`, `markdown group rejected: missing
source`, `markdown group rejected: empty mapping`, or `markdown group not used:
deterministic fallback decided claim`. "It did not appear in the prompt" is one
observation with several causes, and the trace says which.

There is no in-place upgrade. `FINGERPRINT_VERSION` is `3`, so an index built
before this rebuilds on the next `ingest` of the same output root — no
re-extraction and no schema change, but the documents do have to be re-indexed
before the fallback is available to them.

Block text prefers `text_full` and falls back to the 500-character `text`,
flagging the unit as truncated. Older runs without `text_full` therefore index
fine, but a strict supporting verdict still needs citable source evidence.

Table values cite four regions -- descriptor, column header, unit, and value
cell. A cell with no geometry of its own falls back to the union of its row's
cells (`row` precision), then to the table (`table` precision).

Table-value cells are not embedded: they are reached through their row, their
fact, and lexical search, and embedding thousands of near-identical `(40.2) %`
strings buys no semantic recall. Narrative blocks, table rows, visual regions,
and page Markdown are embedded.

## How a verdict is reached

1. Parse the claim into subject, metric, value, unit, periods, scope, geography.
2. Retrieve independently through fact/graph filters, PostgreSQL full-text, and
   pgvector.
3. Merge with reciprocal-rank fusion, boosting exact numbers, years, units, and
   scope tokens.
4. Expand the top candidates with their page neighbours, following the source:
   units sharing a `context_key` first (a table value's own row and its sibling
   cells), then the same page by distance in `source_order`, then evidence id
   only to break ties. Auto-increment ids reflect insertion and resume history,
   not what the page says, so they no longer decide what counts as context. A
   version indexed before this falls back to page order until it is re-indexed.
5. Crop and re-verify any visual candidate.
6. Compare arithmetically whenever every material qualifier aligns.
7. Fall back to the structured LLM verifier only for semantic qualification and
   ambiguity.
8. Only on `insufficient`, retry steps 5–7 with the mapped page context and the
   original units it resolves to. The citations remain those original units.

The arithmetic has three rules, and a model decides none of them:

* An exact claim needs exact displayed agreement on `Decimal`.
* A bound — "at least 40%", "no more than 30%" — is satisfied or it is not.
  When the claim states a direction the comparison is on magnitude, because
  "reduced by at least 40%" is about how big the drop was and the signed value
  of a 40.2% reduction is the *smaller* number.
* A hedge — "about", "roughly", "approximately" — is compared at the precision
  the claim was written to. "about 40%" accepts a reported 40.2%; "about 40.0%"
  does not, and "about 21.3" still refuses 21.5. There is no global percentage
  tolerance: 5% of a small figure and 5% of a large one are different claims,
  and neither is what the writer wrote.

Two things are canonicalized at comparison time rather than at index time, so
an existing index answers under the current rules without being rebuilt:

* **Periods.** `FY24` and `FY2024` are one fiscal period. Neither is calendar
  `2024` — a fiscal year that ends in June overlaps two calendar years, and
  equating them would compare a figure to a period the report never reported.
* **Units.** `MtCO2e`, `Mt CO2-eq`, and "million tonnes of CO₂ equivalent" are
  one unit, and same-quantity prefixes convert with exact `Decimal` multipliers
  (21.3 MtCO2e is 21300000 tCO2e, not 21299999.999999998). A tonne is not a
  tonne of CO2-equivalent, energy is not mass, and a bare "tonnes of CO2" is
  not canonicalized to CO2e. Anything not convertible stays incomparable and
  falls through to cited semantic adjudication.

`compare()` returns *incomparable*, not *contradicted*, when a qualifier does
not line up, which is what stops a vague claim from being forced into a
confident contradiction against the nearest number that happens to exist.

Invalid model JSON gets one retry, then a hard error. A supported verdict
without a direct or re-verified citation is downgraded to `insufficient`.

Facts come from two places. Table facts are arithmetic and need no model: the
column header gives the reporting period, `vs. 2020` in the row descriptor
gives the baseline, and accounting parentheses give the sign. Narrative facts go
through the LLM, but each one must echo a quote that really appears in its own
evidence unit or it is discarded.

## Measured on the Danone URD

The completed 494-page run indexes to 21,988 evidence units (5,901 narrative,
3,457 table rows, 11,535 table values, 602 visual, 493 page-Markdown) and
11,168 deterministic table facts, in roughly seven minutes end to end.

Geometry precision across those units: 14,323 cell, 5,901 block, 602 crop, 493
page, 472 row-fallback, 197 table-fallback. The 472 row fallbacks are exactly
the table cells the extractor emitted without their own bounding box.

These counts were measured at `FINGERPRINT_VERSION` 2. The narrative, table and
fact figures are unaffected by the current rules, but the visual, page-Markdown,
crop and page numbers will be lower under them: only typed, unhedged charts
become visual units now, and a segment whose anchors are ambiguous is no longer
indexed. Re-run `tests/acceptance_danone.py` to restate them.

Scanning every table fact in the document against the three reference claims
yields 2 matches and 0 conflicts for the 40.2% claim, 0 and 2 for the 90%
variant, and nothing comparable for "all carbon emissions" — so the first two
verdicts are decided arithmetically and the third correctly falls through to
`insufficient`.

## Frontend-support API

A local frontend must never query these tables or re-derive the provenance
rules. Everything it needs is on the client:

```python
client.initialize_database()          # idempotent schema + health report
client.health()                       # database, schema, pgvector, models, counts


client.list_documents()               # -> list[DocumentSummary]
client.get_document(document_id)
client.remove_document(document_id, confirm_document_id=document_id)

client.get_audit_trace(audit_id)      # -> AuditTrace
client.get_evidence(evidence_id)      # -> EvidenceDetail
```

Ids may be passed as `int` or `str`; anything else raises `ValidationError`.

### Source identity

Three separate values, because one hash cannot honestly do three jobs:

| Value | What it is |
|---|---|
| `source_sha256` | the actual SHA-256 of the PDF bytes, verifiable with `sha256sum`. Null when no PDF was supplied — never an output fingerprint wearing the source's name. |
| extraction fingerprint | content hash of the manifest, `blocks.jsonl`, and every manifest-listed page's evidence artifacts *including* `page.png`, streamed in page order. |
| version fingerprint | the tagged combination of source hash, extraction fingerprint, embedding model, and dimensions. |

The extraction fingerprint hashes contents, not file sizes. A re-run that
improves a table reconstruction or redraws a page image without changing its
length must not look like no change at all — a same-size edit moves it, and a
stale directory the manifest no longer lists does not.

A better extraction of the same PDF therefore builds a *new version of the same
document*; the previous one is retired, not deleted. `identity_key` is
deliberately still derived from the pre-correction PDF token, so fixing the
public hash does not split every already-indexed PDF into a second document.

`source_uri` is where the report was published. It stays null unless a caller
supplies one; an output directory is not a provenance claim.

### Build states and health totals

A version is `building`, `ready`, `inactive`, or `failed`. When ingestion
raises, that building version is marked `failed` with a safe `failure_code` and
`failure_phase` — never `str(exc)` — and an older `ready` version is left
exactly as it was. Retrying reopens the same failed attempt and clears its
failure metadata first, so a live build is never described by last time's
verdict.

A killed process cannot write its own failure, so `health()` classifies it by
silence instead: still `building` with no progress for longer than
`CLAIM_EVIDENCE_BUILD_STALE_MINUTES` is reported as `documents_interrupted`.
That is a read — `health()` never mutates a row.

```text
documents_ready  documents_building  documents_failed
documents_interrupted  documents_inactive
```

`evidence_units`, `embeddings`, and `facts` count only rows belonging to a
`ready` version: that is the queryable index, and a total that grew when a build
broke would describe storage instead. `stored_evidence_units` keeps the
historical figure for diagnostics.

`schema_embedding_dimensions` and `configured_embedding_dimensions` are both
reported. `vector(N)` is templated in only when the table is first created, so
re-initializing an existing database at a different dimension would otherwise
leave the old column in place; a mismatch fails `db init` with
`IndexNotReadyError` before the first embedding write and makes `health()`
not-current while PostgreSQL stays reachable.

### Errors

`NotFoundError`, `ValidationError`, `DependencyUnavailableError`, and
`IndexNotReadyError` all derive from `ClaimEvidenceError`, so a caller maps a
type rather than parsing a message. Driver exceptions and Ollama response
bodies stay inside the package: `health()` reports a failing query by class,
because a driver message can carry the host and user.

### Re-indexing and removal

`ingest_document(..., force=True)` rebuilds an unchanged source. The current
version keeps serving queries throughout and is only retired once the
replacement passes its integrity checks; if the rebuild fails, the previous
version is still `ready`.

Documents are identified internally by the strongest identity available — the
source PDF's content hash, else `source_uri`, else the canonical output root —
so two unrelated reports whose directories happen to share a basename are two
documents, and removing one cannot remove the other. Registering two copies
under one explicit `source_uri` still makes them versions of one document. The
key is internal: no public result field exposes it.

Resuming an interrupted build reconciles the half-finished version against the
source rather than adding to it. Changed text is overwritten and its embedding
is discarded and recomputed; units, pages, facts, and fact-evidence links the
source no longer produces are deleted. Only the `building` version is ever
touched — a `ready` or `inactive` one is left exactly as it is.

`remove_document()` requires the id twice and deletes only index rows,
returning per-table counts. It never touches the source PDF, the
`document_extract` output directory, page images, or Ollama models —
re-ingestion is the whole recovery path.

### Rendering evidence

`EvidenceDetail` carries `page_image_path` (the registered `page.png`, resolved
from stored metadata — never from a caller-supplied path), `page_width`,
`page_height`, `artifact_paths`, and every `Region`. Regions keep their roles
separate: a table citation is descriptor + header + unit + value, not one box
around the lot.

```text
claim_text  descriptor  header  unit  value  supporting_context  visual_region  unknown
```

Each region's `bbox` is `[left, top, right, bottom]` with
`coordinate_space = "normalized_top_left"`, so multiplying by the page image's
pixel size is all a renderer has to do. The package deliberately draws nothing:

```python
detail = client.get_evidence(evidence_id)
page = Image.open(detail.page_image_path)
w, h = page.size
for region in detail.regions:
    left, top, right, bottom = region.bbox
    draw.rectangle((left * w, top * h, right * w, bottom * h), outline=colors[region.role])
```

`output_root` and `source_pdf` on `DocumentSummary` are server-side path
metadata. A frontend backend may use them after its own allowed-root
validation; they are not for an untrusted browser.

### Progress events

Both long operations accept an optional callback. Callers passing nothing are
unaffected.

```python
client.ingest_document(output_root, progress=events.append)
client.audit_claim(claim, progress=events.append)
```

Each `ProgressEvent` carries `operation`, `phase`, `status`
(`start`/`progress`/`completed`/`warning`/`failed`), a human-readable
`message`, `completed`/`total`/`percent`, `document_id`, `audit_id`,
`current_item`, `details`, and `timestamp`.

`completed`, `total`, and `percent` are null when the work genuinely cannot be
measured — an honest spinner beats an invented percentage. `percent` is
phase-local and never decreases within a phase. A phase with nothing to do
reports `0/0` at 100%, not a division error.

Phases, in order:

```text
ingest   validating_input building_evidence embedding_evidence extracting_facts
         building_indexes activating_version completed
audit    parsing_claim retrieving_graph retrieving_full_text retrieving_vectors
         fusing_candidates expanding_context verifying_visuals deciding_verdict
         persisting_trace completed
```

Totals are real: pages for evidence, embedding batches, claim-like passages for
fact extraction, integrity steps for index checks, and returned-versus-requested
candidates for each retrieval channel.

A callback that raises is dropped for the rest of the operation and the work
continues. A broken UI must never corrupt an index build.

#### Completion summaries

The terminal `completed` event carries counts the operation already computed —
no extra queries. Ingestion mirrors its `IngestReport`:

```text
document_version_id page_count evidence_count visual_evidence_count
embedding_count fact_count warning_count elapsed_seconds no_op
```

Audit reports its retrieval story:

```text
verdict citation_count graph_candidate_count full_text_candidate_count
vector_candidate_count fused_candidate_count expanded_candidate_count
visual_candidate_count visually_verified_count selected_evidence_count
elapsed_seconds
```

Channel counts are what each retriever *returned*; the persisted trace records
only the candidates that survived fusion's cut, so the event count is the
larger of the two. A channel that could not run at all — for example vectors
when embeddings are unavailable — omits its key rather than reporting zero,
because "ran and found nothing" and "never ran" are different facts.

#### Failures

A terminal `failed` event carries `failed_phase`, `error_code`, and `retryable`
in `details`, then the original exception is re-raised unchanged.

| `error_code` | `retryable` |
|---|---|
| `dependency_unavailable` | yes |
| `internal_error` | yes |
| `index_not_ready` | yes |
| `validation_error` | no |
| `not_found` | no |

Only the package's own typed errors are quoted back to the caller; their
messages are written for public consumption. Anything else — a driver error, a
model response body, an unexpected bug — is reported by category, because those
messages carry hosts, credentials, prompts, and source text.

### Retrieval trace

`get_audit_trace()` returns the parsed claim, every candidate with its
per-channel rank and score, the fused rank and score, context-expansion links,
visual re-verification status, whether it was selected, and a short reason.
This is retrieval metadata, not model reasoning — no prompts and no
chain-of-thought are stored.

## Tests

```bash
python tests/run_all.py
```

```bash
python -m compileall -q src
```

The deterministic suites need no GPU, no Ollama, and no Docker; Ollama is
replaced by a fake HTTP session. `tests/test_integration.py` needs the Compose
database and skips cleanly without it.

The live acceptance run is separate, because a first ingest of a 494-page
report is not a unit test:

```bash
python tests/acceptance_danone.py
```

It indexes the completed Danone output and audits the three reference claims
(supported at 40.2%, contradicted at 90%, and an "all carbon emissions" claim
that must stay `insufficient`). It skips cleanly when the output root,
database, or models are missing.

## Non-goals

No autonomous agent loop, no REST API or MCP server, no generic ontology or
community detection, no separate graph and vector stores, and one atomic claim
per audit call.

## Prototype boundaries

This is a version-1 prototype for one person on one machine. Everything below
is a deliberate limit, not an oversight, and each one has a written trigger for
when it stops applying.

| Boundary | What is supported | What is not |
|---|---|---|
| Language | English source documents and English claims | Any other language, and any cross-language matching |
| Network | Loopback only (`127.0.0.1`), one trusted local user, CSRF plus same-origin checks | Non-loopback binds, remote access, TLS, authentication, more than one user |
| Concurrency | One frontend process and one ingestion worker | A second worker or process, external concurrent CLI ingestion, cross-process locking, distributed jobs |
| Claims | One reporting entity per audit; free-form English text, split into atomic claims and grounded back to it before anything is audited | Auditing a post without human review of the split, more than 20 atomic claims at once, or establishing a comparison the selected documents do not make |
| Data | The index and its audits are disposable and rebuildable; `db reset-dev` drops and rebuilds a `_test`/`_dev` database | Migration between schema versions, preserved audit history, backup, restore, or archival |
| Scale | A handful of documents, rebuilt from their sources when anything changes | Large corpora, latency targets, throughput guarantees, or a certified hardware envelope |

### Deferred groups

These are recorded in `document_extract/PROTOTYPE_DECISIONS.md` and
`COMPLETE_GAP_REGISTER.md`, and none of them is implemented here.

- **DG-01 — until the schema is stable.** No migration ledger, no in-place
  upgrade, no preserved document ids or audits, no backup/restore. Triggered
  when a schema version is declared stable, or any database must survive an
  upgrade.
- **DG-02 — until concurrent or multi-process use.** No second worker, no
  cross-process advisory locking, no external CLI ingestion running alongside
  the frontend. Triggered when a second process may write to the same store.
- **DG-03 — until remote or multi-user deployment.** No authentication,
  authorization, TLS, remote bind, or user isolation. Triggered by any
  non-loopback bind, remote database, or second user.
- **DG-04 — until the corpus is large enough that rebuilding hurts.** No
  amendment/restatement policy, no multilingual support, no general knowledge
  graph or layout reasoning, no durable audit retention. Triggered past roughly
  twenty documents, a two-hour rebuild, or a non-English requirement.
- **DG-05 — until measured need.** No archival or compaction, no large-load
  fixture, no latency budget, no telemetry programme, no performance
  certification. Triggered by a measured failure, an operational target, or
  distribution to another machine.

### Acceptance

One command proves the whole workflow, and reports a blocked or missing check
as a failed run rather than a skip:

```powershell
gw_detector_v2\scripts\verify_prototype.ps1
```

It writes a timestamped JSON and Markdown report to
`gw_detector_v2\verification\prototype\` recording the repository revisions,
schema version and checksum, contract versions, model tags and digests, and
fixture version behind that result.
