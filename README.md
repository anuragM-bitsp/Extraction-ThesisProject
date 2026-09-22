# Scientific Paper Extraction System — Build Log

Building this incrementally, one step per session, so each piece is
understood (and defensible in a thesis defense / interview) before the next
is layered on top.

## Roadmap

- [x] **Step 1 — Target schema.** The domain JSON schema + provenance model
      every extraction strategy must conform to.
- [x] **Step 2 — Data model & storage.** PostgreSQL schema (via SQLAlchemy) +
      object storage layout, wired through a repository layer.
- [x] **Step 3 — Document ingestion.** PyMuPDF text/bbox extraction, conditional
      OCR (OCRmyPDF+Tesseract), optional GROBID enrichment (title/authors/
      abstract/references + best-effort section tagging).
- [x] **Step 4 — Chunking + retrieval.** Section-aware overlapping chunker,
      `EmbeddingModel` interface, brute-force cosine `Retriever` scoped per
      document version.
- [x] **Step 5 — Rule-based extractor.** Deterministic regex + unit
      normalization for temperature/time/concentration/volume/mass; first
      of the four strategies compared in the ablation study.
- [x] **Step 6 — NER extractor.** Entity recognition (precursor/solvent/
      material/characterization) plus nearest-span linking of quantities to
      the entity they describe.
- [x] **Step 7 — RAG + LLM extractor.** Per-field retrieval + schema-
      constrained generation; the strategy that handles novel names and
      multi-step context the other three can't.
- [x] **Step 8 — Hybrid fusion + conflict resolution.** Runs all three
      prior strategies and fuses their outputs field-by-field using
      per-field source-priority ranking, not concatenation.
- [x] **Step 9 — Normalization + entity linking.** Collapses "AgNO3" and
      "silver nitrate" into one canonical entity and links it to a CAS
      number — closes the gap Steps 6-8 all documented and tested for
      without fixing.
- [x] **Step 10 — Annotation system + gold dataset.** Storage for
      independent per-annotator submissions, inter-annotator agreement
      (Cohen's kappa), human adjudication, and paper-level train/test
      splitting.
- [x] **Step 11 — Evaluation framework.** Entity/relation/field-level
      precision/recall/F1 with unit-aware numeric comparison, corpus-level
      ablation aggregation, and a per-failure error taxonomy.
- [x] **Step 12 — Experiment runner + MLflow tracking.** Wires any
      extractor over any set of documents into one repeatable, optionally
      parallel run, scored against available gold and logged to a local,
      offline MLflow instance.
- [x] **Step 13 — API + orchestration.** FastAPI endpoints backed by Celery
      async task queue, tying every prior step into an actual service you
      can point at PDFs. **All 13 steps of the roadmap are now complete.**
- [x] **Streamlit frontend.** Thin UI over `PipelineService` / `get_extractor`
      (upload → ingest → extract → JSON/evidence). Not a new extraction engine.

## Step 1 — Target Schema

**Why this comes first:** the whole point of this project is to compare four
extraction strategies (rule-based, NER, RAG+LLM, hybrid) fairly. That's only
possible if all four are forced to produce the exact same structure. Defining
the schema *before* any extractor exists prevents each method from silently
drifting toward whatever representation is easiest for it.

### Files

```
schemas/
  extraction_schema.py   # SynthesisExtraction (the domain schema) + ExtractionResult
  provenance.py           # EvidenceSpan, CandidateFact, SourceType
tests/
  test_schema.py          # 7 tests covering validation, round-tripping,
                           # provenance linkage, and JSON-schema export
```

### Key design decisions (and why)

1. **Provenance is separate from the domain schema.** `SynthesisExtraction`
   (the "what") never carries confidence/source/evidence — that lives in
   `CandidateFact` (the "why should I trust this"). This means the same
   `SynthesisExtraction` type represents a gold annotation, a raw prediction,
   or a fused result, with no special-casing.

2. **Quantities keep (value, unit) as given, not pre-normalized.** `0.5 M`
   is *not* converted to `500 mM` at extraction time. Unit normalization is
   an evaluation-time concern (comparing predictions to gold) — normalizing
   early risks silently destroying information if a unit was misread
   upstream. This lands in Step 11.

3. **`ExtractionResult` is the actual return type of `Extractor.extract()`.**
   It bundles the prediction with one `CandidateFact` per populated field.
   Every extractor built in Steps 5–8 returns this same wrapper.

4. **Everything in `SynthesisExtraction` is `Optional`.** Extractors run
   field-by-field / task-by-task (LLD section 14 — "RAG shouldn't be one
   giant prompt"), so a partially-filled instance must stay valid.

5. **`model_json_schema()` works out of the box.** This is what Step 7
   (RAG+LLM extractor) will hand the model for schema-constrained
   generation — verified now so nothing breaks downstream.

### Running the tests

```bash
pip install pydantic pytest
pytest tests/ -v
```

All 7 tests pass as of this step.

## Step 2 — Data Model & Storage

**Why this comes second:** Step 1 defined *what* a prediction looks like.
Before any parser or extractor exists, we need somewhere for raw papers,
their parsed structure, and every intermediate artifact to actually live —
and for that storage to already implement the provenance chain
(`paper -> version -> page -> block`) that `EvidenceSpan` depends on.

### Files

```
storage/
  db_types.py      # GUID / JSONVariant / EmbeddingType — dialect-aware column types
  models.py        # PaperORM, DocumentVersionORM, DocumentBlockORM, ChunkORM
  object_store.py  # ObjectStore interface; LocalObjectStore (tested); S3ObjectStore (prod)
  read_models.py   # Pydantic mirrors of the ORM tables (from_attributes=True)
  repository.py    # PaperRepository — the only module that touches SQLAlchemy directly
tests/
  test_storage.py  # 9 tests: idempotent versioning, object-store round trip,
                    # provenance chain, embedding round trip, DB-level constraint
```

### Key design decisions (and why)

1. **One ORM schema, two databases.** `storage/db_types.py` defines column
   types (`GUID`, `JSONVariant`, `EmbeddingType`) that pick the Postgres-native
   implementation (`UUID`, `JSONB`) when talking to Postgres, and a portable
   fallback (`CHAR(32)`, `JSON`) otherwise. The test suite runs against
   `sqlite:///:memory:` — no Postgres server needed for CI or local dev — and
   the exact same `models.py` runs unchanged against real Postgres. No
   "test version" of the schema to keep in sync.

2. **Raw content never touches Postgres.** `PaperRepository.add_document_version`
   writes PDF bytes to an `ObjectStore` (`papers/{paper_id}/v{n}/original.pdf`)
   and only stores the content hash + metadata row in the DB (LLD section 3).

3. **Idempotent ingestion via content hash.** Re-uploading byte-identical
   content returns the existing `DocumentVersionRecord` instead of minting a
   new version — this is what stops a re-run pipeline from reprocessing a
   paper it's already seen (LLD sections 4 and 27).

4. **`UNIQUE(paper_id, version)` is enforced at the DB level, not just in
   application code** — tested directly by bypassing the repository and
   hitting the ORM.

5. **Embeddings are JSON-array-backed for now, not `pgvector.Vector` yet.**
   The vector dimension is a function of the embedding model, which isn't
   chosen until Step 4. Wiring pgvector now would mean hard-coding a
   dimension nothing else depends on. `EmbeddingType` is the single place
   that changes when Step 4 picks a model.

6. **Repository is the only SQLAlchemy-aware module.** Extractors, the
   future API layer, and tests all interact through `PaperRepository` and
   get back plain Pydantic (`storage.read_models`) objects — the DB
   implementation can change (e.g. swap the driver, add a read replica)
   without touching anything upstream.

### Running the tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

16 tests pass (7 from Step 1 + 9 new).

### Deploying against real Postgres (when infra is available)

```python
from sqlalchemy import create_engine
engine = create_engine("postgresql+psycopg2://user:pass@host/dbname")
# Base.metadata.create_all(engine) — or run this via Alembic migrations in practice
```

No other code changes — `GUID`/`JSONVariant` detect the `postgresql` dialect
automatically. For `S3ObjectStore`, point `endpoint_url` at your MinIO
instance if not using AWS directly.

## Step 3 — Document Ingestion

**Why this comes third:** Step 2 gave us somewhere to put a parsed
document; this step is what actually produces one. Every extractor built
in Steps 5-8 will read a `CanonicalDocument`, never a raw PDF — so getting
this representation right (and honest about its limitations) matters more
than any individual extractor downstream.

### Files

```
ingestion/
  canonical.py       # CanonicalDocument, CanonicalBlock — the pipeline's output type
  pymupdf_parser.py  # always-on text/bbox extraction (no external service)
  ocr.py             # needs_ocr() pure decision fn + run_ocr() ocrmypdf subprocess wrapper
  grobid_tei.py       # pure TEI-XML -> title/authors/abstract/sections/references parser
  grobid_client.py    # GrobidClient interface + HttpGrobidClient (real HTTP transport)
  pipeline.py         # DocumentIngestionPipeline — orchestrates all of the above
tests/
  test_ingestion.py   # 14 tests, see below
```

### Key design decisions (and why)

1. **GROBID is optional, not a hard dependency.** GROBID runs as its own
   service (normally a Docker container on :8070) and there's no live
   instance in this environment. `DocumentIngestionPipeline` accepts an
   optional `GrobidClient`; if it's `None`, or if `process_fulltext()`
   raises for *any* reason (down, timeout, malformed response), ingestion
   still succeeds — just without title/authors/abstract and without
   section labels on blocks. Tested explicitly
   (`test_pipeline_falls_back_gracefully_when_grobid_is_down`).

2. **TEI parsing is separated from the HTTP transport.** `grobid_tei.py`
   (`parse_tei`) has zero network code and is tested against a hand-written
   canned TEI-XML fixture. `grobid_client.py` is the thin, harder-to-test
   layer that gets bytes onto the wire — its test fakes `requests.post`
   rather than needing a live GROBID server. This mirrors the
   ObjectStore split from Step 2: isolate the part you can't test for real
   from the part you can.

3. **OCR decision logic is a pure function.** `needs_ocr(pages, chars)` has
   no I/O and is fully parametrized-tested. `run_ocr()` (the actual
   `ocrmypdf` subprocess call) is exercised end-to-end against a *real*,
   programmatically-generated image-only PDF — not mocked — so the test
   proves OCR genuinely recovers text, not just that a function was called.

4. **Section tagging from GROBID is explicitly a heuristic, not ground
   truth.** PyMuPDF and GROBID segment text independently and their block
   boundaries don't always align. `_assign_sections()` does substring
   matching on a text snippet and is documented as "good enough for
   retrieval filtering, not good enough to trust blindly for evaluation."
   Said out loud in code comments so it doesn't get silently relied on
   later as if it were exact.

5. **PyMuPDF never attempts section detection.** It sees layout, not
   semantics — asking it to guess "Introduction" vs "Experimental" would
   just be worse heuristics duplicating what GROBID already does properly.
   Table extraction (Camelot/pdfplumber, LLD section 10) is intentionally
   deferred — it only changes which blocks get `block_type=TABLE`; it
   doesn't change this step's contract.

### A real gotcha hit while building this (worth knowing about)

`ocrmypdf` (installed via `apt-get install ocrmypdf`) failed with
`ImportError: cannot import name 'PdfMatrix' from 'pikepdf'` because a
newer, pip-installed `pikepdf` (10.5.1) was shadowing the apt-provided one
(8.7.1) that ocrmypdf actually expects. Fixed with `pip uninstall pikepdf`
so Python falls back to the apt version. This is exactly the kind of
system-dependency friction worth documenting rather than papering over —
noted in `requirements.txt` for when this gets deployed elsewhere.

### Running the tests

```bash
pip install -r requirements.txt
apt-get install -y ocrmypdf tesseract-ocr   # system deps for OCR
pytest tests/ -v
```

30 tests pass (7 + 9 + 14). The OCR end-to-end test is skipped automatically
if `ocrmypdf` isn't on PATH (`@pytest.mark.skipif`), so the rest of the
suite still runs in environments without it installed.

### Running against a real GROBID instance

```bash
docker run -p 8070:8070 grobid/grobid:0.8.0
```

```python
from ingestion.grobid_client import HttpGrobidClient
from ingestion.pipeline import DocumentIngestionPipeline

pipeline = DocumentIngestionPipeline(grobid_client=HttpGrobidClient("http://localhost:8070"))
canonical_doc = pipeline.ingest(paper_id="P001", version=1, pdf_bytes=pdf_bytes)
```

## Step 4 — Chunking + Retrieval

**Why this comes fourth:** Step 3 gave us structured blocks per paper; this
step turns those blocks into what a RAG-style extractor (Step 7) actually
queries against — sized, embedded, searchable chunks. Getting chunk
boundaries and the retrieval interface right now means Step 7 is just
"write extraction-specific queries against `Retriever.search()`", not
"also invent chunking and embeddings while building an extractor."

### Files

```
retrieval/
  chunking.py    # chunk_blocks() — section-aware, overlapping chunker
  embeddings.py  # EmbeddingModel interface; HashingEmbedder (tested, offline);
                  # SentenceTransformerEmbedder (production BGE/E5, untested here)
  indexing.py    # index_document() — chunk -> embed -> persist, end to end
  retriever.py   # Retriever interface; CosineRetriever (brute-force, per version)
tests/
  test_retrieval.py   # 12 tests: chunking behavior, embedder properties,
                       # indexing + retrieval integration
```

### Key design decisions (and why)

1. **No real embedding model is reachable from this environment.** BGE/E5
   weights live on the Hugging Face Hub, which isn't in the sandbox's
   allow-listed domains. Rather than fake test results against a mocked
   model, `HashingEmbedder` is a real, deterministic, offline
   feature-hashing vectorizer — it's not semantically meaningful, but it's
   genuinely exercised by every retrieval test (e.g.
   `test_hashing_embedder_similar_text_scores_higher_than_unrelated`
   actually checks that shared-vocabulary text scores higher, not just that
   a function ran). `SentenceTransformerEmbedder` is the real production
   path; swapping it in changes zero lines elsewhere, since both satisfy
   `EmbeddingModel`.

2. **Chunks break at section boundaries before hitting the token target.**
   Directly serves LLD doc 2 section 10's retrieval strategy ("synthesis
   conditions are more likely in Experimental than Introduction") — that
   only helps if a chunk's `section` field is unambiguous, which requires
   never letting one chunk span two sections.

3. **Overlap carries the literal tail words forward, not a re-summarization.**
   A fact split across a chunk boundary ("...heated at 80 °C" | "for 2 h.")
   needs to appear intact in at least one chunk. Tested directly
   (`test_chunk_overlap_carries_tail_words_forward`) by asserting the exact
   word-for-word overlap, not just "some overlap exists."

4. **Retrieval is brute-force cosine over one document_version's chunks —
   not a pgvector ANN index yet.** At 50-100 papers this is genuinely the
   right call, not a placeholder: scoring every chunk in a paper is
   microseconds. The upgrade path is documented in `retriever.py` as a
   specific SQL query plus the one line in `storage/db_types.py`
   (`EmbeddingType` -> `pgvector.sqlalchemy.Vector(dim)`) that changes if
   corpus scale ever justifies it — deferred consistently with Step 2's own
   reasoning about not introducing infrastructure before it's needed.

5. **Reranking is deliberately NOT in this step.** LLD section 11 treats
   reranking as optional and downstream of retrieval. It depends on how
   Step 7's extractor phrases its per-field queries (LLD section 14 — "RAG
   shouldn't be one giant prompt"), so it belongs there, not baked into a
   generic `Retriever`.

### Running the tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

42 tests pass (16 storage/schema + 14 ingestion + 12 retrieval).

### Swapping in real embeddings + pgvector later

```python
from retrieval.embeddings import SentenceTransformerEmbedder
embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")  # needs HF Hub access
```

```python
# storage/db_types.py, once a model (and its dimension) is fixed:
from pgvector.sqlalchemy import Vector
# EmbeddingType -> Vector(dim) for the `embedding` column on ChunkORM
```

## Step 5 — Rule-Based Extractor

**Why this comes fifth:** it's the first of the four strategies the thesis
actually compares, and it's the one that needs the least new
infrastructure — no embeddings, no model weights, just `Extractor` (a new
shared interface) applied to the `CanonicalDocument` Step 3 already
produces. Building it first also forced the `Extractor` interface itself to
exist, which NER (Step 6), RAG+LLM (Step 7), and Hybrid (Step 8) all reuse
unchanged.

### Files

```
extractors/
  base.py              # Extractor(ABC) — the Strategy-pattern interface (LLD section 7)
  rule_extractor.py    # RuleExtractor — runs all rules, builds ExtractionResult
  rules/
    base.py            # ExtractionRule(ABC), RuleMatch
    units.py           # unit-normalization lookup tables, shared by every rule
    temperature.py      # TemperatureRule
    reaction_time.py    # ReactionTimeRule
    concentration.py    # ConcentrationRule
    volume.py           # VolumeRule
    mass.py             # MassRule
tests/
  test_rule_extractor.py   # 25 tests: per-rule correctness + full extractor behavior
```

### Key design decisions (and why)

1. **`Extractor` drops the `schema` parameter from the LLD's sketch.**
   `Extractor.extract(document, schema)` made sense before a schema was
   fixed; Step 1 already pinned every strategy to `SynthesisExtraction`, so
   threading a schema argument through every extractor risks one strategy
   silently using a different schema version than the others — exactly the
   apples-to-oranges comparison the whole project structure exists to
   prevent.

2. **Concentration/volume/mass matches are found but NOT written into
   `prediction.precursors`.** A bare "10 mM" is meaningless in the schema
   until you know *which precursor* it modifies, and that requires entity
   linking — NER identifying "silver nitrate" as an entity, then relating
   it to the nearby "10 mM" (LLD sections 16-19, Steps 6-9). Writing
   these into the schema anyway would mean silently guessing an
   association the rule engine has no basis for. They're still fully
   present in `ExtractionResult.provenance`, so nothing is lost — just not
   yet resolved. Tested explicitly
   (`test_rule_extractor_leaves_unlinked_quantities_out_of_prediction_but_in_provenance`).

3. **"First occurrence wins" for repeated fields, and it's called out as a
   simplification, not hidden.** If a paper mentions two temperatures,
   `RuleExtractor` alone has no way to know which is "the" synthesis
   temperature — proper multi-mention resolution needs the conflict
   resolver in the hybrid fusion layer (Step 8). Both mentions still appear
   in provenance; only the schema slot uses the first. Tested directly.

4. **Mass and concentration rules don't collide on compound units.**
   `"5 mg/mL"` is a concentration, not a 5 mg mass — `MassRule` uses a
   negative lookahead (`(?!\s*/)`) so it backs off whenever a unit is
   immediately followed by a `/`, leaving compound units to
   `ConcentrationRule`. Tested with both
   `test_mass_rule_does_not_misfire_on_compound_concentration_unit` and its
   positive counterpart on `ConcentrationRule`.

5. **Unit matching is case-insensitive by design, not oversight.** Exact
   case (`mM` vs `Mm`) is technically meaningful in chemistry notation, but
   PDF-extracted scientific text is inconsistent enough that requiring
   exact case would cause more missed matches than it prevents false ones.
   Documented as a known trade-off in `units.py`.

### A real bug caught by the tests (worth knowing about)

The concentration regex originally ended in `\b` after `wt%` — but `%` is
not a word character, so there's no word-boundary transition between `%`
and a following space, and the match silently failed. Fixed by replacing
the trailing `\b` with `(?![A-Za-z0-9])`, which asserts "not immediately
followed by an alphanumeric" instead of relying on `\b`'s word-character
semantics. This is exactly the kind of regex edge case unit tests exist to
catch — it would have silently under-counted `wt%` mentions in a real
corpus.

### Running the tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

67 tests pass (42 from before + 25 new).

## Step 6 — NER Extractor

**Why this comes sixth:** LLD doc 2 section 8 is explicit that NER alone
is incomplete — "it tells us what entities exist, but not necessarily how
those entities are related." Step 5 deliberately left concentration/mass
values unlinked to any precursor; this step is where that link finally gets
made, which is also why it's the natural second strategy to build rather
than jumping straight to the RAG+LLM extractor.

### Files

```
extractors/
  ner_extractor.py         # NerExtractor — the second strategy
  ner/
    base.py                # NerModel(ABC), Entity
    labels.py              # EntityLabel: PRECURSOR/SOLVENT/MATERIAL/CHARACTERIZATION
    gazetteer.py            # GazetteerNER — offline, dictionary-based, tested
    transformer_ner.py      # TransformerNER — production (HF Hub, untested here)
    linking.py               # find_nearest_match() — the entity-relation linking step
tests/
  test_ner_extractor.py    # 14 tests: gazetteer matching, linking, full extractor
```

### Key design decisions (and why)

1. **`GazetteerNER` is a real, tested `NerModel` — not a mock of one.**
   Same reasoning as `HashingEmbedder` (Step 4): a trained domain NER model
   needs weights this sandbox can't download, so instead of faking outputs,
   there's a genuine dictionary + word-boundary matcher that finds real
   entities in real text, tested against real edge cases (substring false
   positives, case sensitivity, repeated mentions). `TransformerNER` is the
   production path — same `NerModel` interface, zero downstream changes to
   swap it in.

2. **Concentration/mass linking reuses Step 5's rules directly**, rather
   than re-implementing number-finding. What's new in this extractor is the
   entity recognition and the *decision* to link a nearby number to an
   entity (`extractors/ner/linking.py`) — that decision is what makes this
   a distinct strategy from Step 5's, not a second regex implementation.

3. **Linking is nearest-span proximity, explicitly not a relation
   classifier.** Documented in `linking.py`'s docstring with the case where
   it would actually fail (two precursors, one ambiguous shared quantity).
   Tested that it correctly *declines* to link when nothing is within range
   (`test_ner_extractor_leaves_concentration_unlinked_when_no_precursor_nearby`)
   rather than always forcing a link.

4. **Synonymous names are NOT merged.** "AgNO3" and "silver nitrate" produce
   two separate `Precursor` entries even in the same sentence — the
   gazetteer has no notion that they're the same compound. This is called
   out as Step 9's job (entity normalization / linking to canonical IDs),
   not silently patched over here. Tested directly
   (`test_synonymous_precursor_names_are_not_merged_by_the_gazetteer`) so
   the limitation is documented behavior, not an accidental gap discovered
   later.

5. **Every CandidateFact from this extractor is tagged `source=NER`**, even
   the ones whose number was found by reusing `ConcentrationRule`'s regex
   internally. `source` records which extraction *strategy* produced a
   fact for the ablation study — not which sub-routine happened to fire —
   so the four-way comparison in Step 11 stays meaningful.

### Running the tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

81 tests pass (67 from before + 14 new).

### Swapping in a real fine-tuned model later

```python
from extractors.ner.transformer_ner import TransformerNER
from extractors.ner_extractor import NerExtractor

extractor = NerExtractor(model=TransformerNER("your-org/materials-ner-finetuned"))
```

## Step 7 — RAG + LLM Extractor

**Why this comes seventh:** rules (Step 5) and the gazetteer NER (Step 6)
both have a hard ceiling — a fixed regex vocabulary and a fixed dictionary,
respectively. This is the strategy built specifically to go past that
ceiling: novel chemical names never seen before, multi-step procedures that
need paragraph-level context, and properties phrased in ways no dictionary
would anticipate. It's also the first strategy to actually use Step 4's
retrieval machinery for something.

### Files

```
extractors/
  rag_llm_extractor.py     # RagLlmExtractor — the third strategy
  rag/
    llm_client.py           # LLMClient(ABC), call_with_retry(), AnthropicLLMClient
                              # (production, untested), FakeLLMClient (scriptable, tested)
    tasks.py                 # FieldTask definitions — one retrieval query +
                              # response schema + apply() per schema field
    prompting.py              # build_prompt() — pure function, no LLM needed to test it
tests/
  test_rag_llm_extractor.py  # 13 tests: retry logic, prompting, full extractor
```

### Key design decisions (and why)

1. **`api.anthropic.com` is network-reachable from this sandbox, but there's
   no API key configured** — so `AnthropicLLMClient` is written against the
   real SDK (tool-use forces schema-constrained output) but still can't be
   exercised by the test suite. `FakeLLMClient` is a genuinely useful test
   double: it validates dicts against the real Pydantic response models
   (so a malformed scripted response fails exactly the way a real malformed
   LLM response would) and records every call for assertion. Swapping in
   `AnthropicLLMClient` wherever a key exists changes zero other code.

2. **One task per schema field, not one giant prompt** — directly
   implementing LLD section 14. Each `FieldTask` has its own retrieval
   query, so `precursors` and `properties` retrieve different chunks even
   from the same paper. Tested explicitly with a query chosen to overlap
   the Experimental block's vocabulary (see the test's own docstring for
   why — `HashingEmbedder` has no semantic understanding, so this test
   checks the retrieval/ranking *wiring*, not whether a bag-of-words hash
   understands scientific writing).

3. **LLM evidence is chunk-level, not a character span.** Rule/NER evidence
   points at an exact substring; generated text isn't naturally anchored to
   one. `EvidenceSpan.char_start/char_end` are simply `None` for LLM-sourced
   facts — stated as a real trade-off, not smoothed over, and tested
   directly (`test_rag_llm_extractor_evidence_is_chunk_level`).

4. **A permanently-failing task doesn't sink the whole extraction.** If the
   LLM call for `synthesis_method` exhausts its retries, that field is left
   `None` and the extractor moves on to `steps`, `precursors`, and
   `properties` independently — per LLD section 26, a permanent failure
   shouldn't be retried forever, and a partial result is still useful.
   Tested by scripting three consecutive transport errors for one task.

5. **Chunking/embedding happens fresh inside `extract()`, not against
   Step 2's persisted chunk store.** Keeps `Extractor.extract(document)`
   identical across all four strategies — no extractor needs a database
   connection just to run. A real pipeline would reuse Step 4's already-
   indexed chunks rather than re-embedding per call; that wiring is
   Step 12's job (experiment runner), not the extractor interface's.

### A real bug this step's tests caught in Step 4's code

Building the retrieval-targeting test above surfaced a genuine bug in
Step 4's chunker: `chunk_blocks()` applied its overlap logic on *every*
chunk break, including section-boundary breaks — so a short Introduction
block got carried whole into the front of the next (Experimental-tagged)
chunk, then again into the one after that, snowballing across the whole
document. That directly undermines the reason section-tagged chunks exist
(LLD doc 2 section 10 — targeting retrieval by section). Fixed so overlap
only applies when a chunk splits due to length, never when it splits
because the section changed. A regression test now lives in
`tests/test_retrieval.py`
(`test_section_boundary_does_not_leak_overlap_text_across_sections`). This
is exactly why building extractors against real infrastructure, rather than
mocking it away, is worth the extra effort — the bug was invisible until
something actually consumed the chunker's output end-to-end.

### Running the tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

95 tests pass (81 from before + 1 chunking regression test + 13 new).

### Running against the real Anthropic API

```python
from extractors.rag.llm_client import AnthropicLLMClient
from extractors.rag_llm_extractor import RagLlmExtractor
from retrieval.embeddings import SentenceTransformerEmbedder

extractor = RagLlmExtractor(
    embedder=SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5"),
    llm_client=AnthropicLLMClient(model="claude-sonnet-5", api_key="sk-ant-..."),
)
```

## Step 8 — Hybrid Fusion + Conflict Resolution

**Why this comes eighth:** it's the fourth strategy the thesis compares,
and the only one that needs the other three to already exist — it runs
Rule, NER, and RAG+LLM against the same document and fuses their outputs,
rather than being a fourth independent extraction method.

### Files

```
extractors/
  hybrid_extractor.py          # HybridExtractor — the fourth strategy
  fusion/
    conflict_resolver.py        # ConflictResolver — per-field source-priority ranking
tests/
  test_hybrid_extractor.py     # 13 tests: resolver unit tests + full fusion pipeline
```

### Key design decisions (and why)

1. **Priority ranking beats raw confidence, on purpose.** `ConflictResolver`
   ranks candidates by (source priority, then confidence) — a low-
   confidence RULE match for a numeric field still beats a high-confidence
   LLM guess, because the entire premise of hybrid fusion (LLD section 18)
   is trusting *strategies* differently by field type, not just trusting
   whichever source sounds most sure of itself. Tested directly
   (`test_resolver_prefers_higher_priority_source_over_higher_confidence`).

2. **The fused CandidateFact keeps its original `source`, not `HYBRID`.**
   When RULE wins a conflict, the provenance entry still says `RULE` — that
   traceability ("which strategy actually produced the winning value") is
   the entire point of carrying provenance through fusion (LLD section 17).
   `HYBRID` as a `SourceType` exists for facts genuinely synthesized *by*
   the fusion layer itself (none currently are — every winning value traces
   to one of the three underlying strategies).

3. **A real, demonstrable payoff, not just plumbing**: NER's concentration
   for "Silver nitrate" is genuinely regex-derived (10 mM, actually present
   in the text); a test scripts the LLM to hallucinate a conflicting value
   (12 mM) for the same precursor, and fusion correctly keeps NER's real
   value. This is the concrete case LLD section 18 keeps gesturing at —
   `test_hybrid_resolves_concentration_conflict_in_favor_of_ner_over_llm`
   makes it a checked fact instead of an assertion in prose.

4. **Precursor merging is exact (case-insensitive) name matching — still no
   synonym canonicalization.** "AgNO3" (from a scripted LLM response) and
   "Silver nitrate" (from NER) remain two separate `Precursor` entries after
   fusion, carried forward unchanged from Steps 6-7's own limitation.
   Tested explicitly (`test_hybrid_does_not_merge_synonymous_precursor_names`)
   so this is documented, expected behavior at every step until Step 9
   actually fixes it — not a bug quietly waiting to be noticed.

5. **List fields with only one real source today (steps, properties) use
   simple passthrough, not a fusion algorithm that doesn't have anything to
   fuse yet.** Building speculative merge logic for a conflict that can't
   currently occur (no second strategy populates `steps`) would be
   complexity with no test that could ever exercise the interesting case
   honestly. The passthrough is structured so adding a second source later
   doesn't require restructuring the extractor.

### Running the tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

108 tests pass (95 from before + 13 new).

### Why HybridExtractor requires a real RagLlmExtractor

Unlike `RuleExtractor()` and `NerExtractor()`, `HybridExtractor` can't
default-construct a `RagLlmExtractor` — that needs an `EmbeddingModel` and
an `LLMClient`, both of which are deployment choices (which embedding
model, which LLM, real or fake). Passing a misconfigured `HybridExtractor`
silently through would mean discovering the missing piece deep inside
`extract()`; failing fast in `__init__` is exactly what the schema-
validation instincts from Step 1 would suggest here too.

## Step 9 — Normalization + Entity Linking

**Why this comes ninth:** it's a post-processing transformation applied to
the *output* of any extractor, not a fifth extraction strategy — which is
exactly why it comes after all four strategies exist rather than being
folded into one of them. It closes the one gap every step since Step 6 has
deliberately documented and tested for instead of quietly working around:
"AgNO3" and "silver nitrate" finally become one entity.

### Note on how this step got built

This step's code and tests (`normalization/canonical_dictionary.py`,
`linking.py`, `normalizer.py`, `pipeline.py`, and
`tests/test_normalization.py`) were already present in the workspace,
essentially complete, when this step started — leftover from an earlier
pass at the same design. Rather than discard and rebuild, they were
reviewed line-by-line, run against the full suite, and kept: the design
(interface + offline implementation + untested production stub, matching
every prior step's pattern) and the reasoning in the docstrings held up.
One real gap was found in review — see below — and fixed. Two other
unrelated stray files (`extractors/llm/`, `extractors/hybrid/`, an
alternate take on Steps 7-8) were found alongside it, confirmed unused by
anything, and removed, the same way Step 6's stray `entities.py`/
`ner_model.py` were.

### Files

```
normalization/
  canonical_dictionary.py   # CanonicalEntity + CANONICAL_ENTITIES — the offline ontology
  normalizer.py             # EntityNormalizer(ABC), DictionaryNormalizer
  linking.py                # OntologyLinker(ABC), LocalCasLinker (tested),
                              # PubChemLinker (production, untested — no network here)
  pipeline.py                # normalize_extraction() — applies normalization to any ExtractionResult
tests/
  test_normalization.py     # 17 tests: dictionary coverage, normalizer, linker, full pipeline
```

### Key design decisions (and why)

1. **Normalization is decoupled from `HybridExtractor` entirely.**
   `normalize_extraction()` takes any `ExtractionResult` — NER's raw output
   has the exact same "AgNO3 vs silver nitrate" duplication problem on its
   own, with no fusion involved. Tying normalization to the hybrid strategy
   specifically would mean the other three strategies' results couldn't be
   normalized before evaluation, which would bias the four-way comparison
   in Step 11. Tested directly by normalizing a bare `NerExtractor` result,
   not just a `HybridExtractor` one.

2. **Two-tier matching: exact alias, then embedding-similarity fallback.**
   Exact match is instant and confidence 1.0. For names close to but not
   exactly a known alias (e.g. "silver nitrate solution"), it falls back to
   cosine similarity over Step 4's `EmbeddingModel` — inheriting
   `HashingEmbedder`'s lexical-only limitation honestly (it catches surface
   variation, not true synonyms sharing no tokens). The returned
   `NormalizedEntity.confidence` reflects which tier matched
   (`match_method`), so a fuzzy match is never indistinguishable from an
   exact one downstream.

3. **Merging backfills rather than picking a single winner.** If NER's
   "AgNO3" mention had no linked concentration but the LLM's "silver
   nitrate" mention did, the merged entry keeps the concentration rather
   than discarding it because it came from the "losing" surface form. This
   is a different (simpler) policy than Step 8's `ConflictResolver` —
   appropriate because normalization-triggered merges are usually
   complementary information about the same entity, not competing
   measurements of the same field the way Step 8's cross-strategy
   disagreements are.

4. **`OntologyLinker` is a real, working interface today** — not just a
   placeholder for later. `LocalCasLinker` looks up the CAS number already
   in `CANONICAL_ENTITIES` (genuinely functional, network-free, limited to
   ~20 compounds). `PubChemLinker` queries the real PubChem PUG REST API by
   name — written correctly, untested here since
   `pubchem.ncbi.nlm.nih.gov` isn't in this sandbox's network allow-list.
   Swapping one for the other changes zero calling code.

5. **A coverage gap found during review, not accidentally shipped.**
   The dictionary's docstring claimed to cover every gazetteer term, but a
   direct comparison showed six solvents (acetone, DMF, DMSO, toluene,
   chloroform, isopropanol) and three nanoparticle materials were missing —
   meaning the gazetteer could recognize an entity that normalization would
   then silently fail to canonicalize. Fixed by extending the dictionary,
   and a permanent regression test now enforces the parity claim
   (`test_canonical_dictionary_covers_every_chemical_entity_the_gazetteer_recognizes`)
   so the two lists can't silently drift apart again as either grows.

### Running the tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

125 tests pass (108 from before + 17 new).

### Applying normalization to any extractor's output

```python
from normalization.normalizer import DictionaryNormalizer
from normalization.linking import LocalCasLinker
from normalization.pipeline import normalize_extraction

normalized_result = normalize_extraction(
    raw_result,                    # output of Rule/NER/RAG+LLM/Hybrid — any of them
    DictionaryNormalizer(),
    linker=LocalCasLinker(),       # optional — omit to skip external-ID linking
)
```

## Step 10 — Annotation System + Gold Dataset

**Why this comes tenth:** every extractor built in Steps 5-9 needs
something to be measured against, and that reference has to come from
humans reading the source papers directly — never from correcting model
output, which would bake whichever extractor's biases into the "ground
truth" it's supposed to be independent of (LLD doc 2 section 20). This step
builds that independent pipeline: two annotators label a paper from
scratch, their agreement is measured, disagreements get a human
adjudication decision, and the result becomes gold data in the exact same
schema every extractor already produces.

### Files

```
storage/
  models.py       # + AnnotationORM (per-annotator payload), GoldAnnotationORM (final gold)
  repository.py    # + submit_annotation/get_annotations, submit_gold/get_gold
annotation/
  paths.py          # flatten_payload/unflatten_payload — nested dict <-> field-path dict
  agreement.py       # cohens_kappa(), compute_agreement() — value + presence agreement
  adjudication.py     # adjudicate(), build_gold_result() — human-resolved merge -> gold ExtractionResult
  dataset.py           # paper_level_split(), export_gold_jsonl()
tests/
  test_annotation.py  # 18 tests: paths, kappa, agreement, adjudication, full pipeline
  test_storage.py     # +5 tests: annotation/gold repository methods
```

### Note on how this step got built

Reaching for this step's storage schema surfaced an existing, substantially
complete `AnnotationORM`/`GoldAnnotationORM` design already sitting in
`storage/models.py` — a third instance this session of finding a prior,
unfinished pass at a step already in the workspace (after Step 6's stray
files and Step 9's near-complete normalization package). It used a
**whole-payload** design (each annotator submits a full
`SynthesisExtraction.model_dump()` per version, not field-level rows the
way `CandidateFact` does) — genuinely a better fit than the field-level
schema drafted first here, because it sidesteps needing per-field-type
merge logic entirely: two full payloads can be flattened, compared, and
merged generically, then handed to Pydantic for validation. That
flatten/merge/validate approach (`annotation/paths.py`) is what got built
around it.

### Key design decisions (and why)

1. **Agreement has two distinct layers, not one number.** `agreement_rate`
   asks "when both annotators marked a field, did they record the same
   value?" `presence_kappa` asks "did they even notice the same
   information existed?" — a genuinely different signal (a property
   mentioned only in the Discussion section might get flagged by one
   annotator and reasonably skipped by the other as out-of-scope, which
   isn't the same kind of disagreement as transcribing "80" vs "85").

2. **Agreement/adjudication comparison is exact-match, not unit-aware.**
   This is a deliberate asymmetry with Step 11: comparing *predictions* to
   gold needs unit-tolerant comparison (0.5 M == 500 mM), but comparing
   *annotators* to each other should stay maximally strict — a value
   mismatch between two humans is a signal to fix the annotation
   guidelines (LLD doc 2 section 21), not something to smooth over before
   it's even been noticed.

3. **`adjudicate()` refuses to guess.** Every disagreed field must appear
   in an explicit `overrides` dict or the whole call raises
   `UnresolvedDisagreement` listing every such field at once. There's no
   automatic tie-breaker (no "trust annotator A") — unlike Step 8's
   `ConflictResolver`, where ranking extraction strategies by known
   strengths is a defensible prior, there's no equivalent prior for which
   human annotator to trust by default.

4. **Gold data is schema-identical to every extractor's output.**
   `build_gold_result()` returns a real `ExtractionResult` with
   `extractor=ExtractorName.HUMAN`, validated through the same
   `SynthesisExtraction` model as Rule/NER/RAG+LLM/Hybrid, with one
   `CandidateFact(source=HUMAN)` per resolved field. Step 11's evaluation
   can therefore compare any extractor's `ExtractionResult` to gold's
   `ExtractionResult` with the exact same comparison code — no gold-shaped
   special case.

5. **Splitting is paper-level, never chunk- or field-level.** Directly
   implements LLD doc 2 section 21's data-leakage warning: with only
   50-100 papers, letting two facts from the same paper land on opposite
   sides of a train/test split would be a real, not theoretical, leakage
   risk. `paper_level_split()` shuffles and splits whole `paper_id`s only.

### Running the tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

148 tests pass (125 from before + 5 storage + 18 new).

### Running the annotation-to-gold flow

```python
from annotation.agreement import compute_agreement
from annotation.adjudication import build_gold_result

report = compute_agreement(payload_a, payload_b)
print(report.disagreements)  # exactly which fields need a decision

gold_result = build_gold_result(
    paper_id="P001", version=1,
    payload_a=payload_a, payload_b=payload_b,
    adjudicated_by="lead_annotator",
    overrides={"temperature.value": 80.0},  # only needed for disagreed fields
)
repo.submit_gold(version.version_id, gold_result.prediction.model_dump(),
                  adjudicated_by="lead_annotator", source_annotation_ids=[...])
```

## Step 11 — Evaluation Framework

**Why this comes eleventh:** every extractor (Steps 5-8) and the gold
dataset (Step 10) now exist in the exact same schema — this is the step
that finally uses that comparability to answer "which one actually
performs best, and where." Nothing before this step could produce a single
number; everything after this step (the experiment runner in Step 12)
depends on being able to.

### Files

```
evaluation/
  units.py           # to_canonical(), quantities_equal() — the ONLY unit-conversion
                       # code in the entire project (deliberately deferred until now)
  metrics.py           # PRF1 — precision/recall/F1 from raw counts, shared by every level
  comparators.py        # compare_scalar_fields(), match_precursors(), entity/relation/field PRF1
  evaluator.py            # evaluate() — one prediction + one gold -> EvaluationReport
  ablation.py               # aggregate_reports(), ablation_table() — corpus-level comparison
  error_analysis.py          # analyze_errors() — per-failure taxonomy, not just aggregate scores
tests/
  test_evaluation.py        # 24 tests across all six modules
```

### Key design decisions (and why)

1. **Unit conversion happens HERE, and only here, in the entire project.**
   Steps 1, 8, and 9 all kept `Quantity` un-normalized specifically to avoid
   destroying information upstream of a final comparison. This is that
   final comparison — there's no "upstream" left to protect, so `0.5 M` and
   `500 mM` are correctly treated as equal (`quantities_equal`), and `80 C`
   vs `353.15 K` too (via a shared Kelvin/base-unit conversion table).

2. **"Wrong value" counts as both a false positive and a false negative** —
   standard span-based NER evaluation convention, applied consistently to
   scalar fields, entities, and relations alike. A prediction that
   confidently asserts the wrong solvent doesn't get partial credit for
   "at least it tried"; it's scored exactly as badly as getting it and
   missing gold's actual value would each be scored alone.

3. **Micro-averaging for ablation, not macro.** `aggregate_reports` sums
   tp/fp/fn across every document for an extractor before computing
   precision/recall/F1, rather than averaging each document's own F1. This
   sidesteps a real distortion: a document with zero gold precursors would
   have undefined or zero recall under macro-averaging even though there
   was nothing to find, artificially dragging down the corpus number.

4. **Precursor matching depends on Step 9 having run — stated as a
   prerequisite, not hidden as a footnote.** `match_precursors` matches by
   `canonical_id` when present, falling back to exact lowercased name. An
   un-normalized "AgNO3" prediction scored against a gold "Silver nitrate"
   entry counts as one missed entity AND one spurious entity — a real way
   to *understate* an extractor's true performance if normalization was
   skipped. Tested directly
   (`test_match_precursors_without_normalization_treats_synonyms_as_unmatched`)
   so this dependency is a documented fact about how to get a fair score,
   not a surprise buried in a metrics table later.

5. **Error analysis is scoped honestly to what an ExtractionResult can
   express.** LLD section 26 also names PDF-parsing and retrieval failures
   as error categories; those live upstream (Step 3's ingestion, Step 4/7's
   retrieval) and aren't reconstructable from a prediction/gold pair alone.
   `error_analysis.py` only claims the categories it can actually detect
   from that pair — missing/spurious/wrong fields, missing/spurious
   entities, wrong relations — rather than gesturing at a taxonomy it can't
   fully back with code.

### Running the tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

172 tests pass (148 from before + 24 new).

### Running the four-way ablation on real results

```python
from evaluation.evaluator import evaluate
from evaluation.ablation import ablation_table

reports = []
for paper_id, gold_result in gold_by_paper.items():
    for extractor_name, extractor in [("RULE", rule_extractor), ("NER", ner_extractor), ...]:
        prediction = extractor.extract(canonical_documents[paper_id])
        reports.append(evaluate(prediction, gold_result))

for row in ablation_table(reports):
    print(row)  # {'extractor': 'HYBRID', 'level': 'relation', 'precision': 0.91, 'recall': 0.87, 'f1': 0.89, ...}
```

## Step 12 — Experiment Runner + MLflow Tracking

**Why this comes twelfth:** Step 11 gave us a way to score one prediction
against one gold record; this step is what runs that across many papers,
many extractors, and many configurations, repeatably — and logs enough
about each run that "Hybrid-v3 achieved F1 = 0.87 on Gold-v2" (LLD section
23) is a claim someone else could actually go verify, not just a number
someone remembers.

### Files

```
experiment/
  config.py       # ExperimentConfig — everything about a run worth logging for reproducibility
  factory.py       # build_extractor() — config -> real Extractor, with embedder/llm_client injected
  runner.py         # ExperimentRunner — runs one extractor over many documents, optionally parallel
  tracking.py        # MlflowTracker — logs params/metrics/per-paper reports to local MLflow
tests/
  test_experiment.py  # 10 tests: config, factory, runner (incl. parallel), real MLflow persistence
```

### Key design decisions (and why)

1. **`build_extractor` takes `embedder`/`llm_client` as explicit arguments,
   not config strings it resolves itself.** `ExperimentConfig` stores
   `embedding_model_name`/`llm_model_name` purely for logging — turning
   those strings into a live `SentenceTransformerEmbedder` or
   `AnthropicLLMClient` (or a `HashingEmbedder`/`FakeLLMClient` for a test
   run) is a deployment decision the caller already has to make. Baking a
   provider registry into the factory would make it responsible for every
   embedding/LLM provider that might ever exist.

2. **Parallelism is a `max_workers` argument, not a different execution
   model.** `ExperimentRunner.run(docs, max_workers=4)` uses a thread pool;
   `max_workers=1` runs sequentially. Threads specifically, not processes —
   the real bottleneck for `rag_llm`/`hybrid` runs is I/O-bound LLM API
   calls, exactly where threads help despite the GIL. This is LLD section
   24's "each paper independently processable" claim made literal: nothing
   about a paper's extraction logic changes based on how many other papers
   are running at the same time. Tested by asserting sequential and
   parallel runs produce identical results, not just that parallel doesn't
   crash.

3. **The runner scores only papers that have gold, and says so.**
   `ExperimentResult.scored_paper_count` can be less than `paper_count` —
   annotation (Step 10) doesn't have to be complete before an experiment
   can run, it just limits how many papers contribute to the reported
   metrics. An empty `ablation` table (zero gold available) is a valid,
   tested result, not an error.

4. **MLflow runs against a local sqlite file, not a server.** No network
   access needed, genuinely tested in this sandbox (unlike GROBID, the HF
   Hub, or PubChem elsewhere in this project, which all need real external
   services this environment can't reach). Pointing `tracking_uri` at a
   real MLflow server later is a one-line change with nothing else
   affected.

5. **A real gotcha found and documented while building this**: MLflow's
   plain filesystem backend (`file:./mlruns` — what most tutorials show
   first) is now in maintenance mode and raises on current MLflow versions
   unless an extra environment variable is set. The sqlite backend
   (`sqlite:///mlflow.db`) is what this module actually uses, noted
   directly in `tracking.py`'s docstring so nobody hits that error blind.

### Running the tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

182 tests pass (172 from before + 10 new).

### Running a real experiment

```python
from experiment.config import ExperimentConfig
from experiment.factory import build_extractor
from experiment.runner import ExperimentRunner
from experiment.tracking import MlflowTracker

config = ExperimentConfig(name="hybrid-v1", extractor="hybrid", top_k=4, gold_dataset_version="gold-v2")
extractor = build_extractor(config, embedder=my_embedder, llm_client=my_llm_client)
runner = ExperimentRunner(config=config, extractor=extractor, gold_by_paper=gold_by_paper)

result = runner.run(canonical_documents, max_workers=4)
print(result.ablation)  # per-(extractor, level) precision/recall/F1

MlflowTracker().log_experiment_result(result)  # -> ./mlflow.db, inspectable via `mlflow ui --backend-store-uri sqlite:///mlflow.db`
```

## Step 13 — API + Orchestration

**Why this comes last:** it's not a new capability so much as a delivery
mechanism for everything built in Steps 1-12 — an HTTP surface plus an
async job queue over ingestion, extraction, and annotation, so the project
can actually be pointed at a folder of PDFs instead of driven from a
Python REPL.

### Files

```
orchestration/
  api.py                  # FastAPI app: POST /papers, /process, /extract; GET /status, /results; POST /annotations
  service.py                # PipelineService — the shared business logic BOTH the API and Celery tasks call
  tasks.py                    # Celery tasks: async wrappers around PipelineService, with retry/DLQ
  celery_app.py                 # Celery app config — eager mode by default (no broker in this sandbox)
  extractor_registry.py           # get_extractor(name) — string -> real Extractor, for HTTP/task boundaries
  db.py                              # env-var-driven engine/session/object-store construction
tests/
  test_orchestration.py      # 16 tests: PipelineService, Celery tasks (real retry/DLQ), FastAPI TestClient
```

### Key design decisions (and why)

1. **`PipelineService` is the one place business logic lives.** Both
   `orchestration/api.py`'s routes and `orchestration/tasks.py`'s Celery
   tasks call `PipelineService`, never each other — the same "thin
   transport, plain-Python logic underneath" split Step 12's
   `ExperimentRunner` established. It's fully testable with the same
   in-memory-SQLite-plus-`LocalObjectStore` fixtures every storage test in
   this project already uses, no HTTP client or running broker required.

2. **Celery runs in eager mode by default** (`CELERY_TASK_ALWAYS_EAGER`) —
   `.delay()` executes a task synchronously in-process rather than
   dispatching to a real worker, since there's no Redis broker in this
   sandbox. One environment variable switches to a real broker with zero
   task-body changes; this is the same "each paper is independently
   processable" property Step 12 expressed as a thread pool, expressed here
   as a task queue instead (LLD section 24).

3. **A real concurrency bug, found and fixed, not just anticipated.**
   Building the FastAPI tests surfaced a genuine ordering bug: a route
   created a job row, then immediately called `.delay()` — which, in eager
   mode, runs the task **synchronously, in the same request**, before the
   route's own database session had committed. The task opens its own
   session (deliberately, to match what a separate worker process would
   have to do) and couldn't see the job it was supposed to update, so
   `mark_job_failed` failed with "no job {id}". Fixed by committing right
   after job creation and before enqueueing — a change that would have
   silently passed against a real broker, where message-delivery latency
   is enough to hide the race, and only failed here because eager mode
   removes that latency. Documented directly in `api.py` at the fix site so
   it isn't rediscovered as a mystery later.

4. **Idempotent extraction, enforced at the database level, not just
   assumed.** `PipelineService.run_extraction` checks for an existing
   result with the same `(version, extractor, extractor_version,
   schema_version)` before running anything — LLD section 27's exact
   wording, backed by the `UNIQUE` constraint Step 2's pattern already
   established for `document_versions`. Tested by actually calling
   extraction twice and asserting only one row exists, not just checking
   the function doesn't raise.

5. **Retry/DLQ is tested against real Celery semantics, not simulated
   from scratch.** An unparseable "PDF" triggers `self.retry()`, which — in
   eager mode — raises `celery.exceptions.Retry` synchronously back to the
   caller rather than being caught and rescheduled by a worker that doesn't
   exist here. The test plays the missing scheduler's part with a small
   retry loop, then asserts the job actually reaches `DEAD_LETTER` after
   exactly `max_retries` attempts — real Celery retry behavior, not a
   mocked stand-in for it.

6. **Security posture matches LLD section 29 at the boundary that
   matters.** Content-type/magic-byte checks and a size limit are enforced
   in `api.py` before anything touches upload bytes. The deeper concern LLD
   section 29 raises — a paper's text containing something that reads like
   an instruction to an LLM — is handled architecturally back in Step 7
   (`extractors/rag/prompting.py` puts paper content only in the
   "excerpts" section of the user prompt, never the system prompt), not
   re-litigated here as a second control.

### Running the tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

203 tests pass (187 from before + 16 new).

### Running the API for real

```bash
export DATABASE_URL="sqlite:///./sci_extract.db"
export OBJECT_STORE_ROOT="./object_store"
uvicorn orchestration.api:app --reload
```

```bash
curl -F "file=@paper.pdf" -F "title=My Paper" http://localhost:8000/papers
curl -X POST http://localhost:8000/papers/<paper_id>/process
curl -X POST http://localhost:8000/papers/<paper_id>/extract -H "Content-Type: application/json" -d '{"extractor": "rule"}'
curl http://localhost:8000/papers/<paper_id>/status
curl http://localhost:8000/papers/<paper_id>/results
```

To actually distribute work across workers instead of running eagerly:

```bash
export CELERY_TASK_ALWAYS_EAGER=false
export CELERY_BROKER_URL="redis://localhost:6379/0"
celery -A orchestration.celery_app.celery_app worker --loglevel=info
```

## Streamlit frontend

A thin UI over the same `PipelineService` + `get_extractor` path the API uses. It does **not** reimplement parsing, regex, NER, RAG, or fusion.

```bash
pip install -r requirements.txt
streamlit run frontend/app.py
```

### Smoke-test procedure (real PDF + real extractors)

1. Upload a real scientific PDF (not a generated demo file).
2. Click **Process Documents**. Status must become `Ready` or show the real ingestion error.
3. Select **Rule-based** → **Run Extraction**. Inspect Structured JSON (an `ExtractionResult`) and Evidence.
4. Select **NER** → **Run Extraction**. Compare with the rule result. The UI should show `NER implementation: GazetteerNER` / `Mode: Offline` unless you wired `TransformerNER` yourself.
5. In the sidebar, enter an Anthropic API key → **Apply**. **RAG + LLM** and **Hybrid** should switch from 🔒 to ✓ Available.
6. Run **RAG + LLM**, then **Hybrid**. Inspect evidence. RAG/Hybrid still need a loadable `SentenceTransformerEmbedder` model (Hugging Face) in addition to the key; a missing model surfaces as a real `ExtractorUnavailableError`, not a fake JSON payload.

### What was tested how

| Claim | Meaning |
| --- | --- |
| **Tested with fake/test doubles** | `tests/test_frontend.py` uses PyMuPDF-generated PDFs (same fixture style as `tests/test_orchestration.py`). Hybrid construction is exercised by substituting `HashingEmbedder` + `FakeLLMClient` for `SentenceTransformerEmbedder` + `AnthropicLLMClient` so `HybridExtractor.extract` can run offline. Those substitutes are **not** used by the Streamlit user path. |
| **Tested with real PDF + real extractor** | Downloaded arXiv:1105.3565 (28-page PDF, not a generated fixture) and ran `register_upload` → `PipelineService.ingest_document` → `RuleExtractor` / `NerExtractor`. Ingestion produced 1211 `CanonicalBlock`s of the paper’s actual text. Rule/NER completed with real `ExtractionResult` objects. This paper is math-ph (hypergeometric functions), not nanoparticle synthesis, so GazetteerNER found no domain entities and RuleExtractor’s numeric matches are whatever patterns appear in that text — expected, not a demo payload. Streamlit widgets were not driven in a browser for this run. |
| **Requires external service/API/model** | GROBID (`GROBID_URL`, default `http://localhost:8070`) if you want title/authors/sections. OCRmyPDF when the native text layer is inadequate. RAG + Hybrid: `ANTHROPIC_API_KEY` or the sidebar key, plus `sentence-transformers` and Hub access for `EMBEDDING_MODEL` (default `BAAI/bge-small-en-v1.5`). `TransformerNER` is not the default NER. |

The sidebar API key is kept in Streamlit session state, passed into `get_extractor(..., api_key=...)`, and is not written to the database, extraction JSON, or provenance.

---

## All 13 Steps Complete

This project now goes from raw PDFs to a served, queryable API: ingestion
(Step 3) → chunking/retrieval (Step 4) → four extraction strategies (Steps
5-8) → normalization (Step 9) → a human-annotated gold dataset (Step 10) →
evaluation (Step 11) → reproducible, tracked experiments (Step 12) → an
actual service (Step 13). 203 tests, every step built on real
infrastructure rather than mocks wherever this sandbox allowed it, and
every deliberate limitation — GROBID's optional enrichment, un-normalized
synonyms until Step 9, chunk-level LLM evidence, and others — documented
and tested for at the step that introduced it rather than discovered later.
