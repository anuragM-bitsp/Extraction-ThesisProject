"""
RagLlmExtractor: the third of the four extraction strategies (LLD sections
9, 12-15).

Per task (LLD section 14 — field-specific, not one giant prompt):
    query -> embed -> rank this document's chunks -> top_k -> build prompt
          -> LLMClient.complete(..., response_model) [with retry]
          -> merge validated response into SynthesisExtraction + provenance

Chunking/embedding happens fresh per `extract()` call rather than depending
on Step 2's persisted chunk store — this keeps `Extractor.extract(document)`
uniform across all four strategies (no extractor needs a database
connection to run). A real pipeline would of course reuse the chunks Step 4
already indexed rather than re-embedding on every call; that wiring belongs
to Step 12 (experiment runner), not to the extractor's interface.
"""

from __future__ import annotations

from extractors.base import Extractor
from extractors.rag.llm_client import LLMClient, LLMExtractionError, call_with_retry
from extractors.rag.prompting import build_prompt
from extractors.rag.tasks import DEFAULT_TASKS, FieldTask
from ingestion.canonical import CanonicalDocument
from retrieval.chunking import chunk_blocks
from retrieval.embeddings import EmbeddingModel
from retrieval.retriever import cosine_similarity
from schemas.extraction_schema import ExtractionResult, ExtractorName, SynthesisExtraction
from schemas.provenance import CandidateFact


class RagLlmExtractor(Extractor):
    name = ExtractorName.RAG_LLM

    def __init__(
        self,
        embedder: EmbeddingModel,
        llm_client: LLMClient,
        tasks: list[FieldTask] | None = None,
        top_k: int = 4,
        max_retries: int = 2,
    ):
        self.embedder = embedder
        self.llm_client = llm_client
        self.tasks = tasks if tasks is not None else DEFAULT_TASKS
        self.top_k = top_k
        self.max_retries = max_retries
        llm_name = getattr(llm_client, "name", llm_client.__class__.__name__)
        self.version = f"rag_llm@{embedder.name}+{llm_name}"

    def extract(self, document: CanonicalDocument) -> ExtractionResult:
        prediction = SynthesisExtraction(paper_id=document.paper_id)
        candidates: list[CandidateFact] = []

        drafts = chunk_blocks(document.blocks)
        if drafts:
            chunk_vectors = self.embedder.embed([d.text for d in drafts])

            for task in self.tasks:
                query_vector = self.embedder.embed([task.query])[0]
                ranked = sorted(
                    zip(drafts, chunk_vectors),
                    key=lambda pair: cosine_similarity(query_vector, pair[1]),
                    reverse=True,
                )
                relevant_chunks = [d for d, _ in ranked[: self.top_k]]
                if not relevant_chunks:
                    continue

                system_prompt, user_prompt = build_prompt(task, relevant_chunks)
                try:
                    response = call_with_retry(
                        self.llm_client, system_prompt, user_prompt, task.response_model, self.max_retries
                    )
                except LLMExtractionError:
                    # One field permanently failing shouldn't sink the whole
                    # extraction (LLD section 26) — a partial result is
                    # still useful, and other tasks are independent.
                    continue

                task.apply(prediction, response, relevant_chunks, candidates, document, self.version)

        return ExtractionResult(
            paper_id=document.paper_id,
            version=document.version,
            extractor=self.name,
            extractor_version=self.version,
            prediction=prediction,
            provenance=candidates,
        )
