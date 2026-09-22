from __future__ import annotations

import pytest
from pydantic import BaseModel

from extractors.rag.llm_client import (
    FakeLLMClient,
    LLMExtractionError,
    LLMOutputInvalid,
    LLMTransportError,
    call_with_retry,
)
from extractors.rag.prompting import build_prompt
from extractors.rag.tasks import DEFAULT_TASKS, PrecursorsResponse, SynthesisMethodResponse
from extractors.rag_llm_extractor import RagLlmExtractor
from ingestion.canonical import BlockType, CanonicalBlock, CanonicalDocument
from retrieval.embeddings import HashingEmbedder


def block(text, page=1, section=None):
    return CanonicalBlock(block_type=BlockType.PARAGRAPH, page_number=page, section=section, text=text)


# ---- FakeLLMClient ------------------------------------------------------------------


def test_fake_llm_client_validates_dict_into_response_model():
    client = FakeLLMClient([{"synthesis_method": "chemical reduction"}])
    result = client.complete("sys", "user", SynthesisMethodResponse)
    assert result.synthesis_method == "chemical reduction"


def test_fake_llm_client_raises_llm_output_invalid_on_bad_dict():
    client = FakeLLMClient([{"precursors": [{"amount": "not-a-quantity-object"}]}])  # missing required 'name'
    with pytest.raises(LLMOutputInvalid):
        client.complete("sys", "user", PrecursorsResponse)


def test_fake_llm_client_raises_scripted_exception_directly():
    client = FakeLLMClient([LLMTransportError("simulated timeout")])
    with pytest.raises(LLMTransportError):
        client.complete("sys", "user", SynthesisMethodResponse)


def test_fake_llm_client_records_calls():
    client = FakeLLMClient([{"synthesis_method": "green synthesis"}])
    client.complete("sys prompt", "user prompt", SynthesisMethodResponse)
    assert client.calls == [("sys prompt", "user prompt", SynthesisMethodResponse)]


# ---- call_with_retry --------------------------------------------------------------------


def test_call_with_retry_succeeds_first_try():
    client = FakeLLMClient([{"synthesis_method": "chemical reduction"}])
    result = call_with_retry(client, "sys", "user", SynthesisMethodResponse, max_retries=2)
    assert result.synthesis_method == "chemical reduction"


def test_call_with_retry_recovers_after_transient_failures():
    client = FakeLLMClient(
        [
            LLMTransportError("timeout 1"),
            LLMOutputInvalid("malformed json"),
            {"synthesis_method": "hydrothermal synthesis"},
        ]
    )
    result = call_with_retry(client, "sys", "user", SynthesisMethodResponse, max_retries=2)
    assert result.synthesis_method == "hydrothermal synthesis"
    assert len(client.calls) == 3


def test_call_with_retry_raises_llm_extraction_error_after_exhausting_retries():
    client = FakeLLMClient([LLMTransportError("1"), LLMTransportError("2"), LLMTransportError("3")])
    with pytest.raises(LLMExtractionError):
        call_with_retry(client, "sys", "user", SynthesisMethodResponse, max_retries=2)
    assert len(client.calls) == 3  # initial attempt + 2 retries


# ---- prompt building -------------------------------------------------------------------


def test_build_prompt_includes_query_and_chunk_text():
    from retrieval.chunking import ChunkDraft

    task = DEFAULT_TASKS[0]
    chunks = [ChunkDraft(text="Silver nitrate was dissolved in ethanol.", section="Experimental", token_count=7)]
    system_prompt, user_prompt = build_prompt(task, chunks)

    assert task.query in user_prompt
    assert "Silver nitrate was dissolved in ethanol." in user_prompt
    assert "Experimental" in user_prompt
    assert "only use information" in system_prompt.lower()


# ---- RagLlmExtractor (full pipeline) ----------------------------------------------------


def _document_for_extraction():
    return CanonicalDocument(
        paper_id="P001",
        version=1,
        blocks=[
            block("Silver nanoparticles have attracted attention for their antibacterial properties.", page=1, section="Introduction"),
            block(
                "Silver nitrate was dissolved in ethanol and reduced using trisodium citrate as a reducing agent "
                "at room temperature to yield colloidal silver nanoparticles.",
                page=2,
                section="Experimental",
            ),
            block(
                "TEM analysis confirmed an average particle size of 15 nm with spherical morphology.",
                page=6,
                section="Results",
            ),
        ],
    )


def test_rag_llm_extractor_populates_all_task_fields():
    scripted = [
        {"synthesis_method": "chemical reduction using trisodium citrate"},
        {"steps": [{"description": "Dissolve silver nitrate in ethanol"}, {"description": "Add trisodium citrate and stir"}]},
        {"precursors": [{"name": "Silver nitrate", "concentration": {"value": 10.0, "unit": "mM"}, "amount": None}]},
        {"properties": [{"name": "particle size", "value": {"value": 15.0, "unit": "nm"}, "qualitative_value": None}]},
    ]
    llm_client = FakeLLMClient(scripted)
    extractor = RagLlmExtractor(embedder=HashingEmbedder(dimension=128), llm_client=llm_client, top_k=3)

    result = extractor.extract(_document_for_extraction())

    assert result.extractor.value == "RAG_LLM"
    assert result.prediction.synthesis_method == "chemical reduction using trisodium citrate"

    assert [s.order for s in result.prediction.steps] == [1, 2]
    assert result.prediction.steps[0].description == "Dissolve silver nitrate in ethanol"

    assert result.prediction.precursors[0].name == "Silver nitrate"
    assert result.prediction.precursors[0].concentration.value == 10.0

    assert result.prediction.properties[0].name == "particle size"
    assert result.prediction.properties[0].value.unit == "nm"

    assert all(c.source.value == "LLM" for c in result.provenance)
    assert len(llm_client.calls) == 4  # one call per DEFAULT_TASKS entry


def test_rag_llm_extractor_retrieval_targets_the_relevant_chunk_per_task():
    """The whole point of per-field retrieval (LLD section 14): scoring and
    top_k selection should surface the chunk whose vocabulary overlaps the
    query, not just whichever chunk comes first.

    Uses a custom task with a query that lexically overlaps the
    Experimental block on purpose: HashingEmbedder is a bag-of-words hash
    with no semantic understanding (Step 4), so a natural-language question
    like "What precursor chemicals were used?" won't reliably out-rank
    unrelated prose that happens to share a stray token — that's a known
    limitation of the offline test embedder, not of the retrieval wiring
    this test is actually checking. Real semantic matching against natural
    questions is exactly what SentenceTransformerEmbedder is for in
    production.
    """
    from extractors.rag.tasks import FieldTask, PrecursorsResponse, _apply_precursors

    lexically_overlapping_task = FieldTask(
        name="precursors",
        query="silver nitrate ethanol trisodium citrate reducing agent",
        response_model=PrecursorsResponse,
        apply=_apply_precursors,
    )
    scripted = [{"precursors": [{"name": "Silver nitrate", "concentration": None, "amount": None}]}]
    llm_client = FakeLLMClient(scripted)
    extractor = RagLlmExtractor(
        embedder=HashingEmbedder(dimension=128),
        llm_client=llm_client,
        tasks=[lexically_overlapping_task],
        top_k=1,
    )

    extractor.extract(_document_for_extraction())

    _, user_prompt, _ = llm_client.calls[0]
    assert "trisodium citrate as a reducing agent" in user_prompt
    assert "antibacterial properties" not in user_prompt


def test_rag_llm_extractor_skips_a_permanently_failing_task_without_crashing():
    scripted = [
        LLMTransportError("down"), LLMTransportError("still down"), LLMTransportError("still down"),  # synthesis_method exhausts retries
        {"steps": [{"description": "Mix reagents"}]},
        {"precursors": []},
        {"properties": []},
    ]
    llm_client = FakeLLMClient(scripted)
    extractor = RagLlmExtractor(embedder=HashingEmbedder(dimension=64), llm_client=llm_client, max_retries=2)

    result = extractor.extract(_document_for_extraction())

    assert result.prediction.synthesis_method is None  # failed field left empty, not crashed
    assert result.prediction.steps[0].description == "Mix reagents"  # later tasks still ran


def test_rag_llm_extractor_handles_document_with_no_extractable_blocks():
    document = CanonicalDocument(paper_id="P001", version=1, blocks=[])
    extractor = RagLlmExtractor(embedder=HashingEmbedder(dimension=32), llm_client=FakeLLMClient([]))

    result = extractor.extract(document)
    assert result.prediction.precursors == []
    assert result.provenance == []


def test_rag_llm_extractor_evidence_is_chunk_level():
    scripted = [
        {"synthesis_method": "chemical reduction"},
        {"steps": []},
        {"precursors": []},
        {"properties": []},
    ]
    llm_client = FakeLLMClient(scripted)
    extractor = RagLlmExtractor(embedder=HashingEmbedder(dimension=64), llm_client=llm_client)

    result = extractor.extract(_document_for_extraction())
    fact = next(c for c in result.provenance if c.field == "synthesis_method")
    assert fact.evidence.section is not None
    assert len(fact.evidence.text) > 0
    assert fact.evidence.char_start is None  # chunk-level, not a precise char span — documented trade-off
