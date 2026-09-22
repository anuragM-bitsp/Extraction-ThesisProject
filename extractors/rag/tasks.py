"""
Field-specific extraction tasks.

LLD section 14: "Instead of 'Extract everything from this paper,' I'd have
field-specific extraction tasks... Different information types occur in
different sections of scientific papers, so task-specific retrieval
improves precision and reduces unnecessary context."

Each FieldTask pairs a retrieval query with a small response schema and an
`apply` function that merges the validated response into the shared
SynthesisExtraction + emits CandidateFacts. Evidence for LLM-sourced facts
is necessarily chunk-level (page range + section + chunk text), not a
precise character span — a generated answer isn't naturally anchored to an
exact offset the way a regex match is. That's a real, stated trade-off
against rule/NER evidence precision, not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type

from pydantic import BaseModel

from ingestion.canonical import CanonicalDocument
from retrieval.chunking import ChunkDraft
from schemas.extraction_schema import MaterialProperty, Precursor, Quantity, SynthesisExtraction, SynthesisStep
from schemas.provenance import CandidateFact, EvidenceSpan, SourceType

EVIDENCE_TEXT_PREVIEW_CHARS = 200


def _chunk_evidence(chunks: list[ChunkDraft], document: CanonicalDocument) -> EvidenceSpan:
    top_chunk = chunks[0]  # highest-ranked retrieved chunk, used as primary evidence
    return EvidenceSpan(
        paper_id=document.paper_id,
        version=document.version,
        page=top_chunk.page_start,
        section=top_chunk.section,
        text=top_chunk.text[:EVIDENCE_TEXT_PREVIEW_CHARS],
    )


def _fact(field: str, value, chunks, document, extractor_version, confidence) -> CandidateFact:
    return CandidateFact(
        field=field,
        value=value,
        source=SourceType.LLM,
        confidence=confidence,
        evidence=_chunk_evidence(chunks, document),
        extractor_version=extractor_version,
    )


# ---- response schemas the LLM's output is constrained to --------------------------


class SynthesisMethodResponse(BaseModel):
    synthesis_method: str | None = None


class StepItem(BaseModel):
    description: str


class StepsResponse(BaseModel):
    steps: list[StepItem] = []


class PrecursorItem(BaseModel):
    name: str
    amount: Quantity | None = None
    concentration: Quantity | None = None


class PrecursorsResponse(BaseModel):
    precursors: list[PrecursorItem] = []


class PropertyItem(BaseModel):
    name: str
    value: Quantity | None = None
    qualitative_value: str | None = None


class PropertiesResponse(BaseModel):
    properties: list[PropertyItem] = []


# ---- FieldTask + apply functions ---------------------------------------------------


@dataclass
class FieldTask:
    name: str
    query: str
    response_model: Type[BaseModel]
    apply: Callable[[SynthesisExtraction, BaseModel, list[ChunkDraft], list[CandidateFact], CanonicalDocument, str], None]


def _apply_synthesis_method(prediction, response: SynthesisMethodResponse, chunks, candidates, document, extractor_version):
    if not response.synthesis_method:
        return
    prediction.synthesis_method = response.synthesis_method
    candidates.append(_fact("synthesis_method", response.synthesis_method, chunks, document, extractor_version, 0.75))


def _apply_steps(prediction, response: StepsResponse, chunks, candidates, document, extractor_version):
    for i, item in enumerate(response.steps, start=1):
        prediction.steps.append(SynthesisStep(order=i, description=item.description))
        candidates.append(_fact(f"steps[{i - 1}].description", item.description, chunks, document, extractor_version, 0.7))


def _apply_precursors(prediction, response: PrecursorsResponse, chunks, candidates, document, extractor_version):
    for item in response.precursors:
        prediction.precursors.append(Precursor(name=item.name, amount=item.amount, concentration=item.concentration))
        idx = len(prediction.precursors) - 1
        candidates.append(_fact(f"precursors[{idx}].name", item.name, chunks, document, extractor_version, 0.75))
        if item.concentration is not None:
            candidates.append(
                _fact(f"precursors[{idx}].concentration", item.concentration.model_dump(), chunks, document, extractor_version, 0.7)
            )
        if item.amount is not None:
            candidates.append(
                _fact(f"precursors[{idx}].amount", item.amount.model_dump(), chunks, document, extractor_version, 0.7)
            )


def _apply_properties(prediction, response: PropertiesResponse, chunks, candidates, document, extractor_version):
    for item in response.properties:
        prediction.properties.append(
            MaterialProperty(name=item.name, value=item.value, qualitative_value=item.qualitative_value)
        )
        idx = len(prediction.properties) - 1
        candidates.append(_fact(f"properties[{idx}].name", item.name, chunks, document, extractor_version, 0.7))


DEFAULT_TASKS: list[FieldTask] = [
    FieldTask(
        name="synthesis_method",
        query="What overall synthesis method or approach was used to prepare the material?",
        response_model=SynthesisMethodResponse,
        apply=_apply_synthesis_method,
    ),
    FieldTask(
        name="steps",
        query="What are the step-by-step procedural actions taken during synthesis, in order?",
        response_model=StepsResponse,
        apply=_apply_steps,
    ),
    FieldTask(
        name="precursors",
        query="What precursor chemicals were used, and what were their amounts or concentrations?",
        response_model=PrecursorsResponse,
        apply=_apply_precursors,
    ),
    FieldTask(
        name="properties",
        query="What properties (size, morphology, band gap, zeta potential, etc.) were measured for the resulting material?",
        response_model=PropertiesResponse,
        apply=_apply_properties,
    ),
]
