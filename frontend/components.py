"""Streamlit widgets. Display-only: they render data produced by session_pipeline."""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.session_pipeline import document_metadata, result_json_bytes
from frontend.state import DocumentRecord
from schemas.provenance import CandidateFact


def render_upload_header() -> None:
    st.title("Scientific PDF Extraction")
    st.subheader("Upload your scientific papers")
    st.caption("Supported: PDF · Maximum: 15 files")


def render_document_table(records: list[DocumentRecord]) -> None:
    if not records:
        return
    st.dataframe(
        [
            {
                "Filename": rec.get("filename", ""),
                "Size": rec.get("size", 0),
                "Status": rec.get("status", ""),
            }
            for rec in records
        ],
        hide_index=True,
        use_container_width=True,
    )
    for rec in records:
        if rec.get("status") == "Failed" and rec.get("error"):
            st.error(f"{rec.get('filename')}: {rec.get('error')}")


def render_extractor_availability(statuses: list[dict]) -> None:
    st.markdown("**Extractor availability**")
    for item in statuses:
        if item["available"]:
            extra = ""
            if item["id"] == "ner":
                extra = f" — implementation: {item.get('ner_implementation')} ({item.get('mode')})"
            st.success(f"{item['label']}  ✓ Available{extra}")
        else:
            reason = item.get("reason") or "Unavailable"
            st.warning(f"{item['label']}  🔒 Requires API key — {reason}")


def render_extractor_config(statuses: list[dict], selected_id: str) -> None:
    item = next((s for s in statuses if s["id"] == selected_id), None)
    if item is None:
        return
    lines = [f"Implementation: {item.get('implementation')}"]
    if item["id"] == "ner":
        lines.append(f"NER implementation: {item.get('ner_implementation')}")
        lines.append(f"Mode: {item.get('mode')}")
    if item["id"] in ("rag_llm", "hybrid") and item.get("available"):
        if item.get("embedding_model"):
            lines.append(f"Embedding model: {item['embedding_model']}")
        if item.get("llm_model"):
            lines.append(f"LLM: {item['llm_model']}")
        if item.get("top_k") is not None:
            lines.append(f"Top K: {item['top_k']}")
        if item["id"] == "hybrid":
            lines.append("Result comes from HybridExtractor (Rule + NER + RAG+LLM + ConflictResolver).")
    st.caption(" · ".join(lines))


def render_batch_status(records: list[DocumentRecord]) -> None:
    st.markdown(f"**{len(records)} documents**")
    for rec in records:
        name = rec.get("filename", "")
        status = rec.get("status", "")
        if status == "Ready":
            st.write(f"{name}     ✓ Ready")
        elif status == "Failed":
            st.write(f"{name}     ✗ Failed")
            if rec.get("error"):
                st.error(rec["error"])
        else:
            st.write(f"{name}     {status}")


def _evidence_granularity(fact: CandidateFact) -> str:
    ev = fact.evidence
    if ev.char_start is not None and ev.char_end is not None:
        return "character span"
    if ev.page is not None or ev.section:
        return "page/section (no character span)"
    return "chunk-level or unlocalized"


def render_evidence(provenance: list[dict[str, Any]]) -> None:
    if not provenance:
        st.info("This extraction result has no provenance entries.")
        return
    for raw in provenance:
        fact = CandidateFact.model_validate(raw)
        ev = fact.evidence
        st.markdown(f"**Field:** `{fact.field}`")
        st.write(f"Value: {fact.value}")
        st.write(f"Source: {fact.source.value if hasattr(fact.source, 'value') else fact.source}")
        if ev.page is not None:
            st.write(f"Page: {ev.page}")
        if ev.section:
            st.write(f"Section: {ev.section}")
        st.write(f"Evidence: {ev.text!r}")
        st.caption(f"Evidence granularity: {_evidence_granularity(fact)}")
        st.divider()


def render_raw_text(canonical_dump: dict | None) -> None:
    if not canonical_dump:
        st.info("No CanonicalDocument is stored for this paper yet.")
        return
    from ingestion.canonical import CanonicalDocument

    doc = CanonicalDocument.model_validate(canonical_dump)
    text = doc.full_text()
    if not text.strip():
        st.warning("CanonicalDocument.full_text() is empty (native PDF text may be missing; OCR may not have run).")
        return
    st.text(text)


def render_extraction_result(record: DocumentRecord, strategy: str, strategy_label: str) -> None:
    filename = record.get("filename", "")
    attempt = (record.get("extractions") or {}).get(strategy)
    st.markdown(f"**{filename}**")
    st.write(f"Extraction Strategy: {strategy_label}")
    if attempt is None:
        st.write("Status: Not run")
        return

    st.write(f"Status: {attempt.get('status')}")
    info = attempt.get("extractor_info") or {}
    if info:
        bits = [info.get("class_name"), info.get("version")]
        if info.get("ner_implementation"):
            bits.append(f"NER implementation: {info['ner_implementation']}")
            if info.get("mode"):
                bits.append(f"Mode: {info['mode']}")
        if info.get("embedding_model"):
            bits.append(f"Embedding model: {info['embedding_model']}")
        if info.get("llm"):
            bits.append(f"LLM: {info['llm']}")
        if info.get("top_k") is not None:
            bits.append(f"Top K: {info['top_k']}")
        if info.get("resolver"):
            bits.append(f"Fusion: {info['resolver']}")
        st.caption(" · ".join(str(b) for b in bits if b))

    if attempt.get("error"):
        st.error(attempt["error"])
        return

    result = attempt.get("result")
    if not result:
        st.error("No ExtractionResult was returned.")
        return

    json_tab, evidence_tab, text_tab, meta_tab = st.tabs(
        ["Structured JSON", "Evidence", "Raw Text", "Metadata"]
    )
    with json_tab:
        st.json(result)
        payload = result_json_bytes(result)
        st.download_button(
            "Download JSON",
            data=payload,
            file_name=f"{filename}.{strategy}.extraction.json",
            mime="application/json",
            key=f"dl-{record.get('version_id')}-{strategy}",
        )
    with evidence_tab:
        render_evidence(result.get("provenance") or [])
    with text_tab:
        render_raw_text(record.get("canonical_doc"))
    with meta_tab:
        st.json(
            document_metadata(
                record.get("canonical_doc"),
                filename,
                record.get("paper_id"),
                record.get("version_id"),
            )
        )
