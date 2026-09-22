"""
Streamlit UI for sci-extract.

Start with:

    streamlit run frontend/app.py

All PDF parsing and extraction happens in Steps 1–13 (`PipelineService`,
`get_extractor`, `DocumentIngestionPipeline`, the four Extractor classes).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from frontend.components import (
    render_batch_status,
    render_document_table,
    render_extraction_result,
    render_extractor_availability,
    render_extractor_config,
    render_upload_header,
)
from frontend.session_pipeline import availability, extract_records, ingest_records, register_upload
from frontend.state import (
    MAX_FILES,
    SESSION_API_KEY,
    SESSION_DOCUMENTS,
    SESSION_KEY_APPLIED,
    init_session,
)

UPLOAD_FINGERPRINT = "upload_fingerprint"
from orchestration.db import get_object_store, get_session
from storage.repository import PaperRepository

STRATEGY_LABELS = {
    "rule": "Rule-based",
    "ner": "NER",
    "rag_llm": "RAG + LLM",
    "hybrid": "Hybrid",
}


def _repo() -> PaperRepository:
    session = get_session()
    return PaperRepository(session, get_object_store())


def _applied_api_key() -> str | None:
    if st.session_state.get(SESSION_KEY_APPLIED) and st.session_state.get(SESSION_API_KEY):
        return st.session_state[SESSION_API_KEY]
    return None


def render_sidebar() -> None:
    st.sidebar.header("LLM Configuration")
    st.sidebar.text_input("API Key", type="password", key=SESSION_API_KEY, help="Used only in this session. Not written to the database or extraction JSON.")
    if st.sidebar.button("Apply"):
        key = st.session_state.get(SESSION_API_KEY) or ""
        if key.strip():
            st.session_state[SESSION_KEY_APPLIED] = True
            st.sidebar.success("API key applied for this session (value is not displayed).")
        else:
            st.session_state[SESSION_KEY_APPLIED] = False
            st.sidebar.warning("No API key entered.")
    elif st.session_state.get(SESSION_KEY_APPLIED):
        st.sidebar.caption("A session API key is configured (not shown).")


def render_app() -> None:
    st.set_page_config(page_title="Scientific PDF Extraction", layout="wide")
    init_session(st.session_state)
    render_sidebar()
    render_upload_header()

    uploads = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if uploads:
        if len(uploads) > MAX_FILES:
            st.error(f"Maximum {MAX_FILES} files.")
        else:
            fingerprint = tuple((u.name, u.size) for u in uploads)
            if st.session_state.get(UPLOAD_FINGERPRINT) != fingerprint:
                repo = _repo()
                records = []
                for uploaded in uploads:
                    try:
                        records.append(register_upload(repo, uploaded.name, uploaded.getvalue()))
                    except Exception as exc:
                        records.append(
                            {
                                "filename": uploaded.name,
                                "size": getattr(uploaded, "size", 0),
                                "status": "Failed",
                                "paper_id": None,
                                "version_id": None,
                                "error": str(exc),
                                "canonical_doc": None,
                                "extractions": {},
                            }
                        )
                st.session_state[SESSION_DOCUMENTS] = records
                st.session_state[UPLOAD_FINGERPRINT] = fingerprint

    documents: list = st.session_state.get(SESSION_DOCUMENTS) or []
    render_document_table(documents)

    if documents and st.button("Process Documents"):
        repo = _repo()
        st.session_state[SESSION_DOCUMENTS] = ingest_records(repo, documents)
        documents = st.session_state[SESSION_DOCUMENTS]

    if documents:
        render_batch_status(documents)
        ready = [d for d in documents if d.get("status") == "Ready"]
        st.markdown(f"Documents ready: {len(ready)}")

        statuses = availability(api_key=_applied_api_key())
        render_extractor_availability(statuses)

        available_ids = [s["id"] for s in statuses if s["available"]]
        labels = {s["id"]: s["label"] for s in statuses}
        if not available_ids:
            st.error("No extractors are currently available.")
            return

        st.markdown("Choose extraction strategy:")
        selected_label = st.radio(
            "Choose extraction strategy:",
            [labels[i] for i in available_ids],
            index=0,
            label_visibility="collapsed",
        )
        selected_id = next(i for i in available_ids if labels[i] == selected_label)
        render_extractor_config(statuses, selected_id)

        locked = [s for s in statuses if not s["available"]]
        if locked:
            st.caption("Unavailable until an API key is applied: " + ", ".join(s["label"] for s in locked))

        if st.button("Run Extraction"):
            repo = _repo()
            st.session_state[SESSION_DOCUMENTS] = extract_records(
                repo, documents, selected_id, api_key=_applied_api_key()
            )
            documents = st.session_state[SESSION_DOCUMENTS]

        for rec in documents:
            attempt = (rec.get("extractions") or {}).get(selected_id)
            if attempt:
                render_extraction_result(rec, selected_id, STRATEGY_LABELS.get(selected_id, selected_id))


if __name__ == "__main__":
    render_app()
