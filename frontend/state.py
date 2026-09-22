"""Session-state keys and document records for the Streamlit UI.

Holds references and serialized results only. Ingestion and extraction are
performed by `frontend.session_pipeline`, which calls `PipelineService`.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

MAX_FILES = 15


class ExtractionAttempt(TypedDict, total=False):
    status: str  # Complete | Failed
    error: Optional[str]
    result: Optional[dict[str, Any]]
    extractor_info: Optional[dict[str, Any]]


class DocumentRecord(TypedDict, total=False):
    filename: str
    size: int
    status: str  # Uploaded | Ingesting | Ready | Failed
    paper_id: Optional[str]
    version_id: Optional[str]
    error: Optional[str]
    canonical_doc: Optional[dict[str, Any]]
    extractions: dict[str, ExtractionAttempt]


SESSION_DOCUMENTS = "documents"
SESSION_API_KEY = "anthropic_api_key"
SESSION_KEY_APPLIED = "api_key_applied"


def empty_document(filename: str, size: int) -> DocumentRecord:
    return {
        "filename": filename,
        "size": size,
        "status": "Uploaded",
        "paper_id": None,
        "version_id": None,
        "error": None,
        "canonical_doc": None,
        "extractions": {},
    }


def init_session(session_state: dict) -> None:
    if SESSION_DOCUMENTS not in session_state:
        session_state[SESSION_DOCUMENTS] = []
    if SESSION_API_KEY not in session_state:
        session_state[SESSION_API_KEY] = ""
    if SESSION_KEY_APPLIED not in session_state:
        session_state[SESSION_KEY_APPLIED] = False
