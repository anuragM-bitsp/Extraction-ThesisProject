"""
get_extractor(): a plain string (as it arrives over HTTP or through a
Celery task's JSON-serialized arguments) -> a real `Extractor`.

RULE and NER need nothing and always work offline. RAG_LLM and HYBRID need
a real embedding model and LLM client — this environment has neither
Hugging Face Hub access nor an ANTHROPIC_API_KEY, so `get_extractor` raises
a clear, actionable error for those two rather than returning something
that would fail confusingly deep inside extraction instead.
"""

from __future__ import annotations

import os
from typing import Optional

from extractors.base import Extractor
from extractors.hybrid_extractor import HybridExtractor
from extractors.ner_extractor import NerExtractor
from extractors.rag.llm_client import AnthropicLLMClient
from extractors.rag_llm_extractor import RagLlmExtractor
from extractors.rule_extractor import RuleExtractor
from retrieval.embeddings import SentenceTransformerEmbedder

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LLM_MODEL = "claude-sonnet-5"
RAG_TOP_K = 4  # RagLlmExtractor's constructor default; exposed so callers can report it


class ExtractorUnavailableError(RuntimeError):
    pass


def resolve_anthropic_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Session-supplied key wins; otherwise the process environment. Never logs the value."""
    if api_key:
        return api_key
    env = os.environ.get("ANTHROPIC_API_KEY")
    return env if env else None


def extractor_status(api_key: Optional[str] = None) -> list[dict]:
    """Which extractors `get_extractor` can currently construct — used by the UI.

    Availability here matches the registry's own gates (API key for RAG/Hybrid).
    Construction can still fail later (embedding weights, Anthropic transport);
    those errors are raised by `get_extractor` / `extract()`, not hidden here.
    """
    has_key = bool(resolve_anthropic_api_key(api_key))
    embedding_model = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    llm_model = os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL)
    missing_key = "Anthropic API key is not configured."

    return [
        {
            "id": "rule",
            "label": "Rule-based",
            "available": True,
            "requires_api_key": False,
            "implementation": "RuleExtractor",
            "reason": None,
        },
        {
            "id": "ner",
            "label": "NER",
            "available": True,
            "requires_api_key": False,
            "implementation": "NerExtractor",
            "ner_implementation": "GazetteerNER",
            "mode": "Offline",
            "reason": None,
        },
        {
            "id": "rag_llm",
            "label": "RAG + LLM",
            "available": has_key,
            "requires_api_key": True,
            "implementation": "RagLlmExtractor",
            "embedding_model": embedding_model,
            "llm_model": llm_model,
            "top_k": RAG_TOP_K,
            "reason": None if has_key else missing_key,
        },
        {
            "id": "hybrid",
            "label": "Hybrid",
            "available": has_key,
            "requires_api_key": True,
            "implementation": "HybridExtractor",
            "embedding_model": embedding_model,
            "llm_model": llm_model,
            "top_k": RAG_TOP_K,
            "reason": None if has_key else missing_key,
        },
    ]


def describe_extractor(extractor: Extractor) -> dict:
    """Report configuration from a live extractor instance — no invented fields."""
    info: dict = {
        "class_name": type(extractor).__name__,
        "name": extractor.name.value if hasattr(extractor.name, "value") else str(extractor.name),
        "version": extractor.version,
    }
    if isinstance(extractor, NerExtractor):
        model = extractor.model
        info["ner_implementation"] = type(model).__name__
        info["ner_model_name"] = getattr(model, "name", type(model).__name__)
        info["mode"] = "Offline" if type(model).__name__ == "GazetteerNER" else "Model"
    if isinstance(extractor, RagLlmExtractor):
        info["embedding_model"] = getattr(extractor.embedder, "name", type(extractor.embedder).__name__)
        info["llm"] = getattr(extractor.llm_client, "name", type(extractor.llm_client).__name__)
        info["top_k"] = extractor.top_k
    if isinstance(extractor, HybridExtractor):
        info["rule_version"] = extractor.rule_extractor.version
        info["ner_version"] = extractor.ner_extractor.version
        info["rag_llm"] = describe_extractor(extractor.rag_llm_extractor)
        info["resolver"] = type(extractor.resolver).__name__
    return info


def get_extractor(name: str, api_key: Optional[str] = None) -> Extractor:
    name = name.lower()

    if name == "rule":
        return RuleExtractor()
    if name == "ner":
        return NerExtractor()

    if name in ("rag_llm", "hybrid"):
        resolved_key = resolve_anthropic_api_key(api_key)
        if not resolved_key:
            raise ExtractorUnavailableError(
                f"extractor={name!r} requires ANTHROPIC_API_KEY to be set (real embeddings also "
                "require Hugging Face Hub access this environment doesn't have). For offline runs "
                "or tests, construct the extractor yourself with HashingEmbedder + FakeLLMClient "
                "and call PipelineService.run_extraction(..., extractor=...) directly instead of "
                "going through this registry."
            )
        embedding_model = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        llm_model = os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL)
        try:
            embedder = SentenceTransformerEmbedder(embedding_model)
        except Exception as exc:
            raise ExtractorUnavailableError(
                f"Configured SentenceTransformerEmbedder model {embedding_model!r} could not be loaded: {exc}"
            ) from exc
        try:
            llm_client = AnthropicLLMClient(model=llm_model, api_key=resolved_key)
        except Exception as exc:
            raise ExtractorUnavailableError(f"AnthropicLLMClient could not be constructed: {exc}") from exc
        rag_llm = RagLlmExtractor(embedder=embedder, llm_client=llm_client, top_k=RAG_TOP_K)
        return rag_llm if name == "rag_llm" else HybridExtractor(rag_llm_extractor=rag_llm)

    raise ValueError(f"unknown extractor name: {name!r}")
