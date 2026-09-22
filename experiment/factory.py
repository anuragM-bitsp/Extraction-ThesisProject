"""
build_extractor(): ExperimentConfig -> a real Extractor instance.

Deliberately takes `embedder`/`llm_client` as explicit arguments rather
than constructing them from `config.embedding_model_name`/`llm_model_name`
strings. Those strings are for logging (see ExperimentConfig's docstring);
turning a string into a live `SentenceTransformerEmbedder` or
`AnthropicLLMClient` — or a `HashingEmbedder`/`FakeLLMClient` for a test
run — is a deployment/test decision the caller already has to make, and
this factory shouldn't need to know about every provider that might ever
exist to make it.
"""

from __future__ import annotations

from experiment.config import ExperimentConfig
from extractors.base import Extractor
from extractors.hybrid_extractor import HybridExtractor
from extractors.ner_extractor import NerExtractor
from extractors.rag.llm_client import LLMClient
from extractors.rag_llm_extractor import RagLlmExtractor
from extractors.rule_extractor import RuleExtractor
from retrieval.embeddings import EmbeddingModel


def build_extractor(
    config: ExperimentConfig,
    embedder: EmbeddingModel | None = None,
    llm_client: LLMClient | None = None,
) -> Extractor:
    if config.extractor == "rule":
        return RuleExtractor()

    if config.extractor == "ner":
        return NerExtractor()

    if config.extractor in ("rag_llm", "hybrid"):
        if embedder is None or llm_client is None:
            raise ValueError(
                f"extractor={config.extractor!r} requires both `embedder` and `llm_client` — "
                "neither RagLlmExtractor nor HybridExtractor can default-construct these "
                "(see Step 8's README for why HybridExtractor already refuses to)"
            )
        rag_llm = RagLlmExtractor(
            embedder=embedder, llm_client=llm_client, top_k=config.top_k, max_retries=config.max_retries
        )
        if config.extractor == "rag_llm":
            return rag_llm
        return HybridExtractor(rag_llm_extractor=rag_llm)

    raise ValueError(f"unknown extractor kind: {config.extractor!r}")
