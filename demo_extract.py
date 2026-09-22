"""
Quick local demo — no database, no API server, no Redis required.

Runs ingestion (Step 3) + the Rule (Step 5) and NER (Step 6) extractors
on a single PDF and prints the structured result as JSON. These two
extractors need nothing beyond what `pip install -r requirements.txt`
already gives you.

If ANTHROPIC_API_KEY is set in your environment, it also runs the RAG+LLM
extractor (Step 7) and the Hybrid extractor (Step 8) — those two make real
calls to the Anthropic API and will incur API costs.

Usage:
    python3 demo_extract.py path/to/paper.pdf
"""

from __future__ import annotations

import os
import sys

from extractors.hybrid_extractor import HybridExtractor
from extractors.ner_extractor import NerExtractor
from extractors.rag.llm_client import AnthropicLLMClient
from extractors.rag_llm_extractor import RagLlmExtractor
from extractors.rule_extractor import RuleExtractor
from ingestion.pipeline import DocumentIngestionPipeline
from retrieval.embeddings import HashingEmbedder


def main(pdf_path: str) -> None:
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    print(f"Ingesting {pdf_path} ...")
    # No GROBID client passed -> falls back to PyMuPDF-only parsing (Step 3).
    # Pass grobid_client=HttpGrobidClient("http://localhost:8070") for
    # section-aware parsing + title/authors/abstract, if you have a GROBID
    # instance running (see README's Step 3 section for the docker command).
    pipeline = DocumentIngestionPipeline()
    doc = pipeline.ingest(paper_id="demo", version=1, pdf_bytes=pdf_bytes)
    print(f"Parsed {len(doc.blocks)} blocks (OCR ran: {doc.was_ocred}, GROBID used: {doc.used_grobid})\n")

    print("=== Rule-based extraction (Step 5) ===")
    rule_result = RuleExtractor().extract(doc)
    print(rule_result.prediction.model_dump_json(indent=2))

    print("\n=== NER extraction (Step 6) ===")
    ner_result = NerExtractor().extract(doc)
    print(ner_result.prediction.model_dump_json(indent=2))

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n(Set ANTHROPIC_API_KEY to also run the RAG+LLM and Hybrid extractors — these make real API calls)")
        return

    print("\n=== Hybrid extraction: Rule + NER + RAG/LLM (Steps 7-8) ===")
    # HashingEmbedder is the offline/test embedder from Step 4 — swap for
    # SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5") for real
    # semantic retrieval quality (requires Hugging Face Hub access).
    embedder = HashingEmbedder(dimension=128)
    llm_client = AnthropicLLMClient(model="claude-sonnet-5", api_key=api_key)
    hybrid = HybridExtractor(rag_llm_extractor=RagLlmExtractor(embedder=embedder, llm_client=llm_client))
    hybrid_result = hybrid.extract(doc)
    print(hybrid_result.prediction.model_dump_json(indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 demo_extract.py path/to/paper.pdf")
        sys.exit(1)
    main(sys.argv[1])
