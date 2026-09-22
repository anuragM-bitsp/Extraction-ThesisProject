"""
Prompt construction for one FieldTask against its retrieved chunks.

Kept as a pure function returning (system_prompt, user_prompt) strings so
it's trivially unit-testable without an LLM client at all — the interesting
behavior to verify is "did the right chunks and query end up in the prompt
text," not anything about generation.
"""

from __future__ import annotations

from extractors.rag.tasks import FieldTask
from retrieval.chunking import ChunkDraft

SYSTEM_PROMPT = (
    "You are extracting structured scientific synthesis information from excerpts "
    "of a research paper. Only use information explicitly present in the provided "
    "excerpts. If the requested information is not present, leave the corresponding "
    "fields null or empty rather than guessing or using outside knowledge."
)


def build_prompt(task: FieldTask, chunks: list[ChunkDraft]) -> tuple[str, str]:
    context = "\n\n".join(
        f"[Excerpt {i}] (section: {c.section or 'unknown'}, page {c.page_start}-{c.page_end})\n{c.text}"
        for i, c in enumerate(chunks)
    )
    user_prompt = f"Question: {task.query}\n\nRelevant excerpts from the paper:\n\n{context}"
    return SYSTEM_PROMPT, user_prompt
