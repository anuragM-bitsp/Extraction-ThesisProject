"""
ExperimentConfig (LLD section 23).

    "Then use MLflow or similar experiment tracking to record: model
    version, prompt version, metrics, parameters, dataset version,
    experiment ID... Now you can reproduce: 'Hybrid-v3 achieved F1 = 0.87
    on Gold-v2.'"

That sentence only means something if "Hybrid-v3" refers to an exact,
recorded set of hyperparameters — not just a class name someone remembers
changing at some point. This config IS that recorded set; `as_params()` is
what gets logged to MLflow so a run can be reproduced later from its logged
parameters alone, without needing to read the code that produced it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ExtractorKind = Literal["rule", "ner", "rag_llm", "hybrid"]


class ExperimentConfig(BaseModel):
    name: str
    extractor: ExtractorKind

    schema_version: str = "v1"
    top_k: int = 4
    max_retries: int = 2

    # Logged for reproducibility even though the actual embedder/LLM
    # objects are injected separately (see experiment/factory.py) — a
    # config string like "BAAI/bge-small-en-v1.5" doesn't by itself know
    # how to construct that model, and hard-coding that mapping here would
    # make this module responsible for every embedding/LLM provider that
    # might ever be used.
    embedding_model_name: str = "unspecified"
    llm_model_name: str = "unspecified"

    gold_dataset_version: str = "unspecified"
    notes: str = ""

    def as_params(self) -> dict[str, str]:
        """Flat str -> str dict, exactly what MLflow's log_params expects."""
        return {k: str(v) for k, v in self.model_dump().items()}
