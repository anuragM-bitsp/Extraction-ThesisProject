# Sci-Extract

## Scientific Literature to Structured Experimental Data

Sci-Extract is a modular pipeline for extracting structured experimental information from scientific papers. It is designed for chemistry and materials-science literature and supports multiple extraction strategies, provenance tracking, quantitative evaluation, and reproducible experiments.

The core objective is to transform:

```text
Scientific PDF
      ↓
Canonical Document
      ↓
Chunks + Retrieval
      ↓
Rule-based / NER / RAG + LLM
      ↓
Normalization + Linking
      ↓
Structured Scientific Data
      ↓
Evaluation + MLflow
```

---

## Architecture

```mermaid
flowchart TD
    A[Scientific PDF] --> B[Ingestion]
    B --> C[Canonical Document]
    C --> D[Chunking]
    D --> E[Embeddings + Retrieval]

    C --> F[Rule Extractor]
    C --> G[NER Extractor]
    E --> H[RAG + LLM Extractor]

    F --> I[Hybrid / Fusion]
    G --> I
    H --> I

    I --> J[Normalization + Entity Linking]
    J --> K[Structured Schema]

    K --> L[PostgreSQL / Object Storage]

    M[Gold Dataset] --> N[Evaluation]
    K --> N
    N --> O[Metrics + Error Analysis]
    O --> P[MLflow]

    Q[FastAPI] --> R[Service Layer]
    R --> S[Celery Workers]
    S --> B
    S --> F
    S --> G
    S --> H
```

---

## Repository Structure

```text
sci-extract/
│
├── schemas/             # Extraction schemas and validation
├── storage/             # Database and object-storage layer
├── ingestion/           # PDF → canonical document
├── retrieval/           # Chunking, embeddings and retrieval
├── extractors/          # Rule, NER, RAG/LLM and hybrid extraction
├── normalization/       # Units and entity normalization/linking
├── evaluation/          # Gold data, metrics and error analysis
├── experiments/         # Reproducible experiments and MLflow
├── services/            # Application/service layer
├── api/                 # FastAPI routes
├── workers/             # Celery tasks and background processing
├── tests/               # Unit and integration tests
├── scripts/             # Utility and pipeline scripts
├── notebooks/           # Research and exploratory analysis
│
├── pyproject.toml
└── README.md
```

---

# Pipeline

## 1. Schema Definition

The schema defines the structure of the extracted scientific information.

Example:

```json
{
  "paper_id": "paper-001",
  "schema_version": "v2",
  "material": null,
  "precursors": [],
  "solvent": null,
  "temperature": null,
  "reaction_time": null,
  "pH": null,
  "synthesis_method": null,
  "steps": [],
  "characterization": [],
  "properties": []
}
```

Schemas provide:

- consistent output across extractors
- validation
- versioning
- easier storage
- comparable evaluation

The schema can evolve toward richer protocol-level structures as the research develops.

---

## 2. Storage

The storage layer separates scientific processing from persistence.

```text
PostgreSQL
├── papers
├── versions
├── blocks
├── chunks
└── extraction records

Object Storage
├── original PDFs
├── canonical documents
└── large artifacts
```

PostgreSQL handles structured metadata and relationships, while object storage handles large document artifacts.

---

## 3. Document Ingestion

The ingestion layer converts a PDF into a canonical representation.

```text
PDF
 │
 ├── PyMuPDF
 ├── GROBID
 └── OCR fallback
       ↓
Canonical Document
       ↓
Pages / Sections / Blocks / Tables / Figures
```

The canonical document provides a common representation for every downstream extractor.

This avoids having each extraction method independently parse the original PDF.

---

## 4. Chunking and Retrieval

The canonical document is divided into retrieval-friendly chunks.

```text
Canonical Document
       ↓
Section-aware segmentation
       ↓
Blocks / Paragraphs
       ↓
Chunks
       ↓
Embeddings
       ↓
Vector Retrieval
```

Chunks retain metadata such as:

```json
{
  "paper_id": "paper-001",
  "page": 4,
  "section": "Experimental Section",
  "block_id": "block-127",
  "text": "..."
}
```

This preserves the relationship between retrieved text and its original location.

For RAG extraction, retrieval supplies the most relevant experimental context to the LLM.

---

# Extraction

## 5. Rule-Based Extraction

The rule-based extractor provides a deterministic baseline.

It targets structured patterns such as:

```text
35 °C
10 min
0.2 mL
25 mM
0.4 M
50 mg
pH 7
```

### Advantages

- deterministic
- fast
- reproducible
- inexpensive
- easy to debug

### Limitation

Rules can detect values without understanding which chemical, step, or experimental protocol they belong to.

---

## 6. NER Extraction

The NER layer identifies scientific entities such as:

```text
AgNO3
HAuCl4
NaBH4
CTAB
CTAC
water
ethanol
TEM
UV-vis
```

Detected entities can later be mapped to the project schema.

NER focuses primarily on entity identification; contextual relationships and attributes are handled by later stages.

---

## 7. RAG + LLM Extraction

The RAG pipeline retrieves relevant experimental context before sending it to an LLM.

```text
Query
  +
Retrieved Chunks
  +
Extraction Schema
       ↓
Schema-Constrained LLM
       ↓
Structured JSON
```

This approach is intended to improve context-sensitive extraction when papers contain:

- multiple experiments
- repeated chemicals
- multiple concentrations
- different protocols
- complex experimental procedures

The LLM should not be treated as the sole source of truth; its output is validated and evaluated against reference data.

---

## 8. Hybrid Extraction

The hybrid layer can combine:

```text
Rule-based
     +
NER
     +
RAG + LLM
     ↓
Fusion
     ↓
Conflict Resolution
     ↓
Final Extraction
```

The goal is to combine deterministic signals with semantic/contextual extraction.

---

## 9. Normalization and Entity Linking

Raw extraction must be normalized before becoming a reliable scientific dataset.

Examples:

```text
35 °C
35°C
35 degrees C
```

→

```text
value = 35
unit = °C
```

Similarly:

```text
silver nitrate
AgNO3
AgNO₃
```

can be linked to a common canonical entity.

Normalization covers:

- units
- chemical names
- abbreviations
- aliases
- numerical representations
- entity identifiers

---

# Evaluation

## 10. Gold Dataset

A research system requires reference annotations rather than relying only on qualitative inspection.

```text
Scientific Papers
       ↓
Human Annotation
       ↓
Gold Dataset
       ↓
Predictions vs Reference
```

The gold dataset can contain:

- entities
- quantities
- units
- relationships
- experimental steps
- protocol context
- provenance

Multiple annotators can be used to measure agreement before adjudication.

---

## 11. Evaluation

The evaluation layer measures extraction quality quantitatively.

Standard metrics include:

```text
Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 = 2 × Precision × Recall
     -------------------------
       Precision + Recall
```

For scientific extraction, evaluation should go beyond entity detection.

### Entity-level

Was the entity detected?

### Attribute-level

Was the correct quantity or condition extracted?

### Relationship-level

Was the quantity associated with the correct entity?

### Context-level

Did the value belong to the correct experimental protocol?

### Unit-level

Are equivalent units handled correctly?

---

## Error Analysis

Typical error categories include:

```text
Entity missed
Entity hallucinated
Wrong quantity
Wrong unit
Wrong entity-value association
Wrong experimental context
Duplicate entity
Protocol mixing
Normalization error
Retrieval failure
Schema validation failure
```

Error analysis is used to identify where improvements are actually required.

---

# 12. MLflow Experiment Tracking

MLflow records the configuration and results of extraction experiments.

```text
Experiment
├── Dataset version
├── Extractor
├── Model
├── Prompt/configuration
├── Chunking parameters
├── Retrieval parameters
├── Metrics
└── Artifacts
```

This enables reproducible comparisons between:

```text
Rule-based
    vs
NER
    vs
RAG + LLM
    vs
Hybrid
```

The objective is to make every reported result traceable to the configuration that produced it.

---

# 13. Application and Orchestration

The application layer exposes the pipeline through FastAPI and supports asynchronous processing with Celery.

```text
Client
  ↓
FastAPI
  ↓
Service Layer
  ↓
Celery Queue
  ↓
Worker
  ↓
Extraction Pipeline
  ↓
PostgreSQL / Object Storage
```

Redis can act as the Celery message broker.

Background processing is useful because PDF parsing, retrieval and LLM calls can be computationally expensive or slow.

---

# Protocol-Aware Extraction

A major challenge is that a scientific paper may contain multiple independent procedures.

For example:

```text
Paper
├── Gold seed synthesis
├── Gold bipyramid synthesis
├── Overgrowth
├── Kinetic study
└── Characterization
```

A flat result such as:

```json
{
  "temperature": 35,
  "reaction_time": 30
}
```

can lose critical context.

A more expressive representation is:

```text
Paper
├── Protocol 1
│   ├── reagents
│   ├── conditions
│   └── steps
│
├── Protocol 2
│   ├── reagents
│   ├── conditions
│   └── steps
│
└── Protocol 3
    ├── reagents
    ├── conditions
    └── steps
```

The long-term objective is to preserve:

```text
Entity
   ↓
Attribute
   ↓
Relationship
   ↓
Experimental Context
```

rather than extracting isolated facts.

---

# Example

A paper may contain:

```text
HAuCl4   → 25 mM
HAuCl4   → 50 mM
AgNO3    → 10 mM
NaBH4    → 25 mM

35 °C
80 °C

30 min
90 min
```

Detecting every value is not sufficient.

The system must determine which value belongs to which chemical and which experimental procedure.

This is why the project combines:

- canonical document representation
- retrieval
- multiple extraction strategies
- provenance
- normalization
- evaluation

---

# Research Workflow

```text
Papers
  ↓
Canonical Documents
  ↓
Gold Dataset
  ↓
┌─────────┬─────────┬─────────┐
│ Rules   │ NER     │ RAG/LLM │
└────┬────┴────┬────┴────┬────┘
     └─────────┼─────────┘
               ↓
          Evaluation
               ↓
        Error Analysis
               ↓
            MLflow
               ↓
         Improve System
               ↓
          Re-evaluate
```

This creates a reproducible research loop:

**Build → Measure → Analyze → Improve → Re-measure**

---

# Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| API | FastAPI |
| Background Jobs | Celery |
| Broker | Redis |
| PDF Processing | PyMuPDF |
| Scientific Parsing | GROBID |
| OCR | OCR fallback |
| Schemas | Pydantic |
| Database | PostgreSQL |
| Retrieval | Vector-store abstraction / pgvector |
| Extraction | Rules, NER, RAG + LLM |
| Experiment Tracking | MLflow |
| Containerization | Docker |
| Version Control | Git / GitHub |

---

# Research Questions

The architecture supports experiments around:

1. How accurately can experimental information be extracted from scientific papers?
2. How well can extracted values be associated with the correct entities?
3. How does retrieval quality affect LLM-based extraction?
4. Does RAG + LLM improve contextual extraction compared with deterministic baselines?
5. Can hybrid extraction improve robustness?
6. Can the complete extraction process be evaluated and reproduced systematically?

---

# Future Work

- Protocol-level extraction
- Improved provenance
- Chemical entity linking
- Unit normalization
- RAG-based extraction
- Schema-constrained LLM output
- Retrieval evaluation
- Gold annotation dataset
- Annotator agreement
- Relationship-level evaluation
- Unit-aware metrics
- MLflow benchmarking
- CSV/Excel export
- Human review interface
- Production monitoring

---

# Design Principles

### Structured
Define the target schema before extracting data.

### Contextual
Preserve relationships between entities, values and experimental procedures.

### Auditable
Maintain provenance back to the source document.

### Reproducible
Track datasets, models, configurations and metrics.

### Measurable
Evaluate extraction methods against reference annotations.

---

# Project Status

**Research / Active Development**

Sci-Extract is being developed as a modular research framework for scientific information extraction, with the architecture designed to support both experimental evaluation and eventual production deployment.
