# 🎯 Production RAG Evaluation Suite

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![DeepEval](https://img.shields.io/badge/evaluation-DeepEval-orange.svg)](https://github.com/confident-ai/deepeval)
[![LangChain](https://img.shields.io/badge/orchestration-LangChain-1C3C3C.svg)](https://python.langchain.com/)
[![Chroma](https://img.shields.io/badge/vector_store-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-grade **Retrieval-Augmented Generation (RAG)** system paired with a comprehensive **component-level and system-level evaluation framework**.

Built on the principle of **isolated component diagnostics** (RAG Triad) and **custom application metrics (G-Eval)**, this repository demonstrates how to systematically evaluate, benchmark, and improve retrieval quality, reranking gains, grounded generation, and end-to-end response quality against golden reference datasets.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture & Design](#-architecture--design)
- [Evaluation Strategy](#-evaluation-strategy)
  - [1. Retriever Evaluation (Baseline vs. Reranked)](#1-retriever-evaluation-baseline-vs-reranked)
  - [2. Generator Evaluation (Strict Isolation)](#2-generator-evaluation-strict-isolation)
  - [3. End-to-End Application Quality (G-Eval)](#3-end-to-end-application-quality-g-eval)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Configuration](#environment-configuration)
- [Usage & Execution](#-usage--execution)
  - [Running the RAG Pipeline](#running-the-rag-pipeline)
  - [Running the Evaluations](#running-the-evaluations)
  - [Generating Synthetic Goldens](#generating-synthetic-goldens)
- [Tech Stack](#-tech-stack)
- [Observability](#-observability)
- [License](#-license)

---

## 🚀 Overview

Evaluating RAG applications is challenging because errors compound across stages:
* Did the **retriever** fail to fetch relevant context?
* Did the **retriever** include too much noise that distracted the model?
* Or did the **generator** hallucinate despite having the right context?

This project breaks down RAG evaluation into **isolated diagnostic layers** rather than treating the system as an opaque black box:

```
[Raw User Query]
       │
       ▼
 ┌─────────────┐       Contextual Recall & Precision
 │  Retriever  │ ────► (Did we fetch all necessary context?
 └──────┬──────┘        Are relevant chunks ranked at the top?)
        │ Over-retrieve (fetch_k=10)
        ▼
 ┌─────────────┐       Cross-Encoder Reranking
 │  Reranker   │ ────► (Benchmarking bi-encoder vs. cross-encoder)
 └──────┬──────┘
        │ Top-k (k=5)
        ▼
 ┌─────────────┐       Faithfulness & Answer Relevancy
 │  Generator  │ ────► (Evaluated on known-good golden context
 └──────┬──────┘        to isolate generator failure from retriever error)
        │
        ▼
 ┌─────────────┐       G-Eval (Rubric-Guided LLM-as-a-Judge)
 │ Final Output│ ────► Correctness (Truth)
 └─────────────┘       Completeness (Coverage)
                       Style & Tone (Pedagogical clarity)
```

---

## 🏗 Architecture & Design

The application serves as a teaching assistant for an LLM Evaluation curriculum, indexed across multi-session lecture transcripts:

* **Ingestion & Indexing (`src/retriever.py`)**:
  - Parses VTT lecture transcripts, strips timestamp boilerplate, and annotates text with session metadata.
  - Chunks text using `RecursiveCharacterTextSplitter` (chunk size: `1000`, overlap: `150`).
  - Computes dense vector embeddings using OpenAI's `text-embedding-3-small`.
  - Persists embeddings locally in a **Chroma** vector database (`chroma_store/`).

* **Two-Stage Retrieval (`src/reranker.py`)**:
  - **Stage 1 (Bi-Encoder Over-Retrieval)**: Chroma performs vector similarity search to quickly fetch candidate chunks (`fetch_k=10`).
  - **Stage 2 (Neural Cross-Encoder Reranking)**: Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to score `(query, document)` pairs jointly and rerank down to the top survivors (`top_k=5`).

* **Faithfulness-First Generator (`src/generator.py`)**:
  - Powered by `gpt-4o-mini` with strict system prompting.
  - Strict grounding: Instructed to answer **only** from retrieved context, avoid external assumptions, and abstain explicitly if the context is insufficient.
  - Anti-jailbreak and privacy guardrails: Untrusted input demarcation, credential scrubbing, and protection against system prompt leakage.
  - Supports both batch generation (`generate`) and token streaming (`generate_stream`) to measure time-to-first-token (TTFT).

* **Pipeline Orchestrator (`src/rag_pipeline.py`)**:
  - Unifies retrieval, reranking, and generation into a clean, callable interface (`RagPipeline.invoke`).
  - Fully instrumented with `@traceable` for LangSmith observability.

---

## 📊 Evaluation Strategy

The project implements the **RAG Triad** and **G-Eval** using [DeepEval](https://github.com/confident-ai/deepeval):

### 1. Retriever Evaluation (Baseline vs. Reranked)
- **`evals/eval_retriever.py`** evaluates baseline Chroma vector search against `golden/retriever_golden.json`.
- **`evals/eval_retriever_with_reranker.py`** evaluates the two-stage retriever with the cross-encoder reranker.
- **Metrics**:
  - **Contextual Recall**: Did the retriever find all the information needed to answer the question?
  - **Contextual Precision**: Are the most relevant chunks positioned at the top of the context window?

### 2. Generator Evaluation (Strict Isolation)
- **`evals/eval_generator.py`** isolates the generator by feeding it **known-good golden contexts** (`golden/faithfulness_dataset.json`), *not* the retriever's dynamic output.
- **Why Isolation?** If an answer is inaccurate, testing in isolation guarantees whether the generator made a reasoning/hallucination error versus whether it was simply given bad context by the retriever.
- **Metrics**:
  - **Faithfulness**: Deconstructs actual output into atomic claims and verifies whether each claim is supported by the context.
  - **Answer Relevancy**: Assesses how directly the output addresses the input prompt.

### 3. End-to-End Application Quality (G-Eval)
- **`evals/eval_application.py`** evaluates the live end-to-end pipeline with custom `GEval` rubrics:
  - **Correctness (Truth)**: Compares factual claims against golden answers. Truth is judged independently from length or coverage (brief, accurate answers are rewarded; omissions are not penalized here).
  - **Completeness (Coverage)**: Judges whether all key concepts from the reference answer were addressed.
  - **Style & Pedagogical Tone (Reference-Free)**: Assesses conversational flow, plain-English explanations before jargon, and explanatory clarity without needing a reference answer.

---

## 📁 Repository Structure

```text
Rag-eval/
├── data/                               # Raw VTT lecture transcripts (Sessions 1-8)
│   ├── YT Sandbox   LLM Evals Session 1.vtt
│   └── ...
├── src/                                # Core RAG application code
│   ├── __init__.py
│   ├── retriever.py                    # VTT loader, splitter, Chroma store builder
│   ├── reranker.py                     # Cross-encoder neural reranker (MS-MARCO)
│   ├── generator.py                    # Prompt-engineered grounded generator (gpt-4o-mini)
│   └── rag_pipeline.py                 # Unified pipeline orchestrator (LangSmith traced)
├── evals/                              # DeepEval evaluation suite
│   ├── __init__.py
│   ├── harness.py                      # Dataset loaders, metric aggregators & summary report
│   ├── eval_retriever.py               # Baseline retriever evaluation
│   ├── eval_retriever_with_reranker.py # Two-stage reranked retriever evaluation
│   ├── eval_generator.py               # Generator evaluation in strict isolation
│   └── eval_application.py             # End-to-end G-Eval application metrics
├── golden/                             # Curated & synthetic test sets
│   ├── correctness_goldens.json        # End-to-end Q&A golden set
│   ├── faithfulness_dataset.json       # Query + ideal context chunks for generator eval
│   ├── retriever_golden.json           # Fixed truth queries + expected answers for retrieval
│   ├── retriever_deepeval_goldens.json # Synthetic evaluation datasets
│   └── golden_generator.py             # DeepEval Synthesizer pipeline for auto-generating goldens
├── resources/                          # Educational guides & intro scripts
│   └── deepeval_intro.py               # Minimal reproducible DeepEval starter
├── .env.example                        # Environment variable configuration template
├── pyproject.toml                      # Project metadata and dependencies
└── uv.lock                             # Pinned reproducible dependency lockfile
```

---

## ⚡ Getting Started

### Prerequisites

- **Python 3.11+**
- **OpenAI API Key** (for embeddings, generation, and LLM-as-a-judge evaluation)
- **uv** (recommended for ultra-fast, reproducible dependency management)

### Installation

Clone the repository and install dependencies:

#### Using `uv` (Recommended)
```bash
git clone https://github.com/nish0753/rag_eval.git
cd rag_eval

# Create virtual environment and sync dependencies from uv.lock
uv sync
```

#### Using `venv` & `pip`
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

### Environment Configuration

Copy the sample environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
OPENAI_API_KEY="sk-..."
OPENAI_MODEL="gpt-4o-mini"
MODEL="gpt-4o-mini"

# Optional: Enable LangSmith tracing
# LANGCHAIN_TRACING_V2="true"
# LANGCHAIN_API_KEY="your-langsmith-key"
# LANGCHAIN_PROJECT="rag-eval"
```

---

## 💻 Usage & Execution

### Running the RAG Pipeline

#### 1. Ingest Data & Build Chroma Vector Store
To parse the VTT transcripts in `data/`, generate embeddings, and build the persistent Chroma store:
```bash
uv run python -m src.retriever
```

#### 2. Run a Pipeline Smoke Test
Test the full Retrieve ➔ Rerank ➔ Generate chain with a sample query:
```bash
uv run python -m src.rag_pipeline
```

---

### Running the Evaluations

All evaluation scripts can be run individually using `uv run` (or within an activated virtual environment using `python -m <module>`):

#### 1. Evaluate Baseline Vector Retriever
Measures Contextual Recall and Precision on the baseline Chroma index:
```bash
uv run python -m evals.eval_retriever
```

#### 2. Evaluate Retriever with Cross-Encoder Reranker
Evaluates the two-stage retrieval pipeline to quantify reranking improvements:
```bash
uv run python -m evals.eval_retriever_with_reranker
```

#### 3. Evaluate Generator in Isolation
Tests hallucination resistance and faithfulness against known-good context chunks:
```bash
uv run python -m evals.eval_generator
```

#### 4. Evaluate End-to-End Application Quality
Scores live application responses across **Correctness**, **Completeness**, and **Style**:
```bash
uv run python -m evals.eval_application
```

---

### Generating Synthetic Goldens

Need more test cases? `golden/golden_generator.py` uses DeepEval's `Synthesizer` to autonomously extract substantive transcript chunks, generate grounded queries and ideal answers, and format them into evaluation-ready JSON datasets:

```bash
uv run python -m golden.golden_generator
```

---

## 🛠 Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Orchestration** | [LangChain](https://python.langchain.com/) | Prompt management, document loading, and LCEL chains |
| **Embeddings** | [OpenAI `text-embedding-3-small`](https://platform.openai.com/docs/guides/embeddings) | Dense vector representations for transcript chunks |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) | Persistent vector store with local SQLite backend |
| **Neural Reranker** | [SentenceTransformers](https://sbert.net/) (`ms-marco-MiniLM-L-6-v2`) | Cross-encoder for reranking candidate documents |
| **Generation Model** | [OpenAI `gpt-4o-mini`](https://platform.openai.com/) | Cost-efficient, high-reasoning LLM generator |
| **Evaluation Suite** | [DeepEval](https://github.com/confident-ai/deepeval) | RAG Triad, G-Eval rubrics, and synthetic data generation |
| **Dependency Tooling**| [uv](https://docs.astral.sh/uv/) | Fast Python package management and deterministic lockfile |
| **Observability** | [LangSmith](https://smith.langchain.com/) | Traceability for retriever, reranker, and generator stages |

---

## 🔍 Observability

This project integrates with **LangSmith** for end-to-end tracing. Set the following in your `.env`:

```env
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="your-langsmith-api-key"
LANGCHAIN_PROJECT="rag-eval"
```

Every invocation logs the full trace hierarchy:
- `RagPipeline` (root chain)
  - `RerankingRetriever` (retrieval + cross-encoder scores)
  - `ChatOpenAI` (generation tokens, latency, and prompt payload)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
