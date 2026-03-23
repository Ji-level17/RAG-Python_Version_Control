<div align="center">

# Version-Aware RAG for FastAPI Code Generation

**A multi-strategy retrieval-augmented generation system for generating version-correct Python code**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B.svg)](streamlit_app.py)

**English** | [中文](README_zh.md)

</div>

---

## 📕 Table of Contents

- [What is this project?](#-what-is-this-project)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Retrieval Strategies](#-retrieval-strategies)
- [Get Started](#-get-started)
- [Running Experiments](#-running-experiments)
- [Results](#-results)
- [Models & Licenses](#-models--licenses)
- [Project Structure](#-project-structure)

---

## 💡 What is this project?

This project builds a **version-aware RAG pipeline** that generates correct FastAPI code for a specified Python version. FastAPI evolves rapidly across releases, and LLMs often generate outdated or version-incompatible code without access to version-specific documentation.

We compare **four retrieval strategies** — from simple dense vector search to import-graph-aware expansion — and evaluate them using [RAGAS](https://github.com/explodinggradients/ragas), [HumanEval](https://github.com/openai/human-eval), and [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) benchmarks.

---

## 🌟 Key Features

- **🔍 Four retrieval strategies** — Baseline Dense, Hybrid (BM25 + Dense + RRF), Hierarchical (parent-child grouping), Import-Graph (dependency expansion)
- **📌 Version-aware filtering** — Each chunk is tagged with `min_version` metadata for targeted retrieval
- **📊 Comprehensive evaluation** — RAGAS (4 metrics × 85 test cases × 4 strategies), HumanEval (164 problems), MBPP (257 problems)
- **🧪 Multi-run comparison** — 4 complete RAGAS evaluation runs with different configurations to ensure robustness
- **⚡ Interactive demo** — Streamlit UI for real-time strategy comparison
- **🔧 Reproducible** — Fixed random seeds, `temperature=0`, pre-computed results included

---

## 🔎 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Source Repository                  │
└──────────────────────────┬──────────────────────────────────┘
                           │ Tree-sitter AST Parsing
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Corpus (695 chunks, 768-dim embeddings)         │
│         Embedder: st-codesearch-distilroberta-base           │
└──────┬──────────┬────────────┬───────────┬──────────────────┘
       │          │            │           │
       ▼          ▼            ▼           ▼
  ┌─────────┐ ┌────────┐ ┌──────────┐ ┌───────────┐
  │Baseline │ │Hybrid  │ │Hierarch- │ │Import-    │
  │Dense    │ │BM25+   │ │ical      │ │Graph      │
  │         │ │Dense+  │ │Parent→   │ │Dependency │
  │Cosine   │ │RRF     │ │Child     │ │Expansion  │
  └────┬────┘ └───┬────┘ └────┬─────┘ └─────┬─────┘
       │          │            │             │
       └──────────┴─────┬──────┴─────────────┘
                        ▼
              ┌───────────────────┐
              │  Gemini 2.5 Flash │
              │  Code Generation  │
              └────────┬──────────┘
                       ▼
              ┌───────────────────┐
              │  RAGAS Evaluation │
              │  (LLM-as-Judge)  │
              └───────────────────┘
```

---

## 🎯 Retrieval Strategies

| # | Strategy | Description | Latency |
|:-:|----------|-------------|:-------:|
| 1 | **Baseline Dense RAG** | Cosine similarity on dense embeddings | ~0.3s |
| 2 | **Hybrid RAG** | BM25 + dense embeddings fused with Reciprocal Rank Fusion (k=60) | ~0.5s |
| 3 | **Hierarchical RAG** | Parent-child file grouping → parent retrieval → child re-ranking | ~3.9s |
| 4 | **Import-Graph RAG** | Dense retrieval + import dependency graph expansion → re-ranking | ~10.7s |

---

## 🎬 Get Started

### Prerequisites

- Python >= 3.10 (tested on 3.11, 3.13)
- [Pinecone](https://app.pinecone.io) API key (vector store)
- [Gemini](https://aistudio.google.com/apikey) API key (generation + evaluation)

### Installation

```bash
# Clone the repository
git clone https://github.com/Ji-level17/RAG-Python_Version_Control.git
cd RAG-Python_Version_Control
git checkout v2

# Create virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### Data Preparation

The corpus (`local_corpus.json`, 695 chunks) and import graph (`import_graph.json`) are **pre-built and included** in the repository. They were generated by running `repo_processor.py` over the [FastAPI source](https://github.com/tiangolo/fastapi) with Tree-sitter AST parsing.

To rebuild from scratch:

```bash
git clone https://github.com/tiangolo/fastapi.git
python repo_processor.py
python rebuild_corpus.py
python rebuild_import_graph.py
```

---

## 🚀 Running Experiments

### RAGAS Evaluation

Evaluates 4 strategies × 85 test cases on faithfulness, answer relevancy, context precision, and context recall:

```bash
python eval_rag_strategies.py \
    --gemini-key $GEMINI_API_KEY \
    --pinecone-key $PINECONE_API_KEY \
    --strategies baseline advanced hierarchical import_graph \
    --num-records 85 \
    --output-dir results_v9
```

### Retrieval Metrics

Measures hit rate, MRR, precision@k, and recall@k without an LLM judge:

```bash
python eval_retrieval_metrics.py \
    --pinecone-key $PINECONE_API_KEY \
    --output-dir results_v9
```

### Code Generation Benchmarks

Runs HumanEval (164 problems) and MBPP (257 problems) with pass@1:

```bash
python eval_humaneval_mbpp.py \
    --gemini-key $GEMINI_API_KEY \
    --output-dir results_v9
```

### Interactive Demo

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501` — select a strategy, enter a FastAPI question, and see retrieved context + generated answer.

---

## 📊 Results

### RAGAS Scores (Gemini 2.5 Flash, n=85, top-k=5)

| Strategy | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|----------|:-----------:|:----------------:|:-----------------:|:--------------:|
| Baseline Dense | **0.634** | 0.768 | 0.735 | 0.665 |
| Hybrid | 0.619 | 0.821 | 0.751 | **0.766** |
| Hierarchical | 0.479 | **0.836** | 0.678 | 0.611 |
| Import-Graph | 0.504 | 0.826 | **0.762** | 0.684 |

### Code Generation Benchmarks

| Benchmark | Problems | Pass@1 |
|-----------|:--------:|:------:|
| HumanEval | 164 | **72.6%** |
| MBPP | 257 | **72.4%** |

---

## 📜 Models & Licenses

| Component | Model | License |
|-----------|-------|---------|
| Code embedder | `flax-sentence-embeddings/st-codesearch-distilroberta-base` (768-dim) | Apache 2.0 |
| Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Apache 2.0 |
| QLoRA base model | `Qwen/Qwen2.5-Coder-7B` + adapter `Ivan17Ji/qwen-lora-250` | Apache 2.0 |
| Fine-tuning dataset | StarInstruct (checkpoint-25) | Apache 2.0 |
| Generation & judge | Gemini 2.5 Flash | Google ToS |
| Target codebase | [FastAPI](https://github.com/tiangolo/fastapi) | MIT |
| Benchmarks | HumanEval, MBPP | MIT |

---

## 📁 Project Structure

```
├── rag_pipeline.py              # Core RAG system (VersionControlRAG)
├── hierarchical_rag.py          # Strategy 3: Hierarchical retrieval
├── rag_strategies.py            # Strategy 4: Import-Graph retrieval
├── repo_processor.py            # FastAPI repo ingestion (AST chunking)
│
├── eval_rag_strategies.py       # RAGAS evaluation (4 strategies × 85 cases)
├── eval_retrieval_metrics.py    # Retrieval metrics (hit rate, MRR, P@k)
├── eval_humaneval_mbpp.py       # HumanEval & MBPP benchmarks
│
├── local_corpus.json            # 695 FastAPI code chunks
├── import_graph.json            # Module dependency graph
├── fastapi_golden_set.json      # Golden QA pairs
├── ragas_dataset.json           # RAGAS evaluation dataset
│
├── results_v9/                  # Evaluation results & charts
├── streamlit_app.py             # Interactive demo (4 strategies)
├── requirements.txt             # Python dependencies
└── .env                         # API keys (not tracked)
```

---

## 🔬 Reproducibility

| Setting | Value |
|---------|-------|
| Random seed | `42` |
| Generation temperature | `0` |
| RAGAS judge | Gemini 2.5 Flash (`thinking_budget=0`) |
| Test split | 85 records (from 561-pair golden set, 70/15/15 split) |
| Vector index | Pinecone `python-rag-v3` (768-dim, cosine) |

Pre-computed results are stored in `results_v9/` for verification without re-running experiments.

---

## ⚙️ Hardware Requirements

| Component | Requirement |
|-----------|------------|
| Embedder + BM25 | CPU only (~2 GB RAM) |
| Cross-encoder reranker | CPU only |
| Gemini API | No local GPU needed |
| QLoRA fine-tuning (optional) | 44+ GB VRAM (A40) |
