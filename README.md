# FastAPI QLoRA RAG Evaluation System

A complete pipeline for ingesting the FastAPI source repository, evaluating a
QLoRA fine-tuned code generation model, and benchmarking it against frontier
models using RAGAS.

---

## Project structure

```
project/
│
├── rag_pipeline_fixed.py          Core RAG system (VersionControlRAG class)
├── repoProcessor_fixed.py         Repository ingestion with tier-aware chunking
│
├── local_corpus.json              889 FastAPI code chunks (scripts/ excluded)
├── processed_state.json           MD5 hash registry — prevents re-ingesting unchanged files
│
├── fastapi_golden_set.jsonl       561 golden QA pairs  (instruction / ground_truth / version)
├── fastapi_golden_set.json        Same set, pretty-printed for inspection
├── fastapi_golden_set_manifest.csv  Lightweight index — no code, no answers
├── fastapi_golden_set_splits.json   Train(392) / Val(84) / Test(85) split
│
├── ragas_dataset.jsonl            576 RAGAS records (question / contexts / ground_truth / answer)
├── ragas_dataset.json             Same, pretty-printed
├── ragas_dataset_hf.csv           HuggingFace Dataset-compatible format
├── ragas_eval_template.py         Standalone RAGAS eval script (reference)
│
├── validate_datasets.py           Runs 17 automated checks on the datasets
│
├── corpus_bootstrap.ipynb         Restore local_corpus.json after kernel restart
├── oom_safe_upsert.ipynb          Re-upsert to Pinecone without OOM (ingest_only mode)
├── recover_corpus.ipynb           Rebuild local_corpus.json from Pinecone if overwritten
│
├── fastapi_qlora_eval.ipynb       Main eval notebook (retrieval + generation + RAGAS)
├── qlora_ragas_eval.ipynb         Standalone QLoRA RAGAS eval — base vs QLoRA
├── groq_eval.ipynb                Frontier eval via Groq (Llama-3.3-70b, Qwen-32B)
├── model_comparison.ipynb         Combined comparison chart (loads from both above)
│
├── retrieval_chunking_eval.ipynb  Chunking strategy A/B test (hit rate / MRR / P@k / recall)
│
├── ragas_rag_pipeline_eval.ipynb  Full RAG pipeline eval (live retriever + generator)
├── ragas_nan_fix.ipynb            Fix: all-NaN RAGAS scores (empty answers / missing judge)
├── ragas_empty_answer_fix.ipynb   Fix: empty model answers (Qwen chat template fix)
│
├── fix_dimension_mismatch.ipynb   Fix: Pinecone 384 vs 768 dim error
├── definitive_fix.ipynb           Fix: semantic hit metric + force re-upsert
├── complete_fix.ipynb             Fix: combined re-upsert + all modes test
```

---

## System overview

### Phase 1 — Corpus ingestion

`repoProcessor_fixed.py` walks the FastAPI repository and calls
`VersionControlRAG.ingest_data()` for each Python function found via Tree-sitter
AST parsing. Each chunk is assigned a **tier**:

| Tier | Source | Kept? |
|---|---|---|
| `docs_src` | Tutorial examples (`docs_src/`) | ✅ 628 chunks |
| `fastapi` | Internal implementation (`fastapi/`) | ✅ 261 chunks |
| `scripts` | Build tooling (`scripts/`) | ❌ excluded — noise |

The embedder (`st-codesearch-distilroberta-base`, 768-dim) encodes each chunk
on CPU. English summaries are prepended to the code before embedding; Chinese
summaries (produced by the QLoRA during ingestion) are stripped and only the
code is embedded.

### Phase 2 — Golden set

561 QA pairs generated from the corpus, split 70/15/15. Each item has:
- `instruction` — natural-language prompt
- `ground_truth` — the correct Python code
- `version` — `{python, style, fastapi_min_version}`

The RAGAS dataset (`ragas_dataset.jsonl`) wraps these into the four fields
RAGAS expects: `question`, `contexts`, `ground_truth`, `answer` (blank — filled
by your model at eval time).

### Phase 3 — RAG pipeline (`rag_pipeline_fixed.py`)

`VersionControlRAG` has three retrieval modes:

| Mode | Description |
|---|---|
| `baseline` | Dense vector search only |
| `advanced` | Hybrid (vector + BM25) with BGE cross-encoder reranking |
| `hyde` | HyDE: generate a hypothetical answer → embed that → hybrid + rerank |

**Stratified retrieval:** concept queries (no backtick-quoted function name)
search only `docs_src/` tier to avoid internal implementation noise. Queries
that explicitly name an internal function (e.g. `` `_extract_form` ``) search
all tiers.

**Memory modes:**

```python
# Ingestion only — no Qwen on GPU (~0.3 GB VRAM)
rag = VersionControlRAG(..., ingest_only=True)

# Full pipeline — Qwen + adapter on GPU (~15.6 GB VRAM)
rag = VersionControlRAG(..., ingest_only=False)

# Free GPU mid-session
rag.free_gpu_memory()
```

### Phase 4 — Retrieval evaluation

`retrieval_chunking_eval.ipynb` measures retrieval quality without any LLM
judge:

| Metric | Definition |
|---|---|
| `hit_rate@k` | 1 if any golden chunk appears in top-k results |
| `mrr@k` | 1 / rank of first golden hit |
| `precision@k` | Relevant retrieved / total retrieved |
| `recall@k` | Relevant retrieved / total golden |

**Current best:** `advanced` mode at `top_k=10` → **70% semantic hit rate**
on the test split.

**Semantic hit metric:** a retrieved chunk counts as a hit if it defines the
same function name as a golden chunk, not just if it is the exact same file.
This correctly treats tutorial variants (`tutorial001_py310.py` vs
`tutorial001_an_py310.py`) as equivalent correct answers.

### Phase 5 — Generation evaluation (RAGAS)

RAGAS scores four dimensions using an LLM judge (Groq `llama-3.3-70b` — free):

| Metric | Question it answers |
|---|---|
| `faithfulness` | Is the answer grounded in the retrieved context? |
| `answer_relevancy` | Does the answer address the question? |
| `context_precision` | Are the retrieved chunks relevant to the question? |
| `context_recall` | Do the chunks cover what is needed to answer? |

**Model comparison:**
1. Run `qlora_ragas_eval.ipynb` → saves `model_comparison_results.csv`
2. Run `groq_eval.ipynb` → loads above + runs frontier models → full chart

---

## Hardware requirements

| Component | Requirement |
|---|---|
| GPU | 44+ GB VRAM (tested on A40) |
| Qwen 7B + QLoRA adapter | ~15.6 GB VRAM |
| Embedder + reranker | CPU only |
| Frontier eval (Groq) | Zero GPU — API only |

---

## Keys required

| Key | Where to get | Used for |
|---|---|---|
| `PINECONE_API_KEY` | app.pinecone.io | Vector store |
| `GROQ_API_KEY` | console.groq.com/keys | Generation + RAGAS judge (free) |

---

## Quick start — kernel restart recovery

```python
# Every time you restart the kernel, run this first:
import rag_pipeline, importlib
importlib.reload(rag_pipeline)
from rag_pipeline import VersionControlRAG

rag = VersionControlRAG(
    pinecone_key = "YOUR_KEY",
    model_path   = "Qwen/Qwen2.5-Coder-7B",
    adapter_path = "Ivan17Ji/qwen-lora-250",
    ingest_only  = False,
)
rag.load_local_corpus("local_corpus.json")
```

If `local_corpus.json` is empty or missing, run `recover_corpus.ipynb` — it
reconstructs the file from Pinecone metadata without re-embedding anything.

---

## Tomorrow — chunking experiments

`retrieval_chunking_eval.ipynb` already has four strategies implemented.
To test a new strategy, add it to the `strategies` dict in Section 1:

```python
strategies = {
    "ast_function":   chunks_ast,       # current baseline
    "ast_with_class": chunks_ast_ctx,   # adds parent class name
    "fixed_512":      chunks_fixed,     # 512-token windows
    "semantic":       chunks_semantic,  # embed + cluster similar fns
    "your_new_method": your_chunks,     # ← add here
}
```

The eval loop runs all strategies automatically and produces hit rate / MRR /
precision / recall charts plus a recall@k curve showing the optimal `top_k`.

---

## Known limitations

- `min_version` metadata is unreliable — QLoRA assigns 3.12 to ~99% of chunks,
  so the version filter in `retrieve_complex` is disabled.
- 60% of test queries are concept-only (no function name in the query). These
  require HyDE for good recall; pure vector search plateaus at ~30%.
- The golden set contains tutorial variants of the same function across
  multiple files. Use `hit_rate_semantic()` (function-name matching) not
  `hit_rate_strict()` (exact fingerprint) for meaningful evaluation.
