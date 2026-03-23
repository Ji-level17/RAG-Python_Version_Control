"""
Streamlit Demo: Version-Aware RAG for FastAPI Code Generation
Run with: streamlit run streamlit_app.py
"""

import os
import json
import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
PINECONE_KEY = os.environ.get("PINECONE_API_KEY", "")
GEMINI_KEY   = os.environ.get("GEMINI_API_KEY", "")

st.set_page_config(page_title="Version-Aware RAG", layout="wide")
st.title("Version-Aware RAG for FastAPI Code Generation")
st.caption("Compare 4 retrieval strategies: Baseline Dense, Hybrid, Hierarchical, Import-Graph")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    strategy = st.selectbox("Retrieval Strategy", [
        "Baseline Dense RAG",
        "Hybrid RAG",
        "Hierarchical RAG",
        "Import-Graph RAG",
    ])
    target_version = st.text_input("Target Python version", value="3.11")
    top_k = st.slider("Top-K chunks", min_value=1, max_value=10, value=3)
    corpus_file = st.text_input("Corpus file", value="local_corpus.json")
    graph_file  = st.text_input("Import graph file", value="import_graph.json")

# ── Gemini client ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising Gemini client...")
def get_gemini_client(api_key):
    from google import genai as _genai

    class _GeminiClient:
        def __init__(self, key):
            self._client = _genai.Client(api_key=key)
            self.chat = self
            self.completions = self

        def create(self, model, messages, max_tokens=512, temperature=0, **_):
            prompt = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages
            )
            resp = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=_genai.types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    thinking_config=_genai.types.ThinkingConfig(thinking_budget=0),
                ),
            )
            text = resp.text if resp.text is not None else ""

            class _Msg:
                content = text
            class _Choice:
                message = _Msg()
            class _Resp:
                choices = [_Choice()]
            return _Resp()

    return _GeminiClient(api_key)


# ── Load RAG system (cached) ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading RAG system...")
def load_rag_system(corpus_path, graph_path):
    from rag_pipeline import VersionControlRAG
    from hierarchical_rag import HierarchicalRAG
    from rag_strategies import ImportGraphRAG

    rag = ImportGraphRAG(pinecone_key=PINECONE_KEY, ingest_only=True)
    rag.load_local_corpus(corpus_path)
    if os.path.exists(graph_path):
        rag.load_import_graph(graph_path)

    hier_rag = HierarchicalRAG(pinecone_key=PINECONE_KEY, ingest_only=True)
    hier_rag.load_local_corpus(corpus_path)

    return rag, hier_rag


if not GEMINI_KEY:
    st.error("Set GEMINI_API_KEY in your .env file.")
    st.stop()

if not os.path.exists(corpus_file):
    st.warning(f"Corpus file `{corpus_file}` not found.")
    st.stop()

rag, hier_rag = load_rag_system(corpus_file, graph_file)
gemini = get_gemini_client(GEMINI_KEY)

# ── Stats ────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Corpus chunks", len(rag.local_corpus))
col2.metric("Import-graph nodes", len(rag.import_graph))
col3.metric("Hierarchical parents", len(hier_rag.parent_groups))

st.divider()

# ── Query ────────────────────────────────────────────────────────────────────
query = st.text_input(
    "Ask a FastAPI question",
    placeholder="e.g. How to use Depends for shared query parameters in FastAPI 0.100+?",
)

if st.button("Retrieve & Generate", type="primary") and query:
    with st.spinner("Retrieving context..."):
        parsed_version = rag._parse_version(target_version) if target_version else None
        strategy_key = strategy.split()[0].lower()  # baseline / hybrid / hierarchical / import

        # ── Retrieval ────────────────────────────────────────────────
        if strategy_key == "baseline":
            contexts = rag.retrieve_complex(
                query, target_version=target_version, top_k=top_k, mode="baseline",
            )
        elif strategy_key == "hybrid":
            contexts = rag.retrieve_complex(
                query, target_version=target_version, top_k=top_k, mode="advanced",
            )
        elif strategy_key == "hierarchical":
            contexts = hier_rag.retrieve_complex(
                query, target_version=target_version, top_k=top_k, mode="advanced",
            )
        else:  # import-graph
            contexts = rag.retrieve_complex(
                query, target_version=target_version, top_k=top_k, mode="graph",
            )

    with st.spinner("Generating answer with Gemini 2.5 Flash..."):
        context_block = "\n\n---\n\n".join(contexts)
        prompt = (
            f"You are a Python/FastAPI expert. Target version: {target_version}.\n"
            "Use ONLY the provided context to answer. "
            "If the answer is not in the context, say 'Not found in context'.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        response = gemini.create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.2,
        )
        answer = response.choices[0].message.content

    # ── Display results ──────────────────────────────────────────────────
    st.subheader("Answer")
    st.write(answer)

    st.subheader(f"Retrieved Context ({len(contexts)} chunks)")
    for i, code in enumerate(contexts):
        with st.expander(f"Chunk {i+1} ({len(code)} chars)"):
            st.code(code, language="python")

    st.subheader("Strategy Details")
    strategy_info = {
        "Baseline Dense RAG": "Pure cosine similarity on dense embeddings (st-codesearch-distilroberta-base).",
        "Hybrid RAG": "BM25 + dense embeddings fused with Reciprocal Rank Fusion (k=60).",
        "Hierarchical RAG": "Parent-child file grouping → parent retrieval → child re-ranking.",
        "Import-Graph RAG": "Dense retrieval + import dependency graph expansion → re-ranking.",
    }
    st.info(strategy_info[strategy])
