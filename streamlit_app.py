"""
Streamlit Demo: Import-Graph Expansion RAG
Run with: streamlit run streamlit_app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from rag_strategies import ImportGraphRAG

load_dotenv()
PINECONE_KEY = os.environ.get("PINECONE_API_KEY", "")
GROQ_KEY     = os.environ.get("GROQ_API_KEY", "")

st.set_page_config(page_title="Import-Graph RAG", layout="wide")
st.title("Import-Graph Expansion RAG")
st.caption("Strategy 3: Augments dense retrieval with import dependency context")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    target_version = "3.11"
    top_k = st.slider("Top-K chunks", min_value=1, max_value=10, value=3)
    corpus_file = st.text_input("Corpus file", value="local_corpus.json")
    graph_file  = st.text_input("Import graph file", value="import_graph.json")

# ── Load RAG (cached) ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading RAG system...")
def load_rag(corpus_path, graph_path):
    rag = ImportGraphRAG(pinecone_key=PINECONE_KEY, groq_api_key=GROQ_KEY)
    rag.load_local_corpus(corpus_path)
    rag.load_import_graph(graph_path)
    return rag

if not os.path.exists(corpus_file) or not os.path.exists(graph_file):
    st.warning("Corpus or import graph not found. Run `demo_import_graph.py --rebuild` first.")
    st.stop()

rag = load_rag(corpus_file, graph_file)

# ── Graph stats ───────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Documents", len(rag.local_corpus))
col2.metric("Graph nodes", len(rag.import_graph))
col3.metric("Graph edges", sum(len(v) for v in rag.import_graph.values()))

st.divider()

# ── Query ─────────────────────────────────────────────────────────────────────
query = st.text_input("Ask a question about the codebase",
                      placeholder="e.g. How does BaseModel validation change from v1 to v2?")

if st.button("Search & Answer", type="primary") and query:
    with st.spinner("Retrieving and generating answer..."):

        # --- retrieval (same logic as generate_answer but expose labels) ---
        import re
        query_vector = rag.embedder.encode(query).tolist()
        parsed_version = rag._parse_version(target_version)
        filter_dict = {"min_version": {"$lte": parsed_version}}

        fetch_k = max(top_k * 3, 10)
        pinecone_res = rag.index.query(
            vector=query_vector, filter=filter_dict,
            top_k=fetch_k, include_metadata=True,
        )
        initial  = {m['id']: m['metadata']['code'] for m in pinecone_res['matches']}
        local_map = {doc["id"]: doc["content"] for doc in rag.local_corpus
                     if doc["metadata"]["min_version"] <= parsed_version}
        expanded = {}
        for doc_id in list(initial.keys()):
            for rid in rag.import_graph.get(doc_id, []):
                if rid not in initial and rid in local_map:
                    expanded[rid] = local_map[rid]
                if len(expanded) >= 20:
                    break
            if len(expanded) >= 20:
                break

        candidates = {**initial, **expanded}
        pairs  = [[query, code] for code in candidates.values()]
        scores = rag.reranker.predict(pairs)
        ranked = sorted(zip(candidates.items(), scores), key=lambda x: x[1], reverse=True)
        top_items = ranked[:top_k]

        # --- generate answer ---
        context_parts = []
        for (doc_id, code), _ in top_items:
            label = "DIRECT HIT" if doc_id in initial else "GRAPH-EXPANDED"
            context_parts.append(f"[{label}]\n{code}")

        prompt = (
            f"You are a Python migration expert (target Python version: {target_version}).\n"
            "The context includes directly relevant code AND code from files it imports "
            "(marked GRAPH-EXPANDED), which may be affected by breaking API changes.\n"
            "Use ONLY the provided context to answer. If not found, say 'Not found in context'.\n\n"
            "Context:\n" + "\n\n---\n\n".join(context_parts) +
            f"\n\nQuestion: {query}\n\nAnswer:"
        )
        response = rag.llm.chat.completions.create(
            model=rag.groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512, temperature=0.2,
        )
        answer = response.choices[0].message.content

    # ── Results ───────────────────────────────────────────────────────────────
    st.subheader("Answer")
    st.write(answer)

    st.subheader(f"Retrieved Chunks ({len(initial)} direct + {len(expanded)} graph-expanded)")
    for (doc_id, code), score in top_items:
        label = "DIRECT HIT" if doc_id in initial else "GRAPH-EXPANDED"
        color = "🟢" if doc_id in initial else "🔵"
        with st.expander(f"{color} [{label}]  {doc_id}  (score: {score:.3f})"):
            st.code(code, language="python")
