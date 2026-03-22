"""
LangChain / LangGraph Integration
===================================
Wraps the existing RAG pipeline into LangChain-compatible components.

Two implementations:
  1. LangChain LCEL chain  — retriever → prompt → Groq LLM
  2. LangGraph pipeline    — StateGraph with retrieve / generate / fallback nodes

Usage:
    from langchain_rag import build_lcel_chain, build_langgraph_pipeline
    from rag_pipeline import VersionControlRAG

    rag = VersionControlRAG(pinecone_key=..., ingest_only=True)
    rag.load_local_corpus("local_corpus.json")

    # LangChain LCEL
    chain = build_lcel_chain(rag, groq_api_key=..., strategy="advanced")
    result = chain.invoke({"question": "How to create a FastAPI endpoint?"})

    # LangGraph
    pipeline = build_langgraph_pipeline(rag, groq_api_key=..., strategy="advanced")
    result = pipeline.invoke({"question": "How to use Pydantic v2 with FastAPI?"})
"""

import os
from typing import List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from operator import itemgetter
from pydantic import ConfigDict, Field

load_dotenv()


# ─────────────────────────────────────────────────────────────
# LangChain Retriever wrapper
# ─────────────────────────────────────────────────────────────

class VersionControlRetriever(BaseRetriever):
    """
    LangChain BaseRetriever wrapping VersionControlRAG (and subclasses).

    Supports all four strategies by setting `strategy`:
      "baseline"    — dense vector search only
      "advanced"    — hybrid BM25 + dense + rerank
      "hierarchical"— parent-group selection → child rerank
      "import_graph"— dense + import-dependency graph expansion
    """

    rag: object = Field(description="VersionControlRAG instance (or subclass)")
    strategy: str = Field(default="advanced")
    top_k: int = Field(default=5)
    target_version: Optional[str] = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        mode_map = {
            "baseline":     "baseline",
            "advanced":     "advanced",
            "hierarchical": "advanced",   # HierarchicalRAG.retrieve_complex uses mode="advanced"
            "import_graph": "graph",
        }
        mode = mode_map.get(self.strategy, "advanced")

        try:
            chunks = self.rag.retrieve_complex(
                query,
                target_version=self.target_version,
                top_k=self.top_k,
                mode=mode,
            )
        except Exception as e:
            print(f"[VersionControlRetriever] retrieval error: {e}")
            chunks = []

        return [
            Document(page_content=chunk, metadata={"strategy": self.strategy})
            for chunk in chunks
        ]


# ─────────────────────────────────────────────────────────────
# LangChain LCEL chain
# ─────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system",
     "You are a FastAPI expert. Use ONLY the provided code context to answer. "
     "Return valid Python code. If the answer is not in the context, say "
     "'Not found in context'."),
    ("human",
     "### Context\n{context}\n\n### Question\n{question}\n\n### Answer"),
])


def _format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def build_lcel_chain(rag, groq_api_key: str, strategy: str = "advanced",
                     top_k: int = 5, target_version: str = None,
                     model: str = "llama-3.3-70b-versatile"):
    """
    Build a LangChain LCEL retrieval chain.

    Returns a runnable that accepts {"question": str} and returns a str answer.

    Example:
        chain = build_lcel_chain(rag, groq_api_key=GROQ_KEY)
        answer = chain.invoke({"question": "How to add OAuth2 to FastAPI?"})
    """
    retriever = VersionControlRetriever(
        rag=rag,
        strategy=strategy,
        top_k=top_k,
        target_version=target_version,
    )
    llm = ChatGroq(api_key=groq_api_key, model=model, temperature=0)

    chain = (
        {
            "context":  itemgetter("question") | retriever | RunnableLambda(_format_docs),
            "question": itemgetter("question"),
        }
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )
    return chain


# ─────────────────────────────────────────────────────────────
# LangGraph pipeline
# ─────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    question: str
    contexts: List[str]
    answer: str
    retrieval_ok: bool


def build_langgraph_pipeline(rag, groq_api_key: str, strategy: str = "advanced",
                              top_k: int = 5, target_version: str = None,
                              model: str = "llama-3.3-70b-versatile"):
    """
    Build a LangGraph RAG pipeline with three nodes:
      retrieve  → generate  → END
                ↘ fallback  → END  (if retrieval returns nothing)

    Returns a compiled graph callable.
    Accept {"question": str}, returns full RAGState dict.

    Example:
        pipeline = build_langgraph_pipeline(rag, groq_api_key=GROQ_KEY)
        result = pipeline.invoke({"question": "..."})
        print(result["answer"])
        print(result["contexts"])
    """
    llm = ChatGroq(api_key=groq_api_key, model=model, temperature=0)

    mode_map = {
        "baseline":     "baseline",
        "advanced":     "advanced",
        "hierarchical": "advanced",
        "import_graph": "graph",
    }
    mode = mode_map.get(strategy, "advanced")

    # ── Node: retrieve ──────────────────────────────────────────
    def retrieve_node(state: RAGState) -> RAGState:
        try:
            chunks = rag.retrieve_complex(
                state["question"],
                target_version=target_version,
                top_k=top_k,
                mode=mode,
            )
            retrieval_ok = bool(chunks)
        except Exception as e:
            print(f"[LangGraph:retrieve] error: {e}")
            chunks = []
            retrieval_ok = False
        return {**state, "contexts": chunks, "retrieval_ok": retrieval_ok}

    # ── Node: generate ──────────────────────────────────────────
    def generate_node(state: RAGState) -> RAGState:
        context = "\n\n---\n\n".join(state["contexts"][:5])
        messages = PROMPT_TEMPLATE.format_messages(
            context=context, question=state["question"]
        )
        response = llm.invoke(messages)
        return {**state, "answer": response.content}

    # ── Node: fallback ──────────────────────────────────────────
    def fallback_node(state: RAGState) -> RAGState:
        print("[LangGraph:fallback] No contexts retrieved — answering without RAG.")
        messages = [
            ("system", "You are a FastAPI expert. Answer from your own knowledge."),
            ("human", state["question"]),
        ]
        response = llm.invoke(messages)
        return {**state, "answer": f"[NO CONTEXT] {response.content}"}

    # ── Routing ─────────────────────────────────────────────────
    def route_after_retrieve(state: RAGState) -> str:
        return "generate" if state["retrieval_ok"] else "fallback"

    # ── Build graph ─────────────────────────────────────────────
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges("retrieve", route_after_retrieve,
                                 {"generate": "generate", "fallback": "fallback"})
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    PINECONE_KEY = os.getenv("PINECONE_API_KEY", "")
    GROQ_KEY     = os.getenv("GROQ_API_KEY", "")

    if not PINECONE_KEY or not GROQ_KEY:
        print("Set PINECONE_API_KEY and GROQ_API_KEY in .env")
        sys.exit(1)

    from rag_pipeline import VersionControlRAG
    print("Loading RAG...")
    rag = VersionControlRAG(pinecone_key=PINECONE_KEY, ingest_only=True)
    rag.load_local_corpus("local_corpus.json")

    question = "How do I create a FastAPI endpoint with Pydantic v2 validation?"

    print("\n── LangChain LCEL chain ──")
    chain = build_lcel_chain(rag, groq_api_key=GROQ_KEY, strategy="advanced")
    answer = chain.invoke({"question": question})
    print(f"Answer:\n{answer}\n")

    print("\n── LangGraph pipeline ──")
    pipeline = build_langgraph_pipeline(rag, groq_api_key=GROQ_KEY, strategy="advanced")
    result = pipeline.invoke({"question": question, "contexts": [], "answer": "", "retrieval_ok": False})
    print(f"Answer:\n{result['answer']}")
    print(f"Contexts retrieved: {len(result['contexts'])}")
    print(f"Retrieval ok: {result['retrieval_ok']}")
