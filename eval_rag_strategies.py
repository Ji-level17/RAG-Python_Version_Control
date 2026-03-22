"""
RAG Strategy Comparison — Evaluation Script
============================================
Compares 4 retrieval strategies using a consistent Groq generator + custom RAGAS metrics.

Strategies evaluated:
  1. baseline       — Dense vector search only
  2. advanced       — Hybrid (BM25 + dense) + Cross-encoder rerank
  3. hierarchical   — Parent-group selection → child rerank (HierarchicalRAG)
  4. import_graph   — Dense + import-dependency expansion (ImportGraphRAG)

Output files:
  strategy_comparison_results.csv    — per-record scores for every strategy
  strategy_comparison_summary.csv    — mean scores per strategy
  strategy_comparison_chart.png      — bar chart

Usage (on A40):
  # Keys auto-loaded from .env — no need to pass them manually
  python eval_rag_strategies.py --quick-n 10
  python eval_rag_strategies.py --strategies advanced hierarchical --quick-n 20
  python eval_rag_strategies.py --quick-n 0   # full test split (~85 records)

  # Or pass keys explicitly if needed:
  python eval_rag_strategies.py --pinecone-key KEY --groq-key KEY --quick-n 10
"""

import argparse
import json
import os

# Auto-load .env so keys don't need to be passed on the command line
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on env vars or CLI args
import random
import time
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pinecone-key",   default=os.getenv("PINECONE_API_KEY"),
                   help="Pinecone API key (default: PINECONE_API_KEY from .env)")
    p.add_argument("--groq-key",       default=os.getenv("GROQ_API_KEY"),
                   help="Groq API key (default: GROQ_API_KEY from .env)")
    p.add_argument("--gemini-key",     default=os.getenv("GEMINI_API_KEY"),
                   help="Gemini API key — if set, Gemini is used instead of Groq")
    p.add_argument("--data-dir",       default=".",        help="folder with corpus/dataset files")
    p.add_argument("--corpus",         default="local_corpus.json")
    p.add_argument("--dataset",        default="ragas_dataset.jsonl")
    p.add_argument("--splits",         default="fastapi_golden_set_splits.json")
    p.add_argument("--quick-n",        type=int, default=0,  help="0 = use full test split")
    p.add_argument("--top-k",          type=int, default=5)
    p.add_argument("--gen-model",      default=None,
                   help="Generation model (default: gemini-2.0-flash or llama-3.3-70b-versatile)")
    p.add_argument("--judge-model",    default=None,
                   help="Judge model (default: gemini-2.0-flash or llama-3.1-8b-instant)")
    p.add_argument("--gen-delay",      type=float, default=None, help="seconds between generation calls")
    p.add_argument("--judge-delay",    type=float, default=None, help="seconds between judge calls")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--strategies",     nargs="+",
                   default=["baseline", "advanced", "hierarchical", "import_graph"],
                   choices=["baseline", "advanced", "hierarchical", "import_graph"])
    p.add_argument("--output-dir",     default=".")
    p.add_argument("--existing-csv",   default=None,
                   help="Path to a partial results CSV to load before running (skips re-running those strategies)")
    args = p.parse_args()
    if not args.pinecone_key:
        p.error("Pinecone key required: pass --pinecone-key or set PINECONE_API_KEY in .env")
    if not args.gemini_key and not args.groq_key:
        p.error("LLM key required: set GEMINI_API_KEY or GROQ_API_KEY in .env")
    # Set defaults based on backend
    use_gemini = bool(args.gemini_key)
    if args.gen_model is None:
        args.gen_model   = "gemini-2.5-flash" if use_gemini else "llama-3.3-70b-versatile"
    if args.judge_model is None:
        args.judge_model = "gemini-2.5-flash" if use_gemini else "llama-3.1-8b-instant"
    if args.gen_delay is None:
        args.gen_delay   = 1.0 if use_gemini else 5.0
    if args.judge_delay is None:
        args.judge_delay = 1.0 if use_gemini else 8.0
    return args


# ─────────────────────────────────────────────
# RAGAS-style metrics (Groq-based)
# ─────────────────────────────────────────────
class MetricEvaluator:
    def __init__(self, judge_model: str, llm_client, delay: float):
        self.judge_model = judge_model
        self.client      = llm_client
        self.delay       = delay
        self._sem        = SentenceTransformer(
            "flax-sentence-embeddings/st-codesearch-distilroberta-base"
        )

    def _judge(self, prompt: str, max_tokens: int = 2048) -> str:
        time.sleep(self.delay)
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(
                    model       = self.judge_model,
                    messages    = [{"role": "user", "content": prompt}],
                    max_tokens  = max_tokens,
                    temperature = 0,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                err = str(e).lower()
                if "429" in str(e) or "rate_limit" in err:
                    wait = 15 * (attempt + 1)
                    print(f"    rate limit hit, waiting {wait}s...")
                    time.sleep(wait)
                elif any(k in err for k in ("10054", "10060", "getaddrinfo", "timeout", "connection")):
                    wait = 10 * (attempt + 1)
                    print(f"    connection error, retrying in {wait}s... ({e})")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Judge failed after 5 retries")

    def _parse_score(self, text: str) -> float:
        """Extract a numeric score (0-10) from judge response, return 0.0-1.0."""
        import re as _re
        # Look for a number 0-10 in the response
        match = _re.search(r'\b(\d{1,2})\b', text)
        if match:
            val = int(match.group(1))
            return min(max(val, 0), 10) / 10.0
        return 0.5  # fallback if no number found

    def faithfulness(self, answer: str, contexts: list[str]) -> float:
        """RAGAS-style faithfulness: extract claims then verify each via NLI."""
        # Step 1: Extract atomic claims from the answer
        claims_raw = self._judge(
            f"Given the following code answer, extract all atomic factual claims. "
            f"Each claim should be a single, verifiable statement about APIs, "
            f"function signatures, parameters, or code patterns used.\n\n"
            f"Answer:\n{answer}\n\n"
            f"List each claim on a separate line, numbered 1., 2., 3., etc. "
            f"If the answer is very short, extract at least 1 claim."
        )
        # Parse claims
        import re as _re
        claims = [line.strip() for line in claims_raw.splitlines()
                  if _re.match(r'^\d+[\.\)]', line.strip())]
        if not claims:
            claims = [claims_raw.strip()]  # fallback: treat whole response as one claim
        claims = claims[:8]  # cap to avoid excessive API calls

        # Step 2: Verify each claim against context (NLI)
        ctx = "\n---\n".join(contexts[:5])
        supported = 0
        for claim in claims:
            verdict = self._judge(
                f"Context (retrieved code):\n{ctx}\n\n"
                f"Claim: {claim}\n\n"
                f"Can this claim be directly inferred from the context above? "
                f"Reply with ONLY one word: Yes or No."
            ).strip().lower()
            if verdict.startswith("yes") or verdict == "1":
                supported += 1
        return supported / len(claims) if claims else 0.0

    def answer_relevancy(self, question: str, answer: str) -> float:
        """How relevant is the answer to the question? (0-10 scale via LLM judge)"""
        verdict = self._judge(
            f"Question: {question}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Rate how relevant and complete the answer is for the question. "
            f"Consider: Does it directly address the question? Is it specific and useful?\n"
            f"Score from 0 to 10 (0=completely irrelevant, 5=partially relevant, "
            f"10=perfectly relevant). Reply with ONLY a number."
        )
        return self._parse_score(verdict)

    def context_precision(self, question: str, contexts: list[str]) -> float:
        """RAGAS-style context precision using Average Precision (AP) formula.
        AP = Σ(Precision@k × v_k) / total_relevant
        where v_k=1 if chunk at rank k is relevant, 0 otherwise.
        This naturally rewards placing relevant chunks at higher ranks."""
        relevance = []  # binary relevance at each rank
        for ctx in contexts[:5]:
            verdict = self._judge(
                f"Question: {question}\n\nCode chunk:\n{ctx}\n\n"
                f"Is this code chunk useful for answering the question? "
                f"Answer 1 if useful, 0 if not useful."
            ).strip()
            relevance.append(1 if "1" in verdict else 0)
        if not relevance:
            return float("nan")
        total_relevant = sum(relevance)
        if total_relevant == 0:
            return 0.0
        # AP formula: Σ(Precision@k × v_k) / total_relevant
        ap_sum = 0.0
        running_relevant = 0
        for k, v_k in enumerate(relevance):
            if v_k == 1:
                running_relevant += 1
                precision_at_k = running_relevant / (k + 1)
                ap_sum += precision_at_k
        return ap_sum / total_relevant

    def context_recall(self, contexts: list[str], ground_truth: str) -> float:
        """How much of the ground truth is covered by the context? (0-10 scale)"""
        ctx = "\n---\n".join(contexts[:5])
        verdict = self._judge(
            f"Context (retrieved code):\n{ctx}\n\n"
            f"Reference implementation:\n{ground_truth}\n\n"
            f"Rate how well the retrieved context covers the information needed "
            f"to produce the reference implementation. Consider: Are the key APIs, "
            f"classes, functions, and patterns present in the context?\n"
            f"Score from 0 to 10 (0=no coverage, 5=partial coverage, "
            f"10=full coverage). Reply with ONLY a number."
        )
        return self._parse_score(verdict)

    def score_record(self, question, answer, contexts, ground_truth) -> dict:
        row = {}
        row["answer_relevancy"] = self.answer_relevancy(question, answer)
        try:
            row["faithfulness"] = self.faithfulness(answer, contexts)
        except Exception as e:
            print(f"    faithfulness error: {e}")
            row["faithfulness"] = float("nan")
        try:
            row["context_precision"] = self.context_precision(question, contexts)
        except Exception as e:
            print(f"    context_precision error: {e}")
            row["context_precision"] = float("nan")
        try:
            row["context_recall"] = self.context_recall(contexts, ground_truth)
        except Exception as e:
            print(f"    context_recall error: {e}")
            row["context_recall"] = float("nan")
        return row


# ─────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a FastAPI expert. Answer the question using ONLY the provided code context. "
    "Do NOT add information beyond what is in the context. "
    "Start with a short Python comment describing what the code does, "
    "then write the code. Be concise and precise. No markdown fences."
)

def generate_answer(client, model: str, question: str,
                    contexts: list[str], delay: float) -> str:
    ctx = "\n\n".join(contexts[:5])
    for attempt in range(5):
        time.sleep(delay)
        try:
            resp = client.chat.completions.create(
                model    = model,
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content":
                     f"### Context\n{ctx}\n\n### Question\n{question}\n\n### Answer"},
                ],
                max_tokens  = 512,
                temperature = 0,
            )
            raw = resp.choices[0].message.content.strip()
            break
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ("429", "rate_limit", "10054", "10060", "getaddrinfo", "timeout", "connection")):
                wait = 10 * (attempt + 1)
                print(f"    gen retry {attempt+1}/5 in {wait}s... ({e})")
                time.sleep(wait)
            else:
                raise
    else:
        return "# generation failed after 5 retries"
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines()
                        if not l.strip().startswith("```")).strip()
    return raw or "# model returned empty response"


# ─────────────────────────────────────────────
# Strategy loader
# ─────────────────────────────────────────────
def _load_reranker():
    """Load CrossEncoder reranker on CPU (~400MB, no GPU needed)."""
    from sentence_transformers import CrossEncoder
    print("  Loading CrossEncoder reranker on CPU...")
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def load_strategy(name: str, pinecone_key: str, data_dir: Path, corpus_file: str):
    """Load a RAG retriever (ingest_only=True — no Qwen, saves ~15 GB VRAM)."""
    if name in ("baseline", "advanced"):
        import importlib
        import rag_pipeline
        importlib.reload(rag_pipeline)
        rag = rag_pipeline.VersionControlRAG(
            pinecone_key = pinecone_key,
            ingest_only  = True,
        )
        rag.load_local_corpus(str(data_dir / corpus_file))
        # baseline: pure dense vector search
        # advanced: dense + BM25 hybrid, NO reranker (cosine ranking)
        # reranker reserved for hierarchical & import_graph only
        return rag, name

    if name == "hierarchical":
        import importlib
        import hierarchical_rag
        importlib.reload(hierarchical_rag)
        rag = hierarchical_rag.HierarchicalRAG(
            pinecone_key = pinecone_key,
            ingest_only  = True,
        )
        rag.load_local_corpus(str(data_dir / corpus_file))
        rag.reranker = _load_reranker()
        return rag, "advanced"

    if name == "import_graph":
        import importlib
        import rag_strategies
        importlib.reload(rag_strategies)
        rag = rag_strategies.ImportGraphRAG(
            pinecone_key = pinecone_key,
            ingest_only  = True,
        )
        rag.load_local_corpus(str(data_dir / corpus_file))
        graph_file = data_dir / "import_graph.json"
        if graph_file.exists():
            rag.load_import_graph(str(graph_file))
        else:
            print(f"  WARNING: import_graph.json not found at {graph_file}. "
                  f"Graph expansion will be skipped (falls back to dense retrieval).")
        rag.reranker = _load_reranker()
        return rag, "graph"

    raise ValueError(f"Unknown strategy: {name}")


def retrieve(rag, mode: str, question: str, top_k: int,
             fallback_contexts: list[str]) -> tuple[list[str], bool]:
    """Returns (contexts, used_real_retrieval)."""
    try:
        contexts = rag.retrieve_complex(question, top_k=top_k, mode=mode)
        if contexts:
            return contexts, True
    except Exception as e:
        err = str(e)
        if "dimension" in err.lower():
            print(f"    Pinecone dimension mismatch — using local BM25 only")
        else:
            print(f"    Retrieval error ({mode}): {err[:120]}")
    # BM25-only local fallback (works without Pinecone)
    local = _bm25_retrieve(rag, question, top_k)
    if local:
        return local, True
    return fallback_contexts[:top_k], False


def _bm25_retrieve(rag, question: str, top_k: int) -> list[str]:
    """BM25-only retrieval from local corpus — no Pinecone needed."""
    if not getattr(rag, "bm25_model", None) or not rag.local_corpus:
        return []
    import re
    tokens = re.findall(r'[a-zA-Z_]\w*|\d+', question)
    scores = rag.bm25_model.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [rag.local_corpus[i]["content"] for i, s in ranked[:top_k] if s > 0]


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def main():
    args = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gemini_key:
        from google import genai as _genai

        class _GeminiClient:
            """Thin wrapper so Gemini looks like an OpenAI client."""
            def __init__(self, api_key):
                self._client = _genai.Client(api_key=api_key)
                self.chat    = self  # satisfy client.chat.completions.create(...)
                self.completions = self

            def create(self, model, messages, max_tokens=512, temperature=0, **_):
                import signal, threading
                prompt = "\n".join(
                    f"{m['role'].upper()}: {m['content']}" for m in messages
                )
                result = [None]
                error  = [None]
                def _call():
                    try:
                        result[0] = self._client.models.generate_content(
                            model    = model,
                            contents = prompt,
                            config   = _genai.types.GenerateContentConfig(
                                max_output_tokens = max_tokens,
                                temperature       = temperature,
                                thinking_config   = _genai.types.ThinkingConfig(thinking_budget=0),
                            ),
                        )
                    except Exception as e:
                        error[0] = e
                t = threading.Thread(target=_call)
                t.start()
                t.join(timeout=60)  # 60s timeout per request
                if t.is_alive():
                    raise TimeoutError("Gemini API request timed out after 60s")
                if error[0]:
                    raise error[0]
                resp = result[0]
                text = resp.text if resp.text is not None else ""
                class _Msg:
                    content = text
                class _Choice:
                    message = _Msg()
                class _Resp:
                    choices = [_Choice()]
                return _Resp()

        llm_client = _GeminiClient(args.gemini_key)
        print(f"Using Gemini backend: {args.gen_model}")
    else:
        from groq import Groq
        llm_client = Groq(api_key=args.groq_key)
        print(f"Using Groq backend: {args.gen_model}")
    evaluator   = MetricEvaluator(args.judge_model, llm_client, args.judge_delay)

    # ── Load test records ────────────────────────────────────────
    print("Loading dataset...")
    dataset_path = data_dir / args.dataset
    # Also accept .json if .jsonl not found
    if not dataset_path.exists() and dataset_path.suffix == ".jsonl":
        fallback = dataset_path.with_suffix(".json")
        if fallback.exists():
            dataset_path = fallback
            print(f"  (.jsonl not found, using {fallback.name})")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"  Download from: https://raw.githubusercontent.com/"
            f"Ji-level17/RAG-Python_Version_Control/Updated_main_version1.1/ragas_dataset.json"
        )
    with open(dataset_path, encoding="utf-8") as f:
        content = f.read().strip()
    # Support both .jsonl (one JSON per line) and .json (JSON array)
    if content.startswith("["):
        all_records = json.loads(content)
    else:
        all_records = [json.loads(l) for l in content.splitlines() if l.strip()]
    print(f"  Loaded {len(all_records)} total records from {dataset_path.name}")

    splits_path = data_dir / args.splits
    if splits_path.exists():
        with open(splits_path, encoding="utf-8") as f:
            splits = json.load(f)
        test_q       = {x["instruction"] for x in splits["test"]}
        test_records = [r for r in all_records if r.get("question") in test_q]
        if not test_records:
            # fallback: splits use "instruction" key, dataset may also use "instruction"
            test_records = [r for r in all_records if r.get("instruction") in test_q]
            for r in test_records:          # normalise key
                r.setdefault("question", r.get("instruction", ""))
    else:
        print("  splits file not found — using all records as test set")
        test_records = all_records

    random.seed(args.seed)
    eval_records = (
        random.sample(test_records, min(args.quick_n, len(test_records)))
        if args.quick_n > 0 else test_records
    )
    print(f"  Eval records: {len(eval_records)}  (test split: {len(test_records)})")

    # ── Load existing partial results ────────────────────────────
    all_rows = []
    done_strategies = set()
    if args.existing_csv and os.path.exists(args.existing_csv):
        df_existing = pd.read_csv(args.existing_csv)
        all_rows = df_existing.to_dict("records")
        done_strategies = set(df_existing["strategy"].unique())
        print(f"  Loaded {len(all_rows)} existing rows from {args.existing_csv}")
        print(f"  Already done: {sorted(done_strategies)} — will skip these strategies")

    for strategy_name in args.strategies:
        if strategy_name in done_strategies:
            print(f"\n  Skipping '{strategy_name}' (already in existing CSV)")
            continue
        print(f"\n{'='*60}")
        print(f"  Strategy: {strategy_name}  ({len(eval_records)} records)")
        print(f"{'='*60}")

        rag, mode = load_strategy(
            strategy_name, args.pinecone_key, data_dir, args.corpus
        )

        rows = []
        for i, record in enumerate(tqdm(eval_records, desc=strategy_name)):
            question     = record["question"]
            ground_truth = record["ground_truth"]
            fallback_ctx = record.get("contexts", [])

            # Retrieve
            contexts, real_retrieval = retrieve(rag, mode, question, args.top_k, fallback_ctx)
            if not real_retrieval:
                print(f"    WARNING: using dataset fallback contexts (retrieval failed)")

            # Generate (consistent LLM generator across all strategies)
            try:
                answer = generate_answer(
                    llm_client, args.gen_model,
                    question, contexts, args.gen_delay
                )
            except Exception as e:
                print(f"    Generation error [{i}]: {e}")
                answer = "# generation error"

            # Score
            scores = evaluator.score_record(question, answer, contexts, ground_truth)

            row = {
                "strategy":     strategy_name,
                "question":     question,
                "answer":       answer,
                "ground_truth": ground_truth,
                **scores,
            }
            rows.append(row)
            all_rows.append(row)

            print(f"  [{i+1:3d}/{len(eval_records)}] "
                  f"faith={scores.get('faithfulness', float('nan')):.2f}  "
                  f"rel={scores.get('answer_relevancy', float('nan')):.2f}  "
                  f"prec={scores.get('context_precision', float('nan')):.2f}  "
                  f"rec={scores.get('context_recall', float('nan')):.2f}")

        # Per-strategy summary
        df_s = pd.DataFrame(rows)
        exist = [c for c in METRIC_COLS if c in df_s.columns]
        print(f"\n  [{strategy_name}] mean scores:")
        for col in exist:
            val = df_s[col].mean()
            print(f"    {col:25s}: {val:.3f}" if pd.notna(val)
                  else f"    {col:25s}: NaN")

    # ── Aggregate results ───────────────────────────────────────
    df_all = pd.DataFrame(all_rows)
    exist  = [c for c in METRIC_COLS if c in df_all.columns]

    all_strategies_order = ["baseline", "advanced", "hierarchical", "import_graph"]
    summary = (
        df_all.groupby("strategy")[exist]
        .mean().round(3)
        .reindex([s for s in all_strategies_order if s in df_all["strategy"].values])
    )

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"{'Strategy':20s}" + "".join(f"  {c[:14]:>14}" for c in exist))
    print("-"*70)
    for strat, row in summary.iterrows():
        vals = "".join(
            f"  {row[c]:14.3f}" if pd.notna(row[c]) else f"  {'NaN':>14}"
            for c in exist
        )
        print(f"{strat:20s}{vals}")
    print("="*70)

    # ── Save ────────────────────────────────────────────────────
    results_path = output_dir / "strategy_comparison_results.csv"
    summary_path = output_dir / "strategy_comparison_summary.csv"
    df_all.to_csv(results_path, index=False)
    summary.to_csv(summary_path)
    print(f"\nSaved: {results_path}")
    print(f"Saved: {summary_path}")

    # ── Plot ────────────────────────────────────────────────────
    _plot_comparison(summary, exist, args.strategies, output_dir)


def _plot_comparison(summary: pd.DataFrame, metrics: list,
                     strategy_order: list, output_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        palette = {
            "baseline":     "#B4B2A9",
            "advanced":     "#1D9E75",
            "hierarchical": "#7F77DD",
            "import_graph": "#D85A30",
        }
        colors = [palette.get(s, "#888888") for s in strategy_order]

        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            vals = summary.reindex(strategy_order)[metric].values
            bars = ax.bar(range(len(strategy_order)), vals,
                          color=colors, width=0.55, edgecolor="white")
            ax.set_xticks(range(len(strategy_order)))
            ax.set_xticklabels(strategy_order, rotation=25, ha="right", fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_title(metric.replace("_", "\n"), fontsize=10)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
            for bar, val in zip(bars, vals):
                if pd.notna(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                            f"{val:.2f}", ha="center", fontsize=8)

        fig.suptitle("RAG Strategy Comparison (RAGAS-style metrics)", fontsize=12)
        plt.tight_layout()
        chart_path = output_dir / "strategy_comparison_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {chart_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
