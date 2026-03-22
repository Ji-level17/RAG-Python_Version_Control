"""
Retrieval Quality Metrics — Recall@k, MRR, Latency
====================================================
Evaluates retrieval quality without LLM calls (pure local computation).
Compares retrieved chunks against ground_truth using token overlap.

Output:
  retrieval_metrics_summary.csv
  retrieval_metrics_chart.png
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pinecone-key", default=os.getenv("PINECONE_API_KEY"))
    p.add_argument("--data-dir", default=".")
    p.add_argument("--corpus", default="local_corpus.json")
    p.add_argument("--dataset", default="ragas_dataset.jsonl")
    p.add_argument("--splits", default="fastapi_golden_set_splits.json")
    p.add_argument("--quick-n", type=int, default=0)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strategies", nargs="+",
                   default=["baseline", "advanced", "hierarchical", "import_graph"])
    p.add_argument("--output-dir", default=".")
    return p.parse_args()


def tokenize(text: str) -> set:
    """Simple code-aware tokenizer."""
    return set(re.findall(r'[a-zA-Z_]\w*|\d+', text.lower()))


def chunk_relevance(chunk: str, ground_truth: str, threshold: float = 0.15) -> bool:
    """Check if a chunk is relevant to ground_truth via token overlap."""
    gt_tokens = tokenize(ground_truth)
    chunk_tokens = tokenize(chunk)
    if not gt_tokens:
        return False
    overlap = len(gt_tokens & chunk_tokens) / len(gt_tokens)
    return overlap >= threshold


def recall_at_k(retrieved: list[str], ground_truth: str, k: int) -> float:
    """1.0 if any of the top-k chunks is relevant, else 0.0."""
    for chunk in retrieved[:k]:
        if chunk_relevance(chunk, ground_truth):
            return 1.0
    return 0.0


def reciprocal_rank(retrieved: list[str], ground_truth: str) -> float:
    """1/rank of the first relevant chunk, or 0 if none found."""
    for i, chunk in enumerate(retrieved):
        if chunk_relevance(chunk, ground_truth):
            return 1.0 / (i + 1)
    return 0.0


def load_strategy(name, pinecone_key, data_dir, corpus_file):
    """Same loader as eval_rag_strategies.py."""
    if name in ("baseline", "advanced"):
        import importlib
        import rag_pipeline
        importlib.reload(rag_pipeline)
        rag = rag_pipeline.VersionControlRAG(
            pinecone_key=pinecone_key, ingest_only=True)
        rag.load_local_corpus(str(data_dir / corpus_file))
        # advanced: dense + BM25 hybrid, NO reranker (cosine ranking)
        # reranker reserved for hierarchical & import_graph only
        return rag, name

    if name == "hierarchical":
        import importlib
        import hierarchical_rag
        importlib.reload(hierarchical_rag)
        rag = hierarchical_rag.HierarchicalRAG(
            pinecone_key=pinecone_key, ingest_only=True)
        rag.load_local_corpus(str(data_dir / corpus_file))
        from sentence_transformers import CrossEncoder
        rag.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return rag, "advanced"

    if name == "import_graph":
        import importlib
        import rag_strategies
        importlib.reload(rag_strategies)
        rag = rag_strategies.ImportGraphRAG(
            pinecone_key=pinecone_key, ingest_only=True)
        rag.load_local_corpus(str(data_dir / corpus_file))
        graph_file = data_dir / "import_graph.json"
        if graph_file.exists():
            rag.load_import_graph(str(graph_file))
        from sentence_transformers import CrossEncoder
        rag.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return rag, "graph"

    raise ValueError(f"Unknown strategy: {name}")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Load test records (same logic as eval_rag_strategies.py)
    dataset_path = data_dir / args.dataset
    if not dataset_path.exists() and dataset_path.suffix == ".jsonl":
        fallback = dataset_path.with_suffix(".json")
        if fallback.exists():
            dataset_path = fallback
    with open(dataset_path, encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        all_records = json.loads(content)
    else:
        all_records = [json.loads(l) for l in content.splitlines() if l.strip()]

    splits_path = data_dir / args.splits
    if splits_path.exists():
        with open(splits_path, encoding="utf-8") as f:
            splits = json.load(f)
        test_q = {x["instruction"] for x in splits["test"]}
        test_records = [r for r in all_records if r.get("question") in test_q]
        if not test_records:
            test_records = [r for r in all_records if r.get("instruction") in test_q]
            for r in test_records:
                r.setdefault("question", r.get("instruction", ""))
    else:
        test_records = all_records

    import random
    random.seed(args.seed)
    eval_records = (
        random.sample(test_records, min(args.quick_n, len(test_records)))
        if args.quick_n > 0 else test_records
    )
    print(f"Eval records: {len(eval_records)}")

    all_rows = []

    for strategy_name in args.strategies:
        print(f"\n{'='*60}")
        print(f"  Strategy: {strategy_name}")
        print(f"{'='*60}")

        rag, mode = load_strategy(strategy_name, args.pinecone_key, data_dir, args.corpus)

        latencies = []
        recall_1, recall_3, recall_5, recall_10 = [], [], [], []
        mrr_scores = []

        for record in tqdm(eval_records, desc=strategy_name):
            question = record["question"]
            ground_truth = record["ground_truth"]

            # Time the retrieval
            t0 = time.perf_counter()
            try:
                contexts = rag.retrieve_complex(question, top_k=args.top_k, mode=mode)
            except Exception as e:
                print(f"  Retrieval error: {e}")
                contexts = []
            latency = time.perf_counter() - t0
            latencies.append(latency)

            if not contexts:
                recall_1.append(0.0)
                recall_3.append(0.0)
                recall_5.append(0.0)
                recall_10.append(0.0)
                mrr_scores.append(0.0)
                continue

            recall_1.append(recall_at_k(contexts, ground_truth, 1))
            recall_3.append(recall_at_k(contexts, ground_truth, 3))
            recall_5.append(recall_at_k(contexts, ground_truth, 5))
            recall_10.append(recall_at_k(contexts, ground_truth, 10))
            mrr_scores.append(reciprocal_rank(contexts, ground_truth))

        row = {
            "strategy": strategy_name,
            "Recall@1": sum(recall_1) / len(recall_1),
            "Recall@3": sum(recall_3) / len(recall_3),
            "Recall@5": sum(recall_5) / len(recall_5),
            "Recall@10": sum(recall_10) / len(recall_10),
            "MRR": sum(mrr_scores) / len(mrr_scores),
            "Latency_mean_s": sum(latencies) / len(latencies),
            "Latency_p50_s": sorted(latencies)[len(latencies)//2],
            "Latency_p95_s": sorted(latencies)[int(len(latencies)*0.95)],
        }
        all_rows.append(row)

        print(f"\n  Recall@1={row['Recall@1']:.3f}  @3={row['Recall@3']:.3f}  "
              f"@5={row['Recall@5']:.3f}  @10={row['Recall@10']:.3f}")
        print(f"  MRR={row['MRR']:.3f}")
        print(f"  Latency: mean={row['Latency_mean_s']:.3f}s  "
              f"p50={row['Latency_p50_s']:.3f}s  p95={row['Latency_p95_s']:.3f}s")

    # Save
    df = pd.DataFrame(all_rows)
    summary_path = output_dir / "retrieval_metrics_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    # Print table
    print(f"\n{'='*90}")
    print(f"{'Strategy':16s} {'R@1':>6s} {'R@3':>6s} {'R@5':>6s} {'R@10':>6s} "
          f"{'MRR':>6s} {'Lat_ms':>8s} {'p95_ms':>8s}")
    print("-"*90)
    for _, r in df.iterrows():
        print(f"{r['strategy']:16s} {r['Recall@1']:6.3f} {r['Recall@3']:6.3f} "
              f"{r['Recall@5']:6.3f} {r['Recall@10']:6.3f} {r['MRR']:6.3f} "
              f"{r['Latency_mean_s']*1000:8.1f} {r['Latency_p95_s']*1000:8.1f}")
    print("="*90)

    # Plot
    _plot(df, output_dir)


def _plot(df, output_dir):
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        palette = {
            "baseline": "#B4B2A9", "advanced": "#1D9E75",
            "hierarchical": "#7F77DD", "import_graph": "#D85A30",
        }

        metrics = ["Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Recall@k + MRR
        x = np.arange(len(metrics))
        width = 0.18
        for i, (_, row) in enumerate(df.iterrows()):
            vals = [row[m] for m in metrics]
            axes[0].bar(x + i * width, vals, width,
                       label=row["strategy"],
                       color=palette.get(row["strategy"], "#888"))
        axes[0].set_xticks(x + width * 1.5)
        axes[0].set_xticklabels(metrics, fontsize=9)
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Recall@k & MRR")
        axes[0].legend(fontsize=8)
        axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=0.7)

        # Right: Latency
        strategies = df["strategy"].tolist()
        lat_mean = df["Latency_mean_s"].values * 1000
        lat_p95 = df["Latency_p95_s"].values * 1000
        colors = [palette.get(s, "#888") for s in strategies]
        bars = axes[1].bar(strategies, lat_mean, color=colors, width=0.5)
        axes[1].errorbar(strategies, lat_mean, yerr=lat_p95 - lat_mean,
                        fmt='none', ecolor='black', capsize=5)
        axes[1].set_title("Retrieval Latency (ms)")
        axes[1].set_ylabel("ms")
        for bar, val in zip(bars, lat_mean):
            axes[1].text(bar.get_x() + bar.get_width()/2, val + 5,
                        f"{val:.0f}", ha="center", fontsize=8)

        fig.suptitle("Retrieval Quality Metrics", fontsize=12)
        plt.tight_layout()
        chart_path = output_dir / "retrieval_metrics_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {chart_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
