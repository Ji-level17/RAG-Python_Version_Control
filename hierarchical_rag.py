import re
from collections import defaultdict
from rank_bm25 import BM25Okapi

from rag_pipeline import VersionControlRAG


class HierarchicalRAG(VersionControlRAG):
    """
    Hierarchical retrieval built on top of the existing VersionControlRAG pipeline.

    Retrieval idea:
    1. Build parent groups from the flat local_corpus.
       Parent = source file path if present in doc_id, else filename-like prefix, else doc_id prefix.
    2. Retrieve top parent groups using aggregated semantic + lexical evidence.
    3. Retrieve top child chunks only within those selected parents.
    4. Re-rank final child candidates with the existing cross-encoder.

    This file is designed to plug into the current project with minimal changes.
    It works even if the current corpus only stores:
      - id
      - content
      - metadata: {min_version, summary, code}

    To get stronger hierarchy later, you can enrich doc_id / metadata during ingestion
    with explicit file_path, class_name, function_name, parent_id, chunk_level, etc.
    """

    def __init__(self, *args, parent_top_k_default=10, parent_candidate_multiplier=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent_top_k_default = parent_top_k_default
        self.parent_candidate_multiplier = parent_candidate_multiplier
        self.parent_groups = {}
        self.parent_bm25 = None
        self.parent_ids = []
        self.parent_texts = []
        self.rebuild_hierarchy()

    # -----------------------------
    # Hierarchy construction
    # -----------------------------
    def rebuild_hierarchy(self):
        """Build parent groups from the current local_corpus."""
        groups = defaultdict(list)

        for idx, doc in enumerate(self.local_corpus):
            parent_id = self._derive_parent_id(doc)
            groups[parent_id].append(idx)

        self.parent_groups = dict(groups)
        self.parent_ids = list(self.parent_groups.keys())
        self.parent_texts = [self._build_parent_text(parent_id) for parent_id in self.parent_ids]

        if self.parent_texts:
            tokenized = [self._tokenize_code_text(text) for text in self.parent_texts]
            self.parent_bm25 = BM25Okapi(tokenized)
        else:
            self.parent_bm25 = None

        print(f"Hierarchy rebuilt with {len(self.parent_groups)} parent groups.")

    def rebuild_bm25(self):
        """
        Keep the original child-level BM25, then rebuild parent hierarchy too.
        """
        super().rebuild_bm25()
        self.rebuild_hierarchy()

    def load_local_corpus(self, filepath="local_corpus.json"):
        """
        Restore corpus using base logic, then rebuild hierarchy.
        """
        super().load_local_corpus(filepath)
        self.rebuild_hierarchy()

    def _derive_parent_id(self, doc):
        """
        Best-effort parent grouping from the existing flat corpus.

        v2: Added _f\\d+$ suffix stripping to correctly group chunks that
            share the same source file (e.g. fastapi_docs_src_tutorial_py_f0,
            _f1, _f2 all belong to one parent).

        Priority:
        1. metadata['file_path'] if available
        2. strip chunk suffix _f\\d+$ from doc_id (covers our corpus format)
        3. doc_id prefix before the first '#'
        4. doc_id prefix before the first '::'
        5. first two slash-separated segments
        6. fallback to entire doc_id

        Previous version (v1) is preserved below as _derive_parent_id_v1.
        """
        metadata = doc.get("metadata", {}) or {}
        doc_id = doc.get("id", "unknown_doc")

        if metadata.get("file_path"):
            return metadata["file_path"]

        # v2: strip chunk suffix like _f0, _f1, _f12 to group by source file
        stripped = re.sub(r'_f\d+$', '', doc_id)
        if stripped != doc_id:
            return stripped

        if "#" in doc_id:
            return doc_id.split("#", 1)[0]
        if "::" in doc_id:
            return doc_id.split("::", 1)[0]

        parts = doc_id.split("/")
        if len(parts) >= 2:
            return "/".join(parts[:2])
        return doc_id

    def _derive_parent_id_v1(self, doc):
        """Previous version — kept for reference / rollback."""
        metadata = doc.get("metadata", {}) or {}
        doc_id = doc.get("id", "unknown_doc")
        if metadata.get("file_path"):
            return metadata["file_path"]
        if "#" in doc_id:
            return doc_id.split("#", 1)[0]
        if "::" in doc_id:
            return doc_id.split("::", 1)[0]
        parts = doc_id.split("/")
        if len(parts) >= 2:
            return "/".join(parts[:2])
        return doc_id

    def _build_parent_text(self, parent_id):
        """
        Create a parent-level summary text by concatenating child summaries and a small slice of code.
        This keeps parent retrieval cheaper and more semantically grounded.
        """
        pieces = [parent_id]
        child_indices = self.parent_groups.get(parent_id, [])

        for idx in child_indices:
            doc = self.local_corpus[idx]
            metadata = doc.get("metadata", {}) or {}
            summary = metadata.get("summary", "")
            code = metadata.get("code", doc.get("content", ""))
            snippet = code[:300]
            if summary:
                pieces.append(summary)
            if snippet:
                pieces.append(snippet)

        return "\n".join(pieces)

    def _tokenize_code_text(self, text):
        return re.findall(r'[a-zA-Z_]\w*|\d+', text)

    # -----------------------------
    # Retrieval helpers
    # -----------------------------
    def _filter_child_indices_by_version(self, child_indices, parsed_version=None):
        if not parsed_version:
            return child_indices

        kept = []
        for idx in child_indices:
            metadata = self.local_corpus[idx].get("metadata", {}) or {}
            min_version = metadata.get("min_version", 0)
            if min_version <= parsed_version:
                kept.append(idx)
        return kept

    def _select_parent_candidates(self, query, query_vector, parsed_version=None, parent_top_k=None):
        """
        Stage 1: retrieve top parent groups using semantic similarity + parent BM25.
        """
        if not self.parent_groups:
            return []

        parent_top_k = parent_top_k or self.parent_top_k_default
        target_parent_count = max(parent_top_k * self.parent_candidate_multiplier, parent_top_k)

        # Semantic scores against parent aggregated texts
        parent_embeddings = self.embedder.encode(self.parent_texts)
        semantic_scores = []
        for i, emb in enumerate(parent_embeddings):
            # cosine-like score via dot product is sufficient for ranking here because
            # sentence-transformers outputs normalized-ish vectors for retrieval use
            score = float((query_vector * emb).sum()) if hasattr(query_vector, 'sum') else None
            if score is None:
                # fallback for plain Python lists
                score = sum(a * b for a, b in zip(query_vector, emb))
            semantic_scores.append((self.parent_ids[i], score))

        semantic_rank = sorted(semantic_scores, key=lambda x: x[1], reverse=True)
        semantic_dict = {pid: score for pid, score in semantic_rank[:target_parent_count]}

        lexical_dict = {}
        if self.parent_bm25 is not None:
            tokens = self._tokenize_code_text(query)
            bm25_scores = self.parent_bm25.get_scores(tokens)
            lexical_rank = sorted(zip(self.parent_ids, bm25_scores), key=lambda x: x[1], reverse=True)
            lexical_dict = {pid: float(score) for pid, score in lexical_rank[:target_parent_count] if score > 0}

        combined = defaultdict(float)
        for pid, score in semantic_dict.items():
            combined[pid] += score
        for pid, score in lexical_dict.items():
            combined[pid] += score

        # Drop parents that have no version-compatible children
        version_checked = {}
        for pid, score in combined.items():
            child_indices = self.parent_groups.get(pid, [])
            filtered_children = self._filter_child_indices_by_version(child_indices, parsed_version)
            if filtered_children:
                version_checked[pid] = score

        ranked_parents = [pid for pid, _ in sorted(version_checked.items(), key=lambda x: x[1], reverse=True)]
        return ranked_parents[:parent_top_k]

    def _collect_child_candidates(self, parent_ids, parsed_version=None):
        candidate_indices = []
        for pid in parent_ids:
            child_indices = self.parent_groups.get(pid, [])
            child_indices = self._filter_child_indices_by_version(child_indices, parsed_version)
            candidate_indices.extend(child_indices)
        return candidate_indices

    # -----------------------------
    # Main retrieval function
    # -----------------------------
    def retrieve_complex(self, query, target_version=None, top_k=3, mode="advanced", parent_top_k=None, return_debug=False):
        """
        Hierarchical retrieval.

        Supported modes:
          - baseline: falls back to parent class baseline retrieval
          - advanced: hierarchical retrieval

        Returns a list of code strings by default, matching the existing notebook usage.
        Set return_debug=True to get richer debugging information.
        """
        if mode == "baseline":
            return super().retrieve_complex(query, target_version=target_version, top_k=top_k, mode="baseline")

        if mode != "advanced":
            raise ValueError(f"Invalid mode '{mode}'. Expected 'baseline' or 'advanced'.")

        if not self.local_corpus:
            print("Warning: local_corpus is empty.")
            return []

        parsed_version = self._parse_version(target_version) if target_version else None
        query_vector = self.embedder.encode(query)

        # Stage 1: parent retrieval
        selected_parents = self._select_parent_candidates(
            query=query,
            query_vector=query_vector,
            parsed_version=parsed_version,
            parent_top_k=parent_top_k or self.parent_top_k_default,
        )

        if not selected_parents:
            return []

        # Stage 2: child candidate collection within selected parents
        child_candidate_indices = self._collect_child_candidates(selected_parents, parsed_version=parsed_version)
        if not child_candidate_indices:
            return []

        # Optional child-level BM25 filtering within selected parents
        tokenized_query = self._tokenize_code_text(query)
        child_texts = [self.local_corpus[idx]["content"] for idx in child_candidate_indices]
        child_bm25 = BM25Okapi([self._tokenize_code_text(text) for text in child_texts])
        child_bm25_scores = child_bm25.get_scores(tokenized_query)

        # Combine semantic shortlist from vector retrieval with hierarchical child pool
        # This keeps the method close to the base pipeline while enforcing hierarchy.
        fetch_k = max(top_k * 5, 15)
        filter_dict = {"min_version": {"$lte": parsed_version}} if parsed_version else {}
        pinecone_res = self.index.query(
            vector=query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector,
            filter=filter_dict,
            top_k=fetch_k,
            include_metadata=True,
        )
        pinecone_docs = {match["metadata"]["code"] for match in pinecone_res["matches"]}

        hierarchical_candidates = []
        for local_i, corpus_idx in enumerate(child_candidate_indices):
            doc = self.local_corpus[corpus_idx]
            code = doc["content"]
            metadata = doc.get("metadata", {}) or {}
            summary = metadata.get("summary", "")
            bm25_score = float(child_bm25_scores[local_i])
            # Small boost if also surfaced by Pinecone
            overlap_boost = 1.0 if code in pinecone_docs else 0.0
            hierarchical_candidates.append({
                "id": doc["id"],
                "parent_id": self._derive_parent_id(doc),
                "code": code,
                "summary": summary,
                "bm25_score": bm25_score,
                "overlap_boost": overlap_boost,
            })

        # Pre-rank before expensive reranking
        hierarchical_candidates.sort(
            key=lambda x: (x["overlap_boost"], x["bm25_score"]),
            reverse=True,
        )
        rerank_pool = hierarchical_candidates[: max(top_k * 5, 20)]

        # Stage 3: cross-encoder rerank (skip if reranker not loaded)
        if self.reranker is not None:
            pairs = [[query, item["code"]] for item in rerank_pool]
            rerank_scores = self.reranker.predict(pairs)
            final_ranked = sorted(
                zip(rerank_pool, rerank_scores),
                key=lambda x: x[1],
                reverse=True,
            )
        else:
            # v2: cosine similarity + BM25 + Pinecone overlap for better ranking
            import numpy as np
            query_vec = np.array(query_vector) if not isinstance(query_vector, np.ndarray) else query_vector
            final_ranked = []
            for item in rerank_pool:
                code_vec = self.embedder.encode(item["code"])
                cos_sim = float(np.dot(query_vec, code_vec) /
                                (np.linalg.norm(query_vec) * np.linalg.norm(code_vec) + 1e-9))
                # Combine: cosine (0~1) + normalized BM25 + overlap boost
                combined = cos_sim + item["bm25_score"] * 0.1 + item["overlap_boost"] * 0.2
                final_ranked.append((item, combined))
            final_ranked.sort(key=lambda x: x[1], reverse=True)
            # --- v1: final_ranked = [(item, item["bm25_score"] + item["overlap_boost"]) for item in rerank_pool] ---

        if return_debug:
            return {
                "selected_parents": selected_parents,
                "child_candidate_count": len(child_candidate_indices),
                "results": [
                    {
                        "id": item["id"],
                        "parent_id": item["parent_id"],
                        "score": float(score),
                        "summary": item["summary"],
                        "code": item["code"],
                    }
                    for item, score in final_ranked[:top_k]
                ],
            }

        return [item["code"] for item, _ in final_ranked[:top_k]]


if __name__ == "__main__":
    print("HierarchicalRAG module loaded. Import this class in your notebook or test script.")