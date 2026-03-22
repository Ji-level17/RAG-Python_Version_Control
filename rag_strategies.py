# Strategy 3: Import-Graph Expansion RAG
import re
import json
import os
from collections import defaultdict
from rag_pipeline import VersionControlRAG, UPSERT_BATCH_SIZE


def parse_imports(parser, file_path):
    """Extract imported module names from a Python file using Tree-sitter."""
    try:
        with open(file_path, 'rb') as f:
            code_bytes = f.read()
    except OSError:
        return []

    tree = parser.parse(code_bytes)
    modules = []
    stack = [tree.root_node]

    while stack:
        node = stack.pop()
        if node.type == 'import_statement':
            for child in node.children:
                if child.type in ('dotted_name', 'aliased_import'):
                    name_node = child if child.type == 'dotted_name' else child.children[0]
                    text = code_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
                    modules.append(text.split('.')[0])
        elif node.type == 'import_from_statement':
            for child in node.children:
                if child.type == 'dotted_name':
                    text = code_bytes[child.start_byte:child.end_byte].decode('utf-8')
                    modules.append(text.split('.')[0])
                    break
                elif child.type == 'relative_import':
                    break
        else:
            stack.extend(reversed(node.children))

    return modules


class ImportGraphRAG(VersionControlRAG):
    """Strategy 3: Import-Graph Expansion RAG."""

    _EXCLUDED_DIRS = {'.git', '__pycache__', 'tests', 'docs', 'benchmarks', 'site'}

    def __init__(self, pinecone_key, groq_api_key=None, index_name="python-rag-v3", **kwargs):
        super().__init__(pinecone_key, index_name=index_name, **kwargs)
        self.import_graph: dict[str, list[str]] = {}
        self.file_to_docs: dict[str, list[str]] = defaultdict(list)
        self.module_to_file: dict[str, str] = {}
        # Optional Groq client for frontier-model generation
        if groq_api_key:
            from groq import Groq
            self.llm = Groq(api_key=groq_api_key)
            self.groq_model = "llama-3.3-70b-versatile"
        else:
            self.llm = None
            self.groq_model = None

    def build_import_graph(self, repo_path: str, repo_name: str) -> None:
        """Build the import dependency graph for a processed repository."""
        if not self.local_corpus:
            print("Warning: local_corpus is empty. Run RepoProcessor.process_repository() first.")
            return

        # Step 1: Recover safe_path → [doc_ids] from the corpus
        prefix = repo_name + "_"
        suffix_pat = re.compile(r'_f\d+$')
        self.file_to_docs = defaultdict(list)

        for doc in self.local_corpus:
            doc_id = doc["id"]
            if not doc_id.startswith(prefix):
                continue
            inner = doc_id[len(prefix):]
            safe_path = suffix_pat.sub('', inner)
            self.file_to_docs[safe_path].append(doc_id)

        # Step 2: Map module names to safe_paths
        self.module_to_file = {}
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in self._EXCLUDED_DIRS]
            for fname in files:
                if not fname.endswith('.py'):
                    continue
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, repo_path)
                safe_path = rel_path.replace('\\', '/').replace('/', '_')

                no_ext = safe_path[:-3] if safe_path.endswith('.py') else safe_path
                dotted = no_ext.replace('_', '.')
                self.module_to_file[dotted] = safe_path

                short_name = os.path.splitext(fname)[0]
                self.module_to_file.setdefault(short_name, safe_path)

                self.module_to_file[no_ext] = safe_path

        # Step 3: Parse imports and populate import_graph
        self.import_graph = {}
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in self._EXCLUDED_DIRS]
            for fname in files:
                if not fname.endswith('.py'):
                    continue
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, repo_path)
                safe_path = rel_path.replace('\\', '/').replace('/', '_')

                imported_modules = parse_imports(self.parser, full_path)
                related_doc_ids = []
                for module in imported_modules:
                    imported_safe = self.module_to_file.get(module)
                    if imported_safe and imported_safe != safe_path:
                        related_doc_ids.extend(self.file_to_docs.get(imported_safe, []))

                seen = set()
                unique_related = []
                for rid in related_doc_ids:
                    if rid not in seen:
                        seen.add(rid)
                        unique_related.append(rid)

                for doc_id in self.file_to_docs.get(safe_path, []):
                    self.import_graph[doc_id] = unique_related

        total_edges = sum(len(v) for v in self.import_graph.values())
        print(f"Import graph built: {len(self.import_graph)} nodes, {total_edges} edges.")

    def save_import_graph(self, filepath: str = "import_graph.json") -> None:
        """Persist the import graph to disk."""
        payload = {
            "import_graph": self.import_graph,
            "file_to_docs": dict(self.file_to_docs),
            "module_to_file": self.module_to_file,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Import graph saved to {filepath}.")

    def load_import_graph(self, filepath: str = "import_graph.json") -> None:
        """Restore the import graph from disk."""
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Call build_import_graph() first.")
            return
        with open(filepath, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        self.import_graph = payload.get("import_graph", {})
        self.file_to_docs = defaultdict(list, payload.get("file_to_docs", {}))
        self.module_to_file = payload.get("module_to_file", {})
        total_edges = sum(len(v) for v in self.import_graph.values())
        print(f"Import graph loaded: {len(self.import_graph)} nodes, {total_edges} edges.")

    def retrieve_complex(self, query: str, target_version: str = None,
                         top_k: int = 3, mode: str = "graph") -> list[str]:
        """Retrieve code snippets using import-graph-expanded dense retrieval."""
        query_vector = self.embedder.encode(query).tolist()

        parsed_version = self._parse_version(target_version) if target_version else None
        filter_dict = {"min_version": {"$lte": parsed_version}} if parsed_version else {}

        fetch_k = max(top_k * 4, 20)
        pinecone_res = self.index.query(
            vector=query_vector,
            filter=filter_dict,
            top_k=fetch_k,
            include_metadata=True
        )

        initial = {m['id']: m['metadata']['code'] for m in pinecone_res['matches']}

        # BM25 hybrid: add keyword-matched docs
        if self.bm25_model is not None:
            tokenized_query = re.findall(r'[a-zA-Z_]\w*|\d+', query)
            bm25_scores = self.bm25_model.get_scores(tokenized_query)
            ranked_bm25 = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
            for i, s in ranked_bm25[:fetch_k]:
                if s > 0:
                    doc = self.local_corpus[i]
                    if doc["id"] not in initial:
                        initial[doc["id"]] = doc["content"]

        expanded = {}
        max_expand = int(fetch_k * 1.5)  # allow more graph-expanded candidates for reranker
        if self.import_graph:
            local_doc_map = {
                doc["id"]: doc["content"]
                for doc in self.local_corpus
                if not parsed_version
                or doc["metadata"]["min_version"] <= parsed_version
            }
            # Round-robin expansion: distribute across all initial docs evenly
            initial_ids = list(initial.keys())
            import_lists = {
                doc_id: [r for r in self.import_graph.get(doc_id, [])
                         if r not in initial and r in local_doc_map]
                for doc_id in initial_ids
            }
            round_idx = 0
            while len(expanded) < max_expand:
                added_any = False
                for doc_id in initial_ids:
                    imports = import_lists.get(doc_id, [])
                    if round_idx < len(imports):
                        related_id = imports[round_idx]
                        if related_id not in expanded:
                            expanded[related_id] = local_doc_map[related_id]
                            added_any = True
                            if len(expanded) >= max_expand:
                                break
                if not added_any:
                    break
                round_idx += 1
        else:
            print("[ImportGraphRAG] Warning: import_graph is empty. "
                  "Call build_import_graph() first. Falling back to dense-only retrieval.")

        candidates = {**initial, **expanded}

        if not candidates:
            return []

        if self.reranker is not None:
            pairs = [[query, code] for code in candidates.values()]
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(candidates.values(), scores),
                            key=lambda x: x[1], reverse=True)
        else:
            # v2: score ALL candidates (including graph-expanded) by cosine similarity
            # so expanded dependencies get a fair ranking instead of being appended last.
            import numpy as np
            query_vec = np.array(query_vector)
            pinecone_scores = {m['id']: m['score'] for m in pinecone_res['matches']}
            scored = []
            for doc_id, code in candidates.items():
                if doc_id in pinecone_scores:
                    scored.append((code, pinecone_scores[doc_id]))
                else:
                    code_vec = self.embedder.encode(code)
                    cos_sim = float(np.dot(query_vec, code_vec) /
                                    (np.linalg.norm(query_vec) * np.linalg.norm(code_vec) + 1e-9))
                    scored.append((code, cos_sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            ranked = scored

            # --- v1 (previous): Pinecone hits score=1, graph-expanded score=0 ---
            # pinecone_ids = {m['id'] for m in pinecone_res['matches']}
            # ranked = (
            #     [(v, 1) for k, v in candidates.items() if k in pinecone_ids] +
            #     [(v, 0) for k, v in candidates.items() if k not in pinecone_ids]
            # )

        print(f"[ImportGraphRAG] {len(initial)} dense hits + {len(expanded)} "
              f"graph-expanded → top {top_k}.")
        return [code for code, _ in ranked[:top_k]]

    def generate_answer(self, query: str, target_version: str = None,
                        top_k: int = 3) -> str:
        """Retrieve with graph expansion, then generate an answer."""
        query_vector = self.embedder.encode(query).tolist()
        parsed_version = self._parse_version(target_version) if target_version else None
        filter_dict = {"min_version": {"$lte": parsed_version}} if parsed_version else {}

        fetch_k = max(top_k * 3, 10)
        pinecone_res = self.index.query(
            vector=query_vector,
            filter=filter_dict,
            top_k=fetch_k,
            include_metadata=True,
        )

        initial = {m['id']: m['metadata']['code'] for m in pinecone_res['matches']}

        expanded = {}
        if self.import_graph:
            local_doc_map = {
                doc["id"]: doc["content"]
                for doc in self.local_corpus
                if not parsed_version or doc["metadata"]["min_version"] <= parsed_version
            }
            for doc_id in list(initial.keys()):
                for related_id in self.import_graph.get(doc_id, []):
                    if related_id not in initial and related_id in local_doc_map:
                        expanded[related_id] = local_doc_map[related_id]

        candidates = {**initial, **expanded}
        if not candidates:
            return "No relevant code found."

        if self.reranker is not None:
            pairs = [[query, code] for code in candidates.values()]
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(candidates.items(), scores), key=lambda x: x[1], reverse=True)
        else:
            # v2: cosine-score all candidates for fair ranking
            import numpy as np
            pinecone_scores = {m['id']: m['score'] for m in pinecone_res['matches']}
            scored = []
            for doc_id, code in candidates.items():
                if doc_id in pinecone_scores:
                    scored.append(((doc_id, code), pinecone_scores[doc_id]))
                else:
                    code_vec = self.embedder.encode(code)
                    cos_sim = float(np.dot(query_vector, code_vec) /
                                    (np.linalg.norm(query_vector) * np.linalg.norm(code_vec) + 1e-9))
                    scored.append(((doc_id, code), cos_sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            ranked = scored
            # --- v1: ranked = list(candidates.items())  # Pinecone cosine order ---

        top_items = ranked[:top_k]
        context_parts = []
        for (doc_id, code), _ in top_items:
            label = "DIRECT HIT" if doc_id in initial else "GRAPH-EXPANDED (imported dependency)"
            context_parts.append(f"[{label}]\n{code}")

        version_note = f" (target Python version: {target_version})" if target_version else ""
        prompt = (
            f"You are a Python migration expert{version_note}.\n"
            "The context below includes directly relevant code AND code from files it imports "
            "(marked GRAPH-EXPANDED), which may be affected by breaking API changes.\n"
            "Use ONLY the provided context to answer. If not found, say 'Not found in context'.\n\n"
            "Context:\n" + "\n\n---\n\n".join(context_parts) + f"\n\nQuestion: {query}\n\nAnswer:"
        )

        if self.llm is not None:
            # Frontier model via Groq
            response = self.llm.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.2,
            )
            answer = response.choices[0].message.content
        else:
            # Fall back to base-class Qwen generation
            result = super().generate_answer(query, target_version=target_version, top_k=top_k)
            answer = result["answer"]

        print(f"[ImportGraphRAG] {len(initial)} direct + {len(expanded)} expanded → top {top_k} used.")
        return {"answer": answer, "contexts": list(candidates.values())[:top_k]}


# Strategy registry — instantiate any strategy by name
from hierarchical_rag import HierarchicalRAG

STRATEGY_REGISTRY = {
    "baseline":     VersionControlRAG,   # mode="baseline"
    "advanced":     VersionControlRAG,   # mode="advanced"
    "hierarchical": HierarchicalRAG,     # mode="advanced"
    "import_graph": ImportGraphRAG,      # mode="graph"
}
