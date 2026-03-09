import os
import re
import json
import torch
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

UPSERT_BATCH_SIZE = 100


class VersionControlRAG:
    def __init__(self, pinecone_key, model_path="Qwen/Qwen2.5-Coder-7B", adapter_path=None, index_name="python-rag-v3"):
        print("Initializing Version-Controlled RAG Carrier...")
        self.pc = Pinecone(api_key=pinecone_key)

        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(name=index_name, dimension=384, metric="cosine",
                                spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        self.index = self.pc.Index(index_name)

        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.reranker = CrossEncoder('BAAI/bge-reranker-base')

        print(f"Loading Qwen from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        )
        if adapter_path:
            print(f"Applying QLoRA adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            self.model = base_model
        self.model.eval()

        self.parser = Parser(Language(tspython.language()))

        self.local_corpus = []
        self._corpus_ids = set()      # Fast duplicate detection for ingest_data
        self.bm25_model = None
        self._upsert_buffer = []      # Batch buffer for Pinecone upserts
        print("System ready.\n")

    def _parse_version(self, version_str):
        """Convert a version string like '3.12' to an integer (312) for numerical filtering."""
        try:
            if not version_str or not isinstance(version_str, str):
                return 0
            version_str = re.search(r'\d+\.\d+', version_str).group()
            parts = version_str.split('.')
            return int(parts[0]) * 100 + int(parts[1])
        except Exception:
            return 308  # Default to 3.8 on parse failure

    def extract_functions(self, file_path):
        """Extract all function definitions from a Python file using Tree-sitter.
        Uses an iterative traversal to avoid hitting Python's recursion limit on large files."""
        with open(file_path, 'rb') as f:
            code_bytes = f.read()
        tree = self.parser.parse(code_bytes)
        functions = []
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type == 'function_definition':
                functions.append(code_bytes[node.start_byte:node.end_byte].decode('utf-8'))
                # Do not descend into function bodies to avoid capturing nested functions
            else:
                # Extend in reverse so left-to-right source order is preserved (LIFO stack)
                stack.extend(reversed(node.children))
        return functions

    def get_qwen_metadata(self, code_snippet):
        """Use the Qwen model to extract min_version and a summary from a code snippet."""
        prompt = f"### Role\nPython Version Expert. Output JSON only.\n### Task\nIdentify min_version (e.g. '3.12') and summary for:\n{code_snippet}\n### Response\n"
        # truncation=True prevents silent failures on functions that exceed the model's context length
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=128, temperature=0.01, do_sample=True)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        try:
            return json.loads(re.search(r'\{.*\}', response, re.DOTALL).group())
        except Exception:
            return {"min_version": "3.8", "summary": "Analysis fallback"}

    def ingest_data(self, doc_id, code_content, min_version, summary):
        """Buffer a document for ingestion. Skips duplicate doc_ids to prevent double-ingestion
        on retry after partial failures. Flushes automatically when batch size is reached.
        Call flush_upsert_buffer() after the final document to commit remaining entries."""
        if doc_id in self._corpus_ids:
            return

        vector_values = self.embedder.encode(summary + " " + code_content).tolist()
        safe_version_num = self._parse_version(min_version)
        metadata = {"min_version": safe_version_num, "summary": summary, "code": code_content}

        self._upsert_buffer.append({"id": doc_id, "values": vector_values, "metadata": metadata})
        self.local_corpus.append({"id": doc_id, "content": code_content, "metadata": metadata})
        self._corpus_ids.add(doc_id)

        if len(self._upsert_buffer) >= UPSERT_BATCH_SIZE:
            self.flush_upsert_buffer()

    def flush_upsert_buffer(self):
        """Send all buffered vectors to Pinecone in a single batch request."""
        if not self._upsert_buffer:
            return
        try:
            self.index.upsert(vectors=self._upsert_buffer)
            self._upsert_buffer = []
        except Exception as e:
            print(f"Warning: Pinecone upsert failed ({len(self._upsert_buffer)} vectors): {e}")
            raise

    def rebuild_bm25(self):
        """Rebuild the BM25 index from the current local corpus.
        Uses regex-based tokenization suited for source code identifiers."""
        if not self.local_corpus:
            print("Warning: local_corpus is empty, skipping BM25 rebuild.")
            return
        tokenized_corpus = [re.findall(r'[a-zA-Z_]\w*|\d+', doc["content"]) for doc in self.local_corpus]
        self.bm25_model = BM25Okapi(tokenized_corpus)
        print(f"BM25 index rebuilt with {len(self.local_corpus)} documents.")

    def retrieve_complex(self, query, target_version=None, top_k=3, mode="advanced"):
        """Retrieve relevant code snippets for a given query.

        mode='baseline': vector search only.
        mode='advanced': hybrid search (vector + BM25) with Cross-Encoder reranking.
        """
        if mode not in ("baseline", "advanced"):
            raise ValueError(f"Invalid mode '{mode}'. Expected 'baseline' or 'advanced'.")

        query_vector = self.embedder.encode(query).tolist()

        parsed_version = self._parse_version(target_version) if target_version else None
        filter_dict = {"min_version": {"$lte": parsed_version}} if parsed_version else {}

        # Fetch more candidates in advanced mode to ensure the reranker has sufficient pool
        fetch_k = top_k if mode == "baseline" else max(top_k * 3, 10)
        pinecone_res = self.index.query(
            vector=query_vector,
            filter=filter_dict,
            top_k=fetch_k,
            include_metadata=True
        )

        if mode == "baseline":
            print("[Baseline Mode]")
            return [match['metadata']['code'] for match in pinecone_res['matches']]

        print("[Advanced Mode]")

        bm25_docs = {}
        if self.bm25_model is not None:
            tokenized_query = re.findall(r'[a-zA-Z_]\w*|\d+', query)
            bm25_scores = self.bm25_model.get_scores(tokenized_query)
            bm25_docs = {self.local_corpus[i]["id"]: self.local_corpus[i]["content"]
                         for i, s in enumerate(bm25_scores) if s > 0
                         if not parsed_version or self.local_corpus[i]["metadata"]["min_version"] <= parsed_version}
        else:
            print("Warning: BM25 corpus is empty. Call rebuild_bm25() or load_local_corpus() first.")

        pinecone_docs = {match['id']: match['metadata']['code'] for match in pinecone_res['matches']}
        candidates = {**pinecone_docs, **bm25_docs}

        if not candidates:
            return []

        pairs = [[query, code] for code in candidates.values()]
        scores = self.reranker.predict(pairs)
        final_results = [c for c, _ in sorted(zip(candidates.values(), scores), key=lambda x: x[1], reverse=True)]

        return final_results[:top_k]

    def save_local_corpus(self, filepath="local_corpus.json"):
        """Persist the local corpus to disk for state restoration on next restart."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.local_corpus, f, ensure_ascii=False, indent=2)
        print("Local corpus saved.")

    def load_local_corpus(self, filepath="local_corpus.json"):
        """Restore the local corpus from disk and rebuild the BM25 index."""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.local_corpus = json.load(f)
            self._corpus_ids = {doc["id"] for doc in self.local_corpus}
            self.rebuild_bm25()
            print(f"Restored {len(self.local_corpus)} entries from local corpus.")
        else:
            print("Warning: No corpus archive file found.")
