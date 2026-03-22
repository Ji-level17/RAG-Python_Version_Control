import os
import re
import json
import torch
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

UPSERT_BATCH_SIZE = 100


def _has_chinese(text: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in text)


class VersionControlRAG:
    def __init__(self, pinecone_key, model_path="Qwen/Qwen2.5-Coder-7B", adapter_path=None,
                 index_name="python-rag-v3", ingest_only=False):
        """
        ingest_only=True  — skip loading Qwen and BGE reranker entirely.
                            Uses CPU for embeddings. Use this for re-upsert to avoid OOM.
                            ~0.3 GB VRAM instead of ~15 GB.
        ingest_only=False — full pipeline (default). Loads Qwen + adapter on GPU.
        """
        print("Initializing Version-Controlled RAG Carrier...")
        self.pc = Pinecone(api_key=pinecone_key)
        self.ingest_only = ingest_only

        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(name=index_name, dimension=768, metric="cosine",
                                spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        self.index = self.pc.Index(index_name)

        # Embedder always runs on CPU during ingest to avoid competing with Qwen for VRAM.
        # During retrieval (ingest_only=False) it can stay on CPU too — encoding a single
        # query takes <50ms on CPU, not worth the VRAM cost of keeping it on GPU.
        embed_device = "cpu"
        self.embedder = SentenceTransformer(
            'flax-sentence-embeddings/st-codesearch-distilroberta-base',
            device=embed_device
        )
        print(f"Embedder loaded on {embed_device}")

        if ingest_only:
            # Lightweight mode — no Qwen, no reranker, no tree-sitter
            self.model     = None
            self.tokenizer = None
            self.reranker  = None
            self.parser    = None
            print("Ingest-only mode: Qwen and reranker skipped (saves ~15 GB VRAM)\n")
        else:
            self.reranker = CrossEncoder('BAAI/bge-reranker-base')

            from transformers import AutoModelForCausalLM, AutoTokenizer
            import tree_sitter_python as tspython
            from tree_sitter import Language, Parser

            print(f"Loading Qwen from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
            )
            if adapter_path:
                from peft import PeftModel
                print(f"Applying QLoRA adapter from {adapter_path}...")
                self.model = PeftModel.from_pretrained(base_model, adapter_path)
            else:
                self.model = base_model
            self.model.eval()

            self.parser = Parser(Language(tspython.language()))

        self.local_corpus = []
        self._corpus_ids = set()
        self.bm25_model = None
        self._upsert_buffer = []
        print("System ready.\n")

    def free_gpu_memory(self):
        """Release Qwen from GPU after ingest or generation. Call before switching modes."""
        import gc
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.reranker is not None:
            del self.reranker
            self.reranker = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"GPU memory freed. Available: {free:.1f} GB")

    def _parse_version(self, version_str):
        """Convert a version string like '3.12' to an integer (312) for numerical filtering."""
        try:
            if not version_str or not isinstance(version_str, str):
                return 0
            version_str = re.search(r'\d+\.\d+', version_str).group()
            parts = version_str.split('.')
            return int(parts[0]) * 100 + int(parts[1])
        except Exception:
            return 308

    def _build_embed_text(self, summary: str, code_content: str) -> str:
        """
        FIX 2: smart embedding text.
        - If summary is English/code (not Chinese) → prepend it: gives the embedder
          a semantic bridge between natural-language queries and code.
        - If summary is Chinese → embed code only: Chinese text distorts the embedding
          space and hurts English query matching.
        """
        if summary and not _has_chinese(summary):
            # Strip the leading [filepath] prefix, keep the descriptive part
            clean = re.sub(r'^\[[^\]]+\]\s*', '', summary).strip()
            if clean:
                return clean + "\n" + code_content
        return code_content

    def extract_functions(self, file_path):
        """Extract all function definitions from a Python file using Tree-sitter."""
        with open(file_path, 'rb') as f:
            code_bytes = f.read()
        tree = self.parser.parse(code_bytes)
        functions = []
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type == 'function_definition':
                functions.append(code_bytes[node.start_byte:node.end_byte].decode('utf-8'))
            else:
                stack.extend(reversed(node.children))
        return functions

    def get_qwen_metadata(self, code_snippet):
        """Use the Qwen model to extract min_version and a summary from a code snippet."""
        prompt = f"### Role\nPython Version Expert. Output JSON only.\n### Task\nIdentify min_version (e.g. '3.12') and summary for:\n{code_snippet}\n### Response\n"
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

    def ingest_data(self, doc_id, code_content, min_version, summary, tier="docs_src"):
        """
        Buffer a document for ingestion.
        tier: "docs_src" | "fastapi" | "scripts"
              scripts/ chunks are skipped — build tooling is pure noise for retrieval.
        """
        if doc_id in self._corpus_ids:
            return
        if tier == "scripts":
            return  # excluded: build tooling crowds out tutorial results

        embed_text = self._build_embed_text(summary, code_content)
        vector_values = self.embedder.encode(embed_text).tolist()

        safe_version_num = self._parse_version(min_version)
        metadata = {
            "min_version": safe_version_num,
            "summary":     summary,
            "code":        code_content,
            "tier":        tier,
        }

        self._upsert_buffer.append({"id": doc_id, "values": vector_values, "metadata": metadata})
        self.local_corpus.append({"id": doc_id, "content": code_content, "metadata": metadata})
        self._corpus_ids.add(doc_id)

        if len(self._upsert_buffer) >= UPSERT_BATCH_SIZE:
            self.flush_upsert_buffer()

    def flush_upsert_buffer(self):
        if not self._upsert_buffer:
            return
        try:
            self.index.upsert(vectors=self._upsert_buffer)
            self._upsert_buffer = []
        except Exception as e:
            print(f"Warning: Pinecone upsert failed ({len(self._upsert_buffer)} vectors): {e}")
            raise

    def rebuild_bm25(self):
        if not self.local_corpus:
            print("Warning: local_corpus is empty, skipping BM25 rebuild.")
            return
        tokenized_corpus = [re.findall(r'[a-zA-Z_]\w*|\d+', doc["content"]) for doc in self.local_corpus]
        self.bm25_model = BM25Okapi(tokenized_corpus)
        print(f"BM25 index rebuilt with {len(self.local_corpus)} documents.")

    def _generate_hypothesis(self, query: str, max_new_tokens: int = 150) -> str:
        if self.ingest_only or self.model is None:
            raise RuntimeError("_generate_hypothesis requires ingest_only=False.")
        """
        FIX 3: HyDE — generate a short hypothetical FastAPI code snippet
        that would answer the query, then embed THAT instead of the query.

        A generated snippet contains function names, parameter names, and imports
        that actually appear in the corpus — massively closing the semantic gap
        between 'share dependencies across routes' and 'common_parameters'.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a FastAPI expert. Write a SHORT Python code snippet "
                    "that directly answers the question. Return ONLY code, no explanation."
                )
            },
            {"role": "user", "content": query}
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        hypothesis = self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        return hypothesis if hypothesis else query   # fallback to raw query if model fails

    def retrieve_complex(self, query, target_version=None, top_k=3, mode="advanced"):
        """
        Retrieve relevant code snippets.

        mode='baseline'  — vector search only (raw query embedding)
        mode='advanced'  — hybrid (vector + BM25) with cross-encoder reranking
        mode='hyde'      — HyDE: generate hypothesis → embed that → hybrid + reranking
                           Best for concept queries where function names are unknown.
        """
        if self.ingest_only and mode == "hyde":
            raise RuntimeError(
                "HyDE requires ingest_only=False (Qwen needed to generate hypothesis). "
                "Use mode='baseline' or mode='advanced' with ingest_only=True."
            )
        if mode not in ("baseline", "advanced", "hyde"):
            raise ValueError(f"Invalid mode '{mode}'. Expected 'baseline', 'advanced', or 'hyde'.")

        # FIX 3: HyDE — embed a generated hypothesis instead of the raw query
        if mode == "hyde":
            print("[HyDE Mode] Generating hypothesis...")
            hypothesis = self._generate_hypothesis(query)
            print(f"  Hypothesis: {hypothesis[:120].strip()!r}")
            query_vector = self.embedder.encode(hypothesis).tolist()
        else:
            query_vector = self.embedder.encode(query).tolist()

        # Search all tiers (docs_src + fastapi). scripts/ excluded at ingest time.
        filter_dict = {}

        fetch_k = max(top_k * 2, 6) if mode == "baseline" else max(top_k * 3, 15)
        pinecone_res = self.index.query(
            vector=query_vector,
            filter=filter_dict,
            top_k=fetch_k,
            include_metadata=True
        )

        if mode == "baseline":
            print("[Baseline Mode]")
            return [match['metadata']['code'] for match in pinecone_res['matches']]

        # Advanced / HyDE: hybrid + rerank
        bm25_docs = {}
        if self.bm25_model is not None:
            tokenized_query = re.findall(r'[a-zA-Z_]\w*|\d+', query)
            bm25_scores = self.bm25_model.get_scores(tokenized_query)
            # Take only top-20 BM25 hits to avoid sending hundreds to reranker
            ranked_bm25 = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
            bm25_docs = {
                self.local_corpus[i]["id"]: self.local_corpus[i]["content"]
                for i, s in ranked_bm25[:fetch_k]
                if s > 0
            }
        else:
            print("Warning: BM25 corpus is empty. Call rebuild_bm25() or load_local_corpus() first.")

        pinecone_docs = {match['id']: match['metadata']['code'] for match in pinecone_res['matches']}
        candidates = {**pinecone_docs, **bm25_docs}

        if not candidates:
            return []

        if self.reranker is not None:
            pairs = [[query, code] for code in candidates.values()]
            scores = self.reranker.predict(pairs)
            scored = sorted(zip(candidates.values(), scores), key=lambda x: x[1], reverse=True)
            # Keep only chunks with reranker score > threshold (filter noise)
            rerank_threshold = -5.0
            final_results = [c for c, s in scored if s > rerank_threshold]
            if not final_results:
                final_results = [scored[0][0]]  # keep at least the best one
        else:
            # RRF (Reciprocal Rank Fusion) to fairly combine dense + BM25 rankings
            # Each source contributes 1/(k+rank) where k=60 is a smoothing constant
            rrf_k = 60
            rrf_scores = {}

            # Dense ranking from Pinecone
            for rank, match in enumerate(pinecone_res['matches']):
                rrf_scores[match['id']] = 1.0 / (rrf_k + rank + 1)

            # BM25 ranking
            if self.bm25_model is not None:
                tokenized_query = re.findall(r'[a-zA-Z_]\w*|\d+', query)
                bm25_scores = self.bm25_model.get_scores(tokenized_query)
                ranked_bm25 = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
                for rank, (i, s) in enumerate(ranked_bm25[:fetch_k]):
                    if s > 0:
                        doc_id = self.local_corpus[i]["id"]
                        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)

            # Sort candidates by RRF score
            scored = sorted(
                [(doc_id, candidates[doc_id], rrf_scores.get(doc_id, 0))
                 for doc_id in candidates],
                key=lambda x: x[2], reverse=True
            )
            final_results = [code for _, code, _ in scored]
            # )
        return final_results[:top_k]

    def generate_answer(self, query: str, target_version: str = None,
                        top_k: int = 5, mode: str = "advanced") -> dict:
        """
        Retrieve contexts then generate an answer with Qwen.
        Returns {"answer": str, "contexts": list[str]} for RAGAS evaluation.

        Polymorphic: when called on HierarchicalRAG or ImportGraphRAG, their
        retrieve_complex() is used automatically.
        """
        if self.ingest_only or self.model is None:
            raise RuntimeError(
                "generate_answer() requires ingest_only=False (Qwen must be loaded). "
                "For frontier-model generation, call retrieve_complex() separately and "
                "pass contexts to your Groq/OpenAI client."
            )

        contexts = self.retrieve_complex(query, target_version=target_version,
                                         top_k=top_k, mode=mode)

        if not contexts:
            return {"answer": "No relevant code found in context.", "contexts": []}

        context_text = "\n\n---\n\n".join(contexts)
        version_note = f" targeting Python {target_version}" if target_version else ""

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a FastAPI and Python expert{version_note}. "
                    "Answer the question using ONLY the provided code context. "
                    "If the answer is not in the context, say 'Not found in context'."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:",
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        answer = self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        return {"answer": answer, "contexts": contexts}

    def save_local_corpus(self, filepath="local_corpus.json"):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.local_corpus, f, ensure_ascii=False, indent=2)
        print("Local corpus saved.")

    def load_local_corpus(self, filepath="local_corpus.json"):
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.local_corpus = json.load(f)
            self._corpus_ids = {doc["id"] for doc in self.local_corpus}
            self.rebuild_bm25()
            print(f"Restored {len(self.local_corpus)} entries from local corpus.")
        else:
            print("Warning: No corpus archive file found.")
