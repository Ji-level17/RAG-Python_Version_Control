import os
import uuid
import re
import json
import jieba
import torch
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class VersionControlRAG:
    def __init__(self, pinecone_key, model_path="Qwen/Qwen2.5-7B-Coder", adapter_path=None, index_name="python-rag-v3"):
        print("🚢 正在初始化版本控制 RAG 载体...") # Initializing Version-Controlled RAG Carrier...
        print("🚢 Initializing Version-Controlled RAG Carrier...")
        self.pc = Pinecone(api_key=pinecone_key)
        
        # 设置 Pinecone Index
        # Pinecone Index Setup
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(name=index_name, dimension=384, metric="cosine",
                                spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        self.index = self.pc.Index(index_name)
        
        # 设置 Models
        # Models Setup
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.reranker = CrossEncoder('BAAI/bge-reranker-base')
        
        # 设置 LLM 大脑 (Qwen + QLoRA)
        # LLM Brain Setup (Qwen + QLoRA)
        print(f"🧠 正在加载 Qwen 大脑 {model_path}...") # Loading Qwen Brain...
        print(f"🧠 Loading Qwen Brain from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        )
        if adapter_path:
            print(f"🧬 应用 QLoRA Adapter {adapter_path}...")
            print(f"🧬 Applying QLoRA Adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            self.model = base_model
        self.model.eval()

        # Tree-sitter 设置
        # Tree-sitter Setup
        self.PY_LANGUAGE = Language(tspython.language())
        self.parser = Parser(self.PY_LANGUAGE)
        
        self.local_corpus = [] 
        self.bm25_model = None
        print("✅ System Ready!\n")

    def _parse_version(self, version_str):
        """Helper: Converts '3.12' to 312 for safe numerical filtering."""
        try:
            if not version_str or not isinstance(version_str, str): return 300
            # 清洗字符串，提取版本号数字部分
            # Clean string like 'Python 3.12' -> '3.12'
            version_str = re.search(r'\d+\.\d+', version_str).group()
            parts = version_str.split('.')
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            return major * 100 + minor
        except:
            return 308 # Default fallback
                       # 默认回退到 3.8，兼容性更好一些

    def extract_functions(self, file_path):
        """Tree-sitter recursive extraction."""
        # tree-sitter 回数法提取函数定义
        with open(file_path, 'r', encoding='utf-8') as f:
            code_bytes = f.read().encode('utf-8')
        tree = self.parser.parse(code_bytes)
        functions = []
        def traverse(node):
            if node.type == 'function_definition':
                functions.append(code_bytes[node.start_byte:node.end_byte].decode('utf-8'))
                return 
            for child in node.children: traverse(child)
        traverse(tree.root_node)
        return functions

    def get_qwen_metadata(self, code_snippet):
        """English Prompt for better QLoRA performance."""
        # English的提示语能让 QLoRA 更好地发挥，毕竟它是用英文训练的
        prompt = f"### Role\nPython Version Expert. Output JSON only.\n### Task\nIdentify min_version (e.g. '3.12') and summary_zh for:\n{code_snippet}\n### Response\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=128, temperature=0.01)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        try:
            return json.loads(re.search(r'\{.*\}', response, re.DOTALL).group())
        except:
            return {"min_version": "3.12", "summary_zh": "Analysis fallback"}

    def ingest_data(self, doc_id, code_content, min_version, summary):
        vector_values = self.embedder.encode(summary + " " + code_content).tolist()
        safe_version_num = self._parse_version(min_version) 
        metadata = {"min_version": safe_version_num, "summary": summary, "code": code_content}
        self.index.upsert(vectors=[{"id": doc_id, "values": vector_values, "metadata": metadata}])
        self.local_corpus.append({"id": doc_id, "content": code_content, "metadata": metadata})
        tokenized_corpus = [list(jieba.cut(doc["content"])) for doc in self.local_corpus]
        self.bm25_model = BM25Okapi(tokenized_corpus)
        
    def retrieve_complex(self, query, target_version=None, top_k=3, mode="advanced"):
        """
        mode="baseline":  Vanilla Vector Search
        mode="advanced":  (Vector + BM25 + CrossEncoder)
        """
        # 1. 基础向量检索 (所有模式共有)
        query_vector = self.embedder.encode(query).tolist()
        
        filter_dict = {"min_version": {"$lte": self._parse_version(target_version)}} if target_version else {}
        
        pinecone_res = self.index.query(
            vector=query_vector, 
            filter=filter_dict, 
            top_k=top_k if mode=="baseline" else 10, 
            include_metadata=True
        )
        
        # 如果是 Baseline 模式，则跳过后续的 BM25 和 Reranker 逻辑，直接返回结果
        # If it's Baseline mode, skip the BM25 and Reranker logic and return results directly
        if mode == "baseline":
            print("🕯️ [Baseline Mode]")
            return [match['metadata']['code'] for match in pinecone_res['matches']]

        # 进阶逻辑 (Hybrid + Rerank)
        # Advanced Logic (Hybrid + Rerank)
        print("🔥 [Advanced Mode]") 
        
        # 增加空仓保护：如果 BM25 还没初始化，就跳过它
        # Add empty corpus protection: if BM25 is not initialized yet, skip it
        bm25_docs = {}
        if self.bm25_model is not None:
            tokenized_query = list(jieba.cut(query))
            bm25_scores = self.bm25_model.get_scores(tokenized_query)
            bm25_docs = {self.local_corpus[i]["id"]: self.local_corpus[i]["content"] 
                         for i, s in enumerate(bm25_scores) if s > 0 
                         if not target_version or self.local_corpus[i]["metadata"]["min_version"] <= self._parse_version(target_version)}
        else:
    
            print("⚠️ [警告] BM25 词库为空！(是不是刚重启还没灌入数据？) 本次仅对向量结果进行重排。") # Warning: BM25 corpus is empty! (Is it just restarted and not ingested data yet? This time only reranking vector results.
            print("⚠️ [Warning] BM25 corpus is empty! (Is it just restarted and not ingested data yet? This time only reranking vector results.")   
        # 合并去重
        # Combine and deduplicate results
        pinecone_docs = {match['id']: match['metadata']['code'] for match in pinecone_res['matches']}
        candidates = {**pinecone_docs, **bm25_docs}
        
        if not candidates: 
            return []

        # Cross-Encoder 重排
        # Cross-Encoder Reranking
        pairs = [[query, code] for code in candidates.values()]
        scores = self.reranker.predict(pairs)
        final_results = [c for c, s in sorted(zip(candidates.values(), scores), key=lambda x: x[1], reverse=True)]
        
        return final_results[:top_k]

    def save_local_corpus(self, filepath="local_corpus.json"):
        """把本地记忆存入硬盘，下次重启秒切状态"""
        # Save local memory to disk for instant state restoration on next restart
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.local_corpus, f, ensure_ascii=False, indent=2)
        print("💾 本地语料库已存档！") # local corpus saved
        print("💾 Local corpus saved!")

    def load_local_corpus(self, filepath="local_corpus.json"):
        """重启 Notebook 后，恢复 BM25"""
        # After restarting the Notebook, instantly restore BM25 radar
        import json, os, jieba
        from rank_bm25 import BM25Okapi
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.local_corpus = json.load(f)
            # 重建 BM25 索引
            # rebuild BM25 index
            tokenized_corpus = [list(jieba.cut(doc["content"])) for doc in self.local_corpus]
            self.bm25_model = BM25Okapi(tokenized_corpus)
            print(f"成功恢复 {len(self.local_corpus)} 条本地语料") # success
            print(f"Successfully restored {len(self.local_corpus)} local corpus entries")
        else:
            print("⚠️ 未找到存档文件。") # no archive found
            print("⚠️ No archive file found.")