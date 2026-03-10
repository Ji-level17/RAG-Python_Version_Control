# 🚀 Version-Aware Code RAG Pipeline | 支持版本控制的代码 RAG 系统

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Applied ML](https://img.shields.io/badge/Course-Applied%20Machine%20Learning-orange)

**English** | [中文版](#中文说明)

This repository contains a fully modularized Retrieval-Augmented Generation (RAG) pipeline designed specifically for codebases (e.g., FastAPI). It features a hybrid search mechanism (Vector + BM25), a Cross-Encoder reranker, and uses a custom QLoRA fine-tuned Qwen model to automatically tag code snippets with their minimum required Python version.

## ✨ Core Features
* **Modular Design:** Data ingestion (`RepoProcessor`) and retrieval (`VersionControlRAG`) are cleanly separated.
* **Hybrid Search & Reranking:** Combines Pinecone vector search with local BM25, refined by a Cross-Encoder.
* **Version Control:** Filters out code snippets that are incompatible with the user's target Python version.
* **Algorithm Lab (For Classmates):** Easily plug in and test your own retrieval algorithms without rewriting the boilerplate code!

## 📦 Quick Start (Out of the Box)

If you just want to run the existing baseline or advanced RAG, it only takes a few lines of code:

```python
from rag_pipeline import VersionControlRAG

# 1. Initialize the pipeline (Handles Pinecone connection, Embeddings, and Qwen)
rag = VersionControlRAG(
    pinecone_key="YOUR_PINECONE_KEY", 
    model_path="Qwen/Qwen2.5-Coder-7B", 
    adapter_path="Ivan17Ji/qwen-lora-250"
)

# 2. Search using the Advanced Hybrid mode
results = rag.retrieve_complex("How to define an APIRouter?", target_version="3.12", mode="advanced")
print(results[0])
```

## 🛠️ For Teammates: How to Test Your Own RAG Algorithms?

**Don't write everything from scratch!** I have encapsulated the dirty work (connecting to DB, chunking, loading models). To test a new retrieval algorithm for our Applied ML assignment, simply **inherit** my base class and override the `retrieve_complex` method:

```python
from rag_pipeline import VersionControlRAG

class MyCustomRAG(VersionControlRAG):
    def retrieve_complex(self, query, target_version, top_k=3, mode="custom"):
        print("🚀 Running my awesome custom algorithm...")
        # Write your own TF-IDF / PageRank / GraphRAG logic here!
        # You can directly access self.local_corpus and self.index
        return ["My algorithm's result"]

# Test your algorithm
my_rag = MyCustomRAG(pinecone_key="...", model_path="...", adapter_path="...")
my_rag.retrieve_complex("Test query", "3.12")
```

---

<a name="中文说明"></a>
## 🇨🇳 中文说明

本项目包含一个完全模块化的 RAG（检索增强生成）管道，专门为代码库（如 FastAPI）设计。它集成了混合检索（向量 + BM25）、交叉编码器（Cross-Encoder）重排，并使用微调后的 Qwen 模型（QLoRA）自动为代码片段打上 Python 版本标签。

## ✨ 核心特性
* **模块化设计**：数据摄取（`RepoProcessor`）和检索逻辑（`VersionControlRAG`）完全解耦。
* **混合检索与重排**：结合 Pinecone 向量检索与本地 BM25，并通过 Cross-Encoder 提升准确率。
* **版本控制**：精准过滤掉不兼容目标 Python 版本的代码。
* **算法实验室（专为同学准备）**：无需重写繁琐的基础代码，即可轻松接入并测试你自己的检索算法！

## 📦 快速开始（开箱即用）

如果你只想测试现有的 Baseline 或 Advanced RAG，只需几行代码：

```python
from rag_pipeline import VersionControlRAG

# 1. 初始化航母（自动处理数据库连接、向量模型和 Qwen 加载）
rag = VersionControlRAG(
    pinecone_key="YOUR_PINECONE_KEY", 
    model_path="Qwen/Qwen2.5-Coder-7B", 
    adapter_path="Ivan17Ji/qwen-lora-250"
)

# 2. 使用进阶混合模式进行搜索
results = rag.retrieve_complex("How to define an APIRouter?", target_version="3.12", mode="advanced")
print(results[0])
```

## 🛠️ 给组员的指南：如何测试你自己的 RAG 算法？

**千万不要从头写脏活累活！** 我已经封装好了底层设施（连接数据库、文本切分、加载大模型等）。为了完成我们 Applied ML 的作业要求，你只需要**继承**我的基类，并重写 `retrieve_complex` 方法即可：

```python
from rag_pipeline import VersionControlRAG

class MyCustomRAG(VersionControlRAG):
    def retrieve_complex(self, query, target_version, top_k=3, mode="custom"):
        print("🚀 正在运行我开发的算法...")
        # 在这里写你自己的 TF-IDF / 各种神仙匹配逻辑！
        # 你可以直接调用 self.local_corpus 和 self.index 获取数据
        return ["这是我的算法找出来的结果"]

# 测试你的新算法
my_rag = MyCustomRAG(pinecone_key="...", model_path="...", adapter_path="...")
my_rag.retrieve_complex("测试问题", "3.12")
```
