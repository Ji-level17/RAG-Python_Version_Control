<div align="center">

# 基于版本感知的 RAG FastAPI 代码生成系统

**多策略检索增强生成系统，用于生成版本正确的 Python 代码**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B.svg)](streamlit_app.py)

[English](README.md) | **中文**

</div>

---

## 📕 目录

- [项目简介](#-项目简介)
- [核心特性](#-核心特性)
- [系统架构](#-系统架构)
- [检索策略](#-检索策略)
- [快速开始](#-快速开始)
- [运行实验](#-运行实验)
- [实验结果](#-实验结果)
- [模型与许可证](#-模型与许可证)
- [项目结构](#-项目结构)

---

## 💡 项目简介

本项目构建了一个**版本感知的 RAG 流水线**，能够为指定的 Python 版本生成正确的 FastAPI 代码。FastAPI 跨版本迭代迅速，LLM 在没有版本特定文档的情况下，往往会生成过时或版本不兼容的代码。

我们对比了**四种检索策略** — 从简单的稠密向量搜索到基于导入图的依赖扩展 — 并使用 [RAGAS](https://github.com/explodinggradients/ragas)、[HumanEval](https://github.com/openai/human-eval) 和 [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) 基准进行评估。

---

## 🌟 核心特性

- **🔍 四种检索策略** — 基线稠密检索、混合检索（BM25 + Dense + RRF）、层次化检索（父子分组）、导入图检索（依赖扩展）
- **📌 版本感知过滤** — 每个代码块标注 `min_version` 元数据，支持目标版本定向检索
- **📊 全面评估** — RAGAS（4 指标 × 85 测试用例 × 4 策略）、HumanEval（164 题）、MBPP（257 题）
- **🧪 多轮对比** — 4 轮完整 RAGAS 评估，采用不同配置以确保结果稳健性
- **⚡ 交互式演示** — Streamlit 界面，实时对比不同策略的检索与生成效果
- **🔧 可复现** — 固定随机种子、`temperature=0`、预计算结果已包含

---

## 🔎 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI 源码仓库                            │
└──────────────────────────┬──────────────────────────────────┘
                           │ Tree-sitter AST 解析
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            语料库（695 代码块，768 维嵌入向量）                  │
│       嵌入模型：st-codesearch-distilroberta-base               │
└──────┬──────────┬────────────┬───────────┬──────────────────┘
       │          │            │           │
       ▼          ▼            ▼           ▼
  ┌─────────┐ ┌────────┐ ┌──────────┐ ┌───────────┐
  │基线稠密  │ │混合检索 │ │层次化    │ │导入图     │
  │检索      │ │BM25+   │ │检索      │ │检索       │
  │         │ │Dense+  │ │父级→    │ │依赖      │
  │余弦相似度│ │RRF     │ │子级      │ │扩展       │
  └────┬────┘ └───┬────┘ └────┬─────┘ └─────┬─────┘
       │          │            │             │
       └──────────┴─────┬──────┴─────────────┘
                        ▼
              ┌───────────────────┐
              │  Gemini 2.5 Flash │
              │    代码生成        │
              └────────┬──────────┘
                       ▼
              ┌───────────────────┐
              │   RAGAS 评估      │
              │  （LLM 评判）     │
              └───────────────────┘
```

---

## 🎯 检索策略

| # | 策略 | 描述 | 延迟 |
|:-:|------|------|:----:|
| 1 | **基线稠密 RAG** | 基于稠密嵌入的余弦相似度检索 | ~0.3s |
| 2 | **混合 RAG** | BM25 + 稠密嵌入，通过倒数排序融合（RRF，k=60）结合 | ~0.5s |
| 3 | **层次化 RAG** | 父子文件分组 → 父级检索 → 子级重排序 | ~3.9s |
| 4 | **导入图 RAG** | 稠密检索 + 导入依赖图扩展 → 重排序 | ~10.7s |

---

## 🎬 快速开始

### 环境要求

- Python >= 3.10（已在 3.11、3.13 上测试）
- [Pinecone](https://app.pinecone.io) API 密钥（向量存储）
- [Gemini](https://aistudio.google.com/apikey) API 密钥（生成与评估）

### 安装

```bash
# 克隆仓库
git clone https://github.com/Ji-level17/RAG-Python_Version_Control.git
cd RAG-Python_Version_Control
git checkout v2

# 创建虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 配置

在项目根目录创建 `.env` 文件：

```env
PINECONE_API_KEY=your_pinecone_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### 数据准备

语料库（`local_corpus.json`，695 个代码块）和导入图（`import_graph.json`）已**预构建并包含**在仓库中。它们通过对 [FastAPI 源码](https://github.com/tiangolo/fastapi) 运行 `repo_processor.py`（Tree-sitter AST 解析）生成。

如需从头构建：

```bash
git clone https://github.com/tiangolo/fastapi.git
python repo_processor.py
python rebuild_corpus.py
python rebuild_import_graph.py
```

---

## 🚀 运行实验

### RAGAS 评估

评估 4 种策略 × 85 条测试用例，指标包括忠实度、回答相关性、上下文精确率和上下文召回率：

```bash
python eval_rag_strategies.py \
    --gemini-key $GEMINI_API_KEY \
    --pinecone-key $PINECONE_API_KEY \
    --strategies baseline advanced hierarchical import_graph \
    --num-records 85 \
    --output-dir results_v9
```

### 检索指标

无需 LLM 评判，测量命中率、MRR、P@k 和 Recall@k：

```bash
python eval_retrieval_metrics.py \
    --pinecone-key $PINECONE_API_KEY \
    --output-dir results_v9
```

### 代码生成基准

运行 HumanEval（164 题）和 MBPP（257 题），计算 pass@1：

```bash
python eval_humaneval_mbpp.py \
    --gemini-key $GEMINI_API_KEY \
    --output-dir results_v9
```

### 交互式演示

```bash
streamlit run streamlit_app.py
```

在 `http://localhost:8501` 打开浏览器，选择策略、输入 FastAPI 问题，实时查看检索上下文和生成结果。

---

## 📊 实验结果

### RAGAS 评分（Gemini 2.5 Flash，n=85，top-k=5）

| 策略 | 忠实度 | 回答相关性 | 上下文精确率 | 上下文召回率 |
|------|:------:|:--------:|:----------:|:----------:|
| 基线稠密 | **0.634** | 0.768 | 0.735 | 0.665 |
| 混合 | 0.619 | 0.821 | 0.751 | **0.766** |
| 层次化 | 0.479 | **0.836** | 0.678 | 0.611 |
| 导入图 | 0.504 | 0.826 | **0.762** | 0.684 |

### 代码生成基准

| 基准 | 题目数 | Pass@1 |
|------|:-----:|:------:|
| HumanEval | 164 | **72.6%** |
| MBPP | 257 | **72.4%** |

---

## 📜 模型与许可证

| 组件 | 模型 | 许可证 |
|------|------|--------|
| 代码嵌入模型 | `flax-sentence-embeddings/st-codesearch-distilroberta-base`（768 维） | Apache 2.0 |
| 交叉编码器重排序 | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Apache 2.0 |
| QLoRA 基础模型 | `Qwen/Qwen2.5-Coder-7B` + 适配器 `Ivan17Ji/qwen-lora-250` | Apache 2.0 |
| 微调数据集 | StarInstruct（checkpoint-25） | Apache 2.0 |
| 生成与评估 | Gemini 2.5 Flash | Google ToS |
| 目标代码库 | [FastAPI](https://github.com/tiangolo/fastapi) | MIT |
| 评测基准 | HumanEval、MBPP | MIT |

---

## 📁 项目结构

```
├── rag_pipeline.py              # 核心 RAG 系统（VersionControlRAG）
├── hierarchical_rag.py          # 策略 3：层次化检索
├── rag_strategies.py            # 策略 4：导入图检索
├── repo_processor.py            # FastAPI 仓库摄入（AST 分块）
│
├── eval_rag_strategies.py       # RAGAS 评估（4 策略 × 85 条）
├── eval_retrieval_metrics.py    # 检索指标（命中率、MRR、P@k）
├── eval_humaneval_mbpp.py       # HumanEval & MBPP 基准
│
├── local_corpus.json            # 695 个 FastAPI 代码块
├── import_graph.json            # 模块依赖图
├── fastapi_golden_set.json      # 黄金 QA 对
├── ragas_dataset.json           # RAGAS 评估数据集
│
├── results_v9/                  # 评估结果与图表
├── streamlit_app.py             # 交互式演示（4 种策略）
├── requirements.txt             # Python 依赖
└── .env                         # API 密钥（未跟踪）
```

---

## 🔬 可复现性

| 设置 | 值 |
|------|-----|
| 随机种子 | `42` |
| 生成温度 | `0` |
| RAGAS 评判模型 | Gemini 2.5 Flash（`thinking_budget=0`） |
| 测试集 | 85 条（561 对黄金集，70/15/15 划分） |
| 向量索引 | Pinecone `python-rag-v3`（768 维，余弦） |

预计算结果存储在 `results_v9/` 中，可直接验证无需重新运行实验。

---

## ⚙️ 硬件要求

| 组件 | 要求 |
|------|------|
| 嵌入模型 + BM25 | 仅 CPU（~2 GB 内存） |
| 交叉编码器重排序 | 仅 CPU |
| Gemini API | 无需本地 GPU |
| QLoRA 微调（可选） | 44+ GB 显存（A40） |
