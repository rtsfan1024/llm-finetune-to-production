# 大模型微调与落地

本仓库涵盖从 NLP 基础算法到 RAG（检索增强生成）文档问答系统的全链路实践。

---

## 📁 项目结构

```
第2模块/
├── 1. NLP 基础
│   ├── n-gram.py                  # N-gram 自回归文本生成
│   └── lda.py                     # LDA 主题模型
│
├── 2. Transformer 架构
│   ├── mini_transformer.py        # 完整 Transformer（Encoder-Decoder）
│   ├── mini_decoder_only.py       # Decoder-Only 模型（GPT 架构）
│   └── trans_onit.py              # Transformer 翻译训练与推断
│
├── 3. 训练与评估
│   ├── train_decoder_only.py      # Decoder-Only 模型训练与生成
│   ├── scaling_law.py             # Scaling Law 算力估算（6ND 公式）
│   └── qwen2.5_1.5b_lora.py      # Qwen2.5-1.5B 意图分类 LoRA 微调
│
├── 4. RAG 文档问答系统（分阶段实现）
│   ├── based_docs_qa1.py          # 阶段1：文档读取与溯源
│   ├── based_docs_qa2.py          # 阶段2：Embedding 向量生成
│   ├── based_docs_qa3.py          # 阶段3：Qdrant 向量存储
│   ├── based_docs_qa4.py          # 阶段4：检索 + LLM 问答 + 溯源
│   ├── based_docs_qa5.py          # 阶段5：LoRA 微调（DeepSeek-7B）
│   ├── based_docs_qa6.py          # LangChain 版全流程
│   └── based_docs_qa.py           # 完整生产版 RAG（语义分段 + BGE + Qdrant + DeepSeek）
│
├── 数据与模型（见下方"大文件说明"）
│   ├── books/                     # 测试文档（西游记.pdf / 水浒传.docx / 三国演义.txt）
│   ├── bank_intent_data/          # 银行意图分类数据集
│   ├── bank_intent_data.zip       # 上述数据集的压缩包
│   ├── mini_translation_pairs.txt # 中英平行翻译语料
│   ├── triple_data.json           # RAG 问答三元组训练数据
│   ├── Qwen2.5-1.5B-Instruct/    # Qwen2.5-1.5B 预训练模型
│   ├── bge-large-zh-v1.5/         # BAAI BGE 中文 Embedding 模型
│   ├── ckpt/                      # Transformer 翻译模型 checkpoint
│   └── ckpt_gpt/                  # Decoder-Only 模型 checkpoint
│
└── README.md
```

---

## 🧩 模块详解

### 1. NLP 基础

| 文件 | 功能 | 说明 |
|------|------|------|
| [n-gram.py](n-gram.py) | N-gram 文本生成 | 基于语料库的自回归逐字生成，演示最基础的语言模型思想 |
| [lda.py](lda.py) | LDA 主题模型 | 使用 sklearn 的 LDA 对中文文本做主题聚类，结合 jieba 分词和规则引擎自动命名主题 |

### 2. Transformer 架构

| 文件 | 功能 | 说明 |
|------|------|------|
| [mini_transformer.py](mini_transformer.py) | 完整 Transformer | 从零实现 Encoder-Decoder 架构，包含正弦位置编码、多头注意力、FFN、残差+LayerNorm，支持 CUDA/MPS/CPU |
| [mini_decoder_only.py](mini_decoder_only.py) | Decoder-Only (GPT) | 仅解码器架构，含 Masked Self-Attention、权重绑定（tie weights）、温度采样生成，代码注释标注了 DEC-1 ~ DEC-11 完整流程 |

### 3. 训练与评估

| 文件 | 功能 | 说明 |
|------|------|------|
| [trans_onit.py](trans_onit.py) | Transformer 翻译训练 | 基于 mini_transformer 的中→英翻译训练，含字符级分词、贪心解码推断，支持 `--train` / `--translate` 命令行 |
| [train_decoder_only.py](train_decoder_only.py) | Decoder-Only 训练 | 基于 mini_decoder_only 的语言模型训练，支持 `--train` / `--prompt` 命令行 |
| [scaling_law.py](scaling_law.py) | Scaling Law 算力估算 | 基于 6ND 公式计算训练所需 GPU 数量，内置 A100/A800/H100/H800 全系列 GPU 算力参数 |
| [qwen2.5_1.5b_lora.py](qwen2.5_1.5b_lora.py) | Qwen2.5 LoRA 微调 | 银行意图分类任务（5类），对比 LoRA 微调前后的 Macro-F1 指标，基于 PEFT + Transformers Trainer |

### 4. RAG 文档问答系统

六个脚本按阶段递进，逐步构建完整的 RAG 系统：

| 阶段 | 文件 | 功能 | 关键技术 |
|------|------|------|----------|
| 1 | [based_docs_qa1.py](based_docs_qa1.py) | 文档读取与溯源 | pdfplumber / python-docx / chardet，按页/段/行拆分并携带 source 元数据 |
| 2 | [based_docs_qa2.py](based_docs_qa2.py) | Embedding 向量生成 | BAAI/bge-large-zh-v1.5 (ModelScope)，768 维归一化句向量 |
| 3 | [based_docs_qa3.py](based_docs_qa3.py) | Qdrant 向量存储 | Qdrant 本地文件模式，余弦距离，批量插入向量+元数据 |
| 4 | [based_docs_qa4.py](based_docs_qa4.py) | 检索 + LLM 问答 | DeepSeek-7B (4bit 量化)，Top-K 检索 + Prompt 约束生成 + 溯源输出 |
| 5 | [based_docs_qa5.py](based_docs_qa5.py) | LoRA 微调 | 三元组指令微调 (instruction/input/output)，LoRA r=8，仅保存增量权重 |
| 6 | [based_docs_qa6.py](based_docs_qa6.py) | LangChain 全流程 | LangChain 的文档加载器 + RecursiveCharacterTextSplitter + RetrievalQA 链 |
| 生产版 | [based_docs_qa.py](based_docs_qa.py) | 完整 RAG 管线 | 语义分段（按句号切分，200 字上限）+ BGE 向量化 + Qdrant + DeepSeek-7B 生成 |

### 测试数据

| 文件 | 说明 |
|------|------|
| [books/西游记.pdf](books/西游记.pdf) | RAG 测试文档 - PDF 格式 |
| [books/水浒传.docx](books/水浒传.docx) | RAG 测试文档 - Word 格式 |
| [books/三国演义.txt](books/三国演义.txt) | RAG 测试文档 - TXT 格式 |
| [triple_data.json](triple_data.json) | RAG 问答三元组训练数据（instruction/input/output 格式，基于上述三本名著） |
| [mini_translation_pairs.txt](mini_translation_pairs.txt) | 中英平行翻译语料（奇数行中文，偶数行英文） |
| [bank_intent_data/](bank_intent_data/) | 银行意图分类数据集（train.jsonl / val.jsonl / test.jsonl，5 类意图） |

---

## 🚀 快速开始

### 环境依赖

```bash
pip install torch transformers peft bitsandbytes datasets
pip install pdfplumber python-docx chardet
pip install qdrant-client modelscope
pip install langchain
pip install jieba scikit-learn pandas
```

### 运行示例

```bash
# N-gram 文本生成
python n-gram.py

# LDA 主题模型
python lda.py

# Transformer 翻译训练
python trans_onit.py --train --epochs 3
python trans_onit.py --translate "我喜欢学习"

# Decoder-Only 训练与生成
python train_decoder_only.py --train --epochs 3
python train_decoder_only.py --prompt "我"

# Scaling Law 估算
python scaling_law.py

# RAG 文档问答（生产版，需先下载模型）
python based_docs_qa.py
```

---

## ⚠️ 大文件说明（Git 忽略）

以下文件/目录体积较大，**不建议推送到 GitHub**，已在 `.gitignore` 中排除。使用前需按说明自行准备。

| 文件/目录 | 大小 | 来源 | 用途 |
|-----------|------|------|------|
| `Qwen2.5-1.5B-Instruct/` | **2.9 GB** | [ModelScope - Qwen2.5-1.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct) | Qwen2.5 意图分类 LoRA 微调的基础模型 |
| `bge-large-zh-v1.5/` | **1.3 GB** | [ModelScope - BAAI/bge-large-zh-v1.5](https://modelscope.cn/models/BAAI/bge-large-zh-v1.5) | 中文 Embedding 模型，用于 RAG 系统的文本向量化 |
| `ckpt/` | **186 MB** | 由 `trans_onit.py --train` 训练生成 | Transformer 翻译模型权重 (`mini_transformer_ckpt.pt`) + 词表 (`meta.json`) |
| `ckpt_gpt/` | **74 MB** | 由 `train_decoder_only.py --train` 训练生成 | Decoder-Only 语言模型权重 (`mini_decoder_only_model.pth`) |
| `bank_intent_data.zip` | **9 KB** | 训练营提供 | `bank_intent_data/` 目录的压缩包（体积小，可保留） |

### .gitignore 参考配置

```gitignore
# 预训练模型（需从 ModelScope 下载）
Qwen2.5-1.5B-Instruct/
bge-large-zh-v1.5/

# 训练产出的 checkpoint
ckpt/
ckpt_gpt/

# 运行时生成的向量数据库
qdrant_db/
langchain_qdrant_db/

# LoRA 微调输出
lora_results/
bank_lora_model/

# Python 缓存
__pycache__/
*.pyc
```

### 模型下载方式

```python
# Qwen2.5-1.5B-Instruct
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='./Qwen2.5-1.5B-Instruct')

# BAAI/bge-large-zh-v1.5
from modelscope import snapshot_download
snapshot_download('BAAI/bge-large-zh-v1.5', local_dir='./bge-large-zh-v1.5')
```

---

## 📐 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG 文档问答系统架构                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │ PDF/Word │───▶│  语义分段     │───▶│ BGE Embedding│     │
│   │ /TXT 文档│    │ (按句号切分)  │    │  (768维向量)  │     │
│   └──────────┘    └──────────────┘    └──────┬───────┘     │
│                                              │              │
│                                              ▼              │
│                                       ┌──────────────┐      │
│                                       │   Qdrant     │      │
│                                       │  向量数据库   │      │
│                                       └──────┬───────┘      │
│                                              │              │
│   ┌──────────┐    ┌──────────────┐           │              │
│   │ 用户提问  │───▶│ 问题向量化    │───────────┘              │
│   └──────────┘    └──────────────┘                          │
│                           │ Top-K 检索                       │
│                           ▼                                  │
│                    ┌──────────────┐    ┌──────────────┐      │
│                    │  相似片段     │───▶│ DeepSeek-7B  │      │
│                    │  + 溯源信息   │    │  生成答案     │      │
│                    └──────────────┘    └──────┬───────┘      │
│                                              │              │
│                                              ▼              │
│                                       ┌──────────────┐      │
│                                       │ 带溯源的答案  │      │
│                                       └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📄 License

本项目仅供学习使用。预训练模型的使用请遵循各自的许可协议：
- Qwen2.5: [Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE)
- BGE: [MIT](https://huggingface.co/BAAI/bge-large-zh-v1.5)
# llm-finetune-to-production
