# encoding=utf-8

import os
import chardet
import pdfplumber
from docx import Document
from modelscope import AutoModel, AutoTokenizer
import torch
from qdrant_client import QdrantClient

from transformers import AutoModelForCausalLM, BitsAndBytesConfig

"""
复用第一阶段读取文档读取函数
"""
def read_pdf(file_path):
    pdf_content = []
    return pdf_content

def read_word(file_path):
    doc_content = []
    return doc_content

def read_txt(file_path):
    txt_content = []
    return txt_content

def read_all_docs(folder_path):
    all_content = []
    return all_content

"""
复用第二阶段向量生成函数
"""
def load_embedding_model():
    model_name = "baai/bge-large-zh-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def generate_embeddings(tokenizer, model, text_segments):
    embeddings_with_info = []
    return embeddings_with_info

"""
复用第三阶段向量数据库初始化
"""
def init_qdrant(collection_name="student_docs_collection"):
    client = QdrantClient(path=os.path.join(os.getcwd(), "qdrant_db"))
    return client, collection_name

"""
第四阶段：检索+LLM问答+溯源
"""
def load_local_llm():
    """
    加载本地DeepSeek-7B-base模型（需提前下载）
    注意：
    1. 模型约13.8GB，需先从ModelScope下载：https://modelscope.cn/models/deepseek-ai/deepseek-llm-7b-base
    2. 下载后将模型文件夹路径替换下方"model_path"
    """
    # 关键：替换为你的本地模型文件夹路径（示例：/Users/zhaoshuai/Downloads/deepseek-llm-7b-base）
    model_path = "/Users/zhaoshuai/models/deepseek-llm-7b-base"

    # 4位量化配置（降低占用，避免卡顿）
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print(f"\n正在加载本地LLM：{model_path}（CPU模式，首次加载约30秒）")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    return tokenizer, model

def retrieve_similar_segments(client, collection_name, query_vector, top_k=3):
    """
    从Qdrant检索与问题最相似的Top-K片段
    top_k=3：默认返回3个最相关片段（可调整，越多回答越全面）
    """
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,  # 检索数量
        with_payload=True  # 返回片段的文本和溯源信息
    )

    # 整理检索结果（过滤相似度低于0.3的不相关片段）
    similar_segments = []
    for hit in search_result:
        if hit.score >= 0.3:  # 相似度阈值，避免无关信息干扰
            similar_segments.append({
                "text": hit.payload["text_content"],
                "source": hit.payload["source_info"],
                "similarity": round(hit.score, 3)  # 相似度分数（0-1，越高越相关）
            })

    if not similar_segments:
        return "未找到相关文档片段，请尝试调整问题表述"
    return similar_segments

def generate_answer_with_source(query, similar_segments, llm_tokenizer, llm_model):
    """
    结合“问题+相似片段”生成带溯源的答案
    核心：用提示词引导LLM只基于检索到的片段回答，避免编造信息
    """
    # 构建提示词（关键：明确LLM的回答规则）
    prompt = f"""
    请基于以下【相关文档片段】回答用户问题，要求：
    1. 只使用片段中的信息，不添加外部知识；
    2. 回答后必须标注“信息来源”（格式：文件名+位置，如《水浒传》docx第5段）；
    3. 若多个片段有关联，可整合信息；若片段无相关信息，直接说“未找到对应信息”。

    【用户问题】：{query}

    【相关文档片段】：
    """
    for i, seg in enumerate(similar_segments, 1):
        # 格式化片段信息（区分不同文件类型的位置）
        if seg["source"]["type"] == "pdf":
            source_str = f"《{seg['source']['file_name']}》PDF第{seg['source']['page']}页"
        elif seg["source"]["type"] == "docx":
            source_str = f"《{seg['source']['file_name']}》Word第{seg['source']['paragraph']}段"
        else:  # txt
            source_str = f"《{seg['source']['file_name']}》TXT第{seg['source']['line']}行"

        prompt += f"{i}. 文本：{seg['text'][:150]}...（相似度：{seg['similarity']}）\n   来源：{source_str}\n"

    # 调用LLM生成答案（CPU运行，约10-20秒/次）
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=512,  # 答案最大长度
            temperature=0.7,  # 随机性（0-1，越低越严谨）
            top_p=0.9,
            do_sample=True,
            eos_token_id=llm_tokenizer.eos_token_id
        )

    # 解码答案（去除提示词部分）
    answer = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.replace(prompt, "").strip()

# 完整问答流程
def docs_qa_pipeline(query):
    """
    文档问答流水线：问题→向量检索→LLM生成答案
    输入：用户问题（字符串）
    输出：带溯源的答案
    """
    try:
        # 初始化依赖（Embedding模型+Qdrant+LLM）
        print("1. 初始化依赖...")
        emb_tokenizer, emb_model = load_embedding_model()
        qdrant_client, collection_name = init_qdrant()
        llm_tokenizer, llm_model = load_local_llm()

        # 问题向量化（与文档向量用同一模型，保证语义一致）
        print("2. 问题向量化...")
        query_vector = text_to_vector(query, emb_tokenizer, emb_model)

        # Qdrant检索相似片段
        print("3. 检索相关文档片段...")
        similar_segs = retrieve_similar_segments(qdrant_client, collection_name, query_vector)
        if isinstance(similar_segs, str):  # 未找到相关片段
            return f"❌ {similar_segs}"

        # LLM生成带溯源的答案
        print("4. 生成带溯源的答案...")
        answer = generate_answer_with_source(query, similar_segs, llm_tokenizer, llm_model)

        # 格式化输出结果
        result = f"""
=====================================
用户问题：{query}

回答：
{answer}

检索到的相关片段（供验证）：
"""
        for i, seg in enumerate(similar_segs, 1):
            result += f"{i}. 相似度：{seg['similarity']} | 来源：《{seg['source']['file_name']}》{seg['source']['type'].upper()}第{seg['source'].get('page') or seg['source'].get('paragraph') or seg['source'].get('line')}位置\n"

        return result + "====================================="

    except Exception as e:
        return f"问答过程出错：{str(e)}"

# 测试入口
if __name__ == "__main__":
    # 示例问题（基于《西游记》《水浒传》《三国演义》，可替换为自己想查的问题）
    test_queries = [
        "桃园三结义的三个人是谁？",
        "鲁智深为什么要倒拔垂杨柳？",
        "孙悟空在五行山下被压了多少年？"
    ]

    # 逐个测试问题
    for query in test_queries:
        print(docs_qa_pipeline(query))
        print("\n" + "-" * 50 + "\n")  # 分隔不同问题的结果



