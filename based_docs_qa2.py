# encoding=utf-8

import os
import chardet
import pdfplumber
from docx import Document
from modelscope import AutoModel, AutoTokenizer
import torch


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

# 第二阶段
def load_embedding_model():
    """从ModelScope加载BAAI/bge-large-zh-v1.5"""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    model_name = "baai/bge-large-zh-v1.5"
    print(f"\n正在从ModelScope加载Embedding模型：{model_name}")
    print("提示：首次运行会自动下载模型（约1.21GB，国内网络支持）")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to("cpu")
    return tokenizer, model

def generate_embeddings(tokenizer, model, text_segments):
    """生成向量（逻辑不变）"""
    embeddings_with_info = []
    texts = [segment["text"] for segment in text_segments]
    print(f"\n正在生成{len(texts)}个文本片段的向量...")

    for i, text in enumerate(texts):
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)

        vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        vector = vector / (vector ** 2).sum() ** 0.5

        embeddings_with_info.append({
            "text": text_segments[i]["text"],
            "vector": vector.tolist(),
            "source": text_segments[i]["source"]
        })

        if (i + 1) % 5 == 0:
            print(f"已处理 {i + 1}/{len(texts)} 个片段")

    print(f"\n向量生成完成！共{len(embeddings_with_info)}个带向量的片段")
    return embeddings_with_info

if __name__ == "__main__":
    DOC_FOLDER = "/Users/zhaoshuai/docs"
    text_segments = read_all_docs(DOC_FOLDER)

    if text_segments:
        tokenizer, model = load_embedding_model()
        embeddings_with_info = generate_embeddings(tokenizer, model, text_segments)

        print("\n=== 前2个带向量的片段示例 ===")
        for i, item in enumerate(embeddings_with_info[:2]):
            print(f"\n片段{i + 1}：")
            text_display = item["text"][:80] + "..." if len(item["text"]) > 80 else item["text"]
            print(f"文本：{text_display}")
            vector_display = [round(x, 4) for x in item["vector"][:5]]
            print(f"向量（前5维）：{vector_display}")
            print(f"溯源：{item['source']}")
    else:
        print("\n未读取到有效文档，无法生成向量")



