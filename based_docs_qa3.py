# encoding=utf-8

import os
import chardet
import pdfplumber
from docx import Document
from modelscope import AutoModel, AutoTokenizer
import torch
# 新增：Qdrant客户端依赖（本地模式，无需服务器）
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


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
第三阶段：Qdrant向量存储核心函数
"""

def init_qdrant():
    """
    初始化Qdrant客户端（本地文件模式）
    注意：
    1. 无需安装Qdrant服务器，数据存在当前目录的"qdrant_db"文件夹；
    2. 若后续想重新存储，删除"qdrant_db"文件夹后重新运行即可。
    """
    # 向量数据存储路径（当前代码所在目录下）
    qdrant_db_path = os.path.join(os.getcwd(), "qdrant_db")
    # 初始化本地客户端
    client = QdrantClient(path=qdrant_db_path)
    print(f"\nQdrant初始化完成！数据将存储在：{qdrant_db_path}")
    return client

def create_qdrant_collection(client, collection_name="student_docs_collection"):
    """
    创建Qdrant集合（类似数据库的“表”）
    注意：
    1. 集合名可自定义（比如改"novels_collection"）；
    2. size=768必须与BAAI模型的向量维度一致，否则会报错！
    """
    # 先检查集合是否已存在，存在则删除（方便重复测试）
    if client.collection_exists(collection_name):
        print(f"集合{collection_name}已存在，先删除旧集合...")
        client.delete_collection(collection_name)

    # 创建新集合：指定向量维度（768）和检索距离（余弦距离，适合文本向量）
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=768,  # 关键：与BAAI/bge-large-zh-v1.5的向量维度一致
            distance=Distance.COSINE  # 关键：文本向量常用余弦距离计算相似度
        )
    )
    print(f"成功创建Qdrant集合：{collection_name}（向量维度768，余弦距离检索）")

def insert_vectors_to_qdrant(client, embeddings_with_info, collection_name="student_docs_collection"):
    """
    将“文本+向量+溯源”数据插入Qdrant
    注意：
    1. 每个向量会绑定元数据（text和source），后续检索时能直接拿到这些信息；
    2. id用索引（0,1,2...）即可，保证唯一就行。
    """
    # 转换为Qdrant要求的格式（PointStruct：id+向量+元数据）
    qdrant_points = []
    for idx, item in enumerate(embeddings_with_info):
        qdrant_points.append(PointStruct(
            id=idx,  # 唯一ID（用索引最简单，不会重复）
            vector=item["vector"],  # 第二阶段生成的向量
            payload={  # 元数据：后续检索需要的文本和溯源信息
                "text_content": item["text"],  # 文本内容
                "source_info": item["source"]  # 溯源信息（文件名、页码等）
            }
        ))

    # 批量插入Qdrant
    client.upsert(
        collection_name=collection_name,
        points=qdrant_points
    )
    print(f"成功向Qdrant插入{len(qdrant_points)}个向量片段！")

if __name__ == "__main__":
    # 配置：替换为自己的文档文件夹路径（和前两阶段一致）
    DOC_FOLDER = "/Users/zhaoshuai/docs"

    # 复用第一阶段：读取文档，得到文本片段
    text_segments = read_all_docs(DOC_FOLDER)
    if not text_segments:
        print("未读取到有效文档，终止流程")
        exit()

    # 复用第二阶段：生成向量，得到“文本+向量+溯源”数据
    tokenizer, embedding_model = load_embedding_model()
    embeddings_data = generate_embeddings(tokenizer, embedding_model, text_segments)

    # 第三阶段：Qdrant存储
    # 初始化Qdrant客户端
    qdrant_client = init_qdrant()
    # 创建集合（确保向量维度匹配）
    create_qdrant_collection(qdrant_client)
    # 插入向量数据
    insert_vectors_to_qdrant(qdrant_client, embeddings_data)

    # 验证：提示学生后续操作
    print("\n✅ 第三阶段全部完成！")
    print("后续学习方向：用Qdrant检索相似向量（第四阶段：问答功能）")
    print("当前可查看：代码所在目录的'qdrant_db'文件夹，里面是存储的向量数据")



