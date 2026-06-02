# encoding=utf-8

import os
import chardet
import pdfplumber
from docx import Document
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import AutoModel, AutoTokenizer as BgeTokenizer

BGE_LOCAL_PATH = "/root/models/bge-large-zh-v1.5"
VECTOR_DIM = 1024
DOC_FOLDER = "/root/docs"
COLLECTION_NAME = "docs_embeddings"
MAX_CHUNK_LENGTH = 200

# 语义分段
def split_text_by_semantic(text):
    sentences = [s.strip() + "。" for s in text.split("。") if s.strip()]
    semantic_chunks = []
    current_chunk = ""
    for sent in sentences:
        if len(current_chunk) + len(sent) > MAX_CHUNK_LENGTH:
            if current_chunk:
                semantic_chunks.append(current_chunk)
            current_chunk = sent
        else:
            current_chunk += sent
    if current_chunk:
        semantic_chunks.append(current_chunk)
    return semantic_chunks

# 文档解析
def read_pdf(file_path):
    pdf_content = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                chunks = split_text_by_semantic(text.strip())
                for chunk_idx, chunk in enumerate(chunks, start=1):
                    pdf_content.append({
                        "text": chunk,
                        "source": {
                            "file_name": os.path.basename(file_path),
                            "page": page_num,
                            "chunk": chunk_idx,
                            "type": "pdf"
                        }
                    })
    return pdf_content

def read_word(file_path):
    doc_content = []
    doc = Document(file_path)
    for para_num, paragraph in enumerate(doc.paragraphs, start=1):
        text = paragraph.text
        if text and text.strip():
            chunks = split_text_by_semantic(text.strip())
            for chunk_idx, chunk in enumerate(chunks, start=1):
                doc_content.append({
                    "text": chunk,
                    "source": {
                        "file_name": os.path.basename(file_path),
                        "paragraph": para_num,
                        "chunk": chunk_idx,
                        "type": "docx"
                    }
                })
    return doc_content

def read_txt(file_path):
    txt_content = []
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding'] or 'utf-8'
    with open(file_path, 'r', encoding=encoding) as f:
        full_text = f.read()
        chunks = split_text_by_semantic(full_text.strip())
        for chunk_idx, chunk in enumerate(chunks, start=1):
            txt_content.append({
                "text": chunk,
                "source": {
                    "file_name": os.path.basename(file_path),
                    "chunk": chunk_idx,
                    "type": "txt"
                }
            })
    return txt_content

def read_all_docs(folder_path):
    all_content = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext == '.pdf':
                pdf_segments = read_pdf(file_path)
                all_content.extend(pdf_segments)
                print(f"已读取PDF：{file}，共{len(pdf_segments)}个语义片段")
            elif file_ext == '.docx':
                word_segments = read_word(file_path)
                all_content.extend(word_segments)
                print(f"已读取Word：{file}，共{len(word_segments)}个语义片段")
            elif file_ext == '.txt':
                txt_segments = read_txt(file_path)
                all_content.extend(txt_segments)
                print(f"已读取TXT：{file}，共{len(txt_segments)}个语义片段")
            elif file != '.DS_Store':
                print(f"跳过不支持的文件格式：{file}")
    print(f"\n所有文档读取完成，共获取{len(all_content)}个语义片段")
    return all_content

# BGE模型加载与向量生成
def load_bge_model():
    print(f"\n【加载本地BGE模型】路径：{BGE_LOCAL_PATH}...")
    try:
        tokenizer = BgeTokenizer.from_pretrained(BGE_LOCAL_PATH, local_files_only=True)
        model = AutoModel.from_pretrained(BGE_LOCAL_PATH, local_files_only=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"BGE模型加载完成！运行设备：{device}")
        return tokenizer, model, device
    except Exception as e:
        raise Exception(f"本地BGE加载失败：{str(e)}")

def get_bge_embedding(texts, tokenizer, model, device):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    embeddings = embeddings / (embeddings ** 2).sum(axis=1, keepdims=True) ** 0.5
    return embeddings.tolist()

def generate_embeddings(text_segments, tokenizer, model, device):
    texts = [seg["text"] for seg in text_segments]
    print(f"\n正在生成{len(texts)}个语义片段的BGE向量...")
    embeddings = get_bge_embedding(texts, tokenizer, model, device)
    embeddings_with_info = []
    for seg, emb in zip(text_segments, embeddings):
        embeddings_with_info.append({
            "text": seg["text"],
            "vector": emb,
            "source": seg["source"]
        })
        if len(embeddings_with_info) % 10 == 0:
            print(f"已处理 {len(embeddings_with_info)}/{len(text_segments)} 个片段")
    print(f"\n向量生成完成！共{len(embeddings_with_info)}个带向量的片段")
    return embeddings_with_info

# Qdrant向量存储
def init_qdrant():
    qdrant_db_path = os.path.join(os.getcwd(), "qdrant_db")
    client = QdrantClient(path=qdrant_db_path)
    print(f"\nQdrant初始化完成！数据存储路径：{qdrant_db_path}")
    return client

def create_qdrant_collection(client):
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
    )
    print(f"已创建Qdrant集合：{COLLECTION_NAME}（向量维度：{VECTOR_DIM}）")

def insert_vectors_to_qdrant(client, embeddings_with_info):
    points = []
    for idx, item in enumerate(embeddings_with_info):
        if len(item["vector"]) != VECTOR_DIM:
            raise ValueError(f"第{idx}个向量维度错误，应为{VECTOR_DIM}维（实际{len(item['vector'])}维）")
        points.append(PointStruct(
            id=idx,
            vector=item["vector"],
            payload={"text_content": item["text"], "source_info": item["source"]}
        ))
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"已向Qdrant插入{len(points)}个BGE向量片段")

# 本地LLM加载
def load_local_llm():
    model_path = "/root/models/deepseek-llm-7b-base"
    # model_path = "/usr/bin/models/deepseek-moe-16b-chat""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    print(f"\n正在加载本地LLM：{model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        print(f"LLM加载完成，运行设备：{model.device}")
        return tokenizer, model
    except Exception as e:
        raise Exception(f"LLM加载失败：{str(e)}")

# 检索与问答
def retrieve_similar_segments(client, query, tokenizer, model, device):
    query_embedding = get_bge_embedding([query], tokenizer, model, device)[0]
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=2,
        with_payload=True
    )

    print(f"\n【调试】问题'{query}'的检索结果：")
    for i, hit in enumerate(search_result.points, 1):
        print(f"第{i}个片段：相似度{hit.score:.3f} | 文本：{hit.payload['text_content']}")

    similar_segments = [
        {
            "text": hit.payload["text_content"],
            "source": hit.payload["source_info"],
            "similarity": round(hit.score, 3)
        }
        for hit in search_result.points if hit.score >= 0.45
    ]

    return similar_segments if similar_segments else "未找到与问题相关的文档片段"

def generate_answer_with_source(query, similar_segments, llm_tokenizer, llm_model):
    top_segment = similar_segments[0]
    key_text = top_segment["text"]

    # 生成正确的来源字符串（区分文件类型）
    if top_segment["source"]["type"] == "pdf":
        source_str = f"《{top_segment['source']['file_name']}》PDF第{top_segment['source']['page']}页（片段{top_segment['source']['chunk']}）"
    elif top_segment["source"]["type"] == "docx":
        source_str = f"《{top_segment['source']['file_name']}》Word第{top_segment['source']['paragraph']}段（片段{top_segment['source']['chunk']}）"
    else:  # TXT
        source_str = f"《{top_segment['source']['file_name']}》TXT（片段{top_segment['source']['chunk']}）"

    # 极简Prompt：无任何格式符号，直接要求输出
    prompt = f"""
    问题：{query}
    片段：{key_text}
    输出要求：
    1. 第一行写“答案：”，后面跟从片段中提取的完整答案（至少10个字）；
    2. 第二行写“信息来源：”，后面跟“{source_str}”（直接复制，不要改）。
    """

    # 生成配置：确保答案完整
    inputs = llm_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False,
        padding=True
    ).to(llm_model.device)

    # prompt = [
    #     {"role": "user", "content": f"""
    #     请根据以下片段回答我的问题，严格按格式输出：
    #     问题：{query}
    #     片段：{key_text}
    #     输出格式：
    #     1. 第一行写“答案：”，后面跟从片段中提取的完整答案（至少10个字）；
    #     2. 第二行写“信息来源：”，后面跟“{source_str}”（直接复制，不要改）。
    #     """}
    # ]
    # # 用Chat模型的tokenizer编码对话格式
    # inputs = llm_tokenizer.apply_chat_template(
    #     prompt,
    #     return_tensors="pt",
    #     truncation=False,
    #     padding=True,
    #     add_generation_prompt=True  # 自动添加“助手”角色前缀
    # ).to(llm_model.device)

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=200,  # 足够长的输出长度
            temperature=0.4,
            do_sample=True,
            eos_token_id=llm_tokenizer.eos_token_id,
            pad_token_id=llm_tokenizer.pad_token_id,
            no_repeat_ngram_size=2  # 避免重复输出
        )

    # 解析并清理输出
    answer = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_lines = [line.strip() for line in answer.split("\n") if line.strip()]

    # 提取答案和来源（无手动兜底，纯LLM生成）
    answer_content = "答案：未提取到相关答案"  # 默认值，避免空输出
    for line in answer_lines:
        if line.startswith("答案：") and len(line) > 5:  # 过滤空答案
            answer_content = line
            break
    final_source = f"信息来源：{source_str}"

    # 【关键修复】添加返回值，确保函数有输出
    return f"{answer_content}\n{final_source}"

# 全流程入口
def full_pipeline():
    print("===== 第一阶段：文档解析（语义分段） =====")
    text_segments = read_all_docs(DOC_FOLDER)
    if not text_segments:
        print("未读取到有效文档，终止流程")
        return

    print("\n===== 第二阶段：加载本地BGE模型 =====")
    bge_tokenizer, bge_model, device = load_bge_model()

    print("\n===== 第三阶段：生成文本BGE向量 =====")
    embeddings_data = generate_embeddings(text_segments, bge_tokenizer, bge_model, device)

    print("\n===== 第四阶段：向量存储到Qdrant =====")
    qdrant_client = init_qdrant()
    create_qdrant_collection(qdrant_client)
    insert_vectors_to_qdrant(qdrant_client, embeddings_data)

    print("\n===== 第五阶段：检索问答 =====")
    llm_tokenizer, llm_model = load_local_llm()
    test_queries = [
        "刘备、关羽、张飞在桃园结义时立下了什么誓言？",
        "鲁智深在相国寺看管菜园时，如何震慑附近的泼皮无赖？",
        "孙悟空因什么事被如来佛祖压在五行山下？"
    ]

    for query in test_queries:
        print(f"\n【用户问题】：{query}")
        similar_segs = retrieve_similar_segments(qdrant_client, query, bge_tokenizer, bge_model, device)
        if isinstance(similar_segs, str):
            print(f"【回答】：{similar_segs}")
            continue
        answer = generate_answer_with_source(query, similar_segs, llm_tokenizer, llm_model)
        print(f"【回答】：\n{answer}")
        print("-" * 60)

if __name__ == "__main__":
    try:
        full_pipeline()
        print("\n===== 全流程运行完成！=====")
    except Exception as e:
        print(f"\n运行错误：{str(e)}")



