# encoding=utf-8

# 导入LangChain组件
import os
import torch
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# 加载并分割文档（替换原代码的read_all_docs）
def load_and_split_docs(doc_folder):
    """用LangChain Loader加载多格式文档，并用分割器处理"""
    docs = []
    # 遍历文件夹加载所有文档
    for file in os.listdir(doc_folder):
        file_path = os.path.join(doc_folder, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())  # 自动按页分割，带页码信息
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            docs.extend(loader.load())  # 自动按段落分割，带文档名
        elif file.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            docs.extend(loader.load())  # 自动按行分割，带行号
        print(f"已加载文档：{file}，当前总片段数：{len(docs)}")

    # 分割文档（按500字符，重叠50字符）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"\n文档分割完成，最终得到 {len(split_docs)} 个文本片段")
    return split_docs

# 初始化向量存储（替换原代码的generate_embeddings+Qdrant操作）
def init_vector_store(split_docs):
    """用BAAI模型生成向量，存入Qdrant"""
    # 初始化Embedding模型（对应原代码的load_embedding_model）
    embeddings = HuggingFaceEmbeddings(
        model_name="baai/bge-large-zh-v1.5",
        model_kwargs={"device": "cuda"},  # 用GPU加速向量化
        encode_kwargs={"normalize_embeddings": True}  # 自动归一化
    )

    # 初始化Qdrant向量库（对应原代码的init_qdrant+insert_vectors）
    qdrant_path = os.path.join(os.getcwd(), "langchain_qdrant_db")
    vector_store = Qdrant.from_documents(
        documents=split_docs,
        embedding=embeddings,
        path=qdrant_path,
        collection_name="docs_embeddings"
    )
    print(f"\nQdrant向量库初始化完成，存储路径：{qdrant_path}")
    return vector_store

# 初始化QA链并运行（替换原代码的load_local_llm+generate_answer）
def init_qa_chain(vector_store):
    """加载DeepSeek模型，初始化RetrievalQA链"""
    # 初始化4位量化配置（和原代码一致）
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 加载LLM并封装成LangChain的Pipeline（原代码的load_local_llm）
    model_path = "/root/models/deepseek-llm-7b-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 封装成HuggingFace Pipeline（适配LangChain）
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    # 初始化QA链（自动串联检索和生成）
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 简单高效的“填充式”链（适合短片段）
        retriever=vector_store.as_retriever(top_k=3),  # 检索Top3相似片段
        return_source_documents=True  # 返回源文档（用于溯源）
    )
    print("\nQA链初始化完成，可开始提问！")
    return qa_chain

# 主函数
def langchain_qa_pipeline():
    DOC_FOLDER = "/root/docs"
    # 加载分割文档
    split_docs = load_and_split_docs(DOC_FOLDER)
    # 初始化向量存储
    vector_store = init_vector_store(split_docs)
    # 初始化QA链
    qa_chain = init_qa_chain(vector_store)
    # 测试问答
    test_queries = [
        "桃园三结义的三个人是谁？",
        "鲁智深为什么要倒拔垂杨柳？",
        "孙悟空在五行山下被压了多少年？"
    ]
    for query in test_queries:
        print(f"\n【用户问题】：{query}")
        result = qa_chain({"query": query})
        # 输出答案
        print(f"【回答】：\n{result['result']}")
        # 输出源文档信息（溯源）
        print("【信息来源】：")
        for i, doc in enumerate(result["source_documents"], 1):
            print(
                f"  {i}. {doc.metadata['source']}（页码/行号：{doc.metadata.get('page', doc.metadata.get('line', ''))}）")
        print("-" * 50)

if __name__ == "__main__":
    torch.set_num_threads(1)
    langchain_qa_pipeline()



