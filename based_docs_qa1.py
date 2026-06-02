# encoding=utf-8

import os
import chardet      # 处理TXT文件
import pdfplumber   # 处理PDF文件
from docx import Document   # 处理WORD文件（.docx格式）

def read_pdf(file_path):
    """读取PDF文件，返回【文本片段+溯源信息】列表"""
    pdf_content = []
    with pdfplumber.open(file_path) as pdf:
        # 遍历每一页，提取文本和页码
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():  # 过滤空文本
                pdf_content.append({
                    "text": text.strip(),  # 文本内容
                    "source": {
                        "file_name": os.path.basename(file_path),  # 文件名
                        "page": page_num,  # 页码（溯源关键）
                        "type": "pdf"
                    }
                })
    return pdf_content

def read_word(file_path):
    """读取Word文件（.docx），返回【文本片段+溯源信息】列表"""
    doc_content = []
    doc = Document(file_path)
    # 遍历每一段，提取文本和段落序号（Word无页码，用段落序号溯源）
    for para_num, paragraph in enumerate(doc.paragraphs, start=1):
        text = paragraph.text
        if text and text.strip():  # 过滤空文本
            doc_content.append({
                "text": text.strip(),
                "source": {
                    "file_name": os.path.basename(file_path),
                    "paragraph": para_num,  # 段落序号（溯源关键）
                    "type": "docx"
                }
            })
    return doc_content

def read_txt(file_path):
    """读取TXT文件，自动识别编码，返回【文本片段+溯源信息】列表"""
    txt_content = []
    # 自动识别TXT编码（避免macOS下中文乱码）
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding'] or 'utf-8'  # 默认utf-8

    # 读取文本，按行分割（或按段落，这里用行号溯源）
    with open(file_path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        for line_num, line in enumerate(lines, start=1):
            text = line.strip()
            if text:  # 过滤空行
                txt_content.append({
                    "text": text,
                    "source": {
                        "file_name": os.path.basename(file_path),
                        "line": line_num,  # 行号（溯源关键）
                        "type": "txt"
                    }
                })
    return txt_content

def read_all_docs(folder_path):
    """读取文件夹下所有支持格式的文档，返回合并后的结构化内容"""
    all_content = []
    # 遍历文件夹下的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()  # 获取文件后缀（小写）

            # 根据后缀调用对应读取函数
            if file_ext == '.pdf':
                all_content.extend(read_pdf(file_path))
                print(f"已读取PDF：{file}，共{len(read_pdf(file_path))}个页面片段")
            elif file_ext == '.docx':
                all_content.extend(read_word(file_path))
                print(f"已读取Word：{file}，共{len(read_word(file_path))}个段落片段")
            elif file_ext == '.txt':
                all_content.extend(read_txt(file_path))
                print(f"已读取TXT：{file}，共{len(read_txt(file_path))}个行片段")
            else:
                print(f"跳过不支持的文件格式：{file}")

    print(f"\n所有文档读取完成，共获取{len(all_content)}个文本片段")
    return all_content

# 测试：读取指定文件夹下的文档
if __name__ == "__main__":
    # 替换为你的文档文件夹路径
    DOC_FOLDER = "books"

    # 读取所有文档
    docs_content = read_all_docs(DOC_FOLDER)

    # 打印前2个片段，验证结果（含溯源信息）
    print("\n=== 前2个文本片段示例 ===")
    for i, segment in enumerate(docs_content[:2]):
        print(f"\n片段{i + 1}：")
        print(f"文本内容：{segment['text'][:100]}..."  # 只显示前100字
              if len(segment['text']) > 100 else f"文本内容：{segment['text']}")
        print(f"溯源信息：{segment['source']}")



