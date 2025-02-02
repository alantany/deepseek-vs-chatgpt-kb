"""
文档处理和向量化模块
"""

import PyPDF2
import docx
import tiktoken
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def load_model():
    """
    加载和缓存sentence transformer模型
    """
    return SentenceTransformer('all-MiniLM-L6-v2')

def num_tokens_from_string(string: str) -> int:
    """
    计算文本的token数量
    """
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(encoding.encode(string))

def extract_text_from_file(file) -> str:
    """
    从不同格式的文件中提取文本
    """
    text = ""
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:  # 假设是txt文件
        text = file.getvalue().decode("utf-8")
    return text

def chunk_text(text: str, max_tokens: int) -> list:
    """
    将文本分割成适当大小的块
    """
    chunks = []
    current_chunk = ""
    for sentence in text.split('.'):
        if num_tokens_from_string(current_chunk + sentence) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += sentence + '.'
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def vectorize_document(file, max_tokens):
    """
    处理文档并创建向量索引
    """
    # 提取文本
    text = extract_text_from_file(file)
    
    # 分块
    chunks = chunk_text(text, max_tokens)
    
    # 向量化
    model = load_model()
    vectors = model.encode(chunks)
    
    # 创建索引
    index = faiss.IndexFlatL2(384)  # 384是向量维度
    index.add(vectors)
    
    return chunks, index 