"""
搜索模块
"""

import jieba
from collections import Counter
import numpy as np
import faiss
from .document_processor import load_model

def extract_keywords(text, top_k=5):
    """
    从文本中提取关键词
    """
    words = jieba.cut(text)
    word_count = Counter(words)
    # 过滤掉停用词和单个字符
    keywords = [word for word, count in word_count.most_common(top_k*2) if len(word) > 1]
    return keywords[:top_k]

def search_documents(keywords, file_indices):
    """
    基于关键词搜索文档
    """
    relevant_docs = []
    for file_name, (chunks, _) in file_indices.items():
        doc_content = ' '.join(chunks)
        if any(keyword in doc_content for keyword in keywords):
            relevant_docs.append(file_name)
    return relevant_docs

def vector_search(query, file_indices, top_k=3):
    """
    执行向量检索
    """
    # 加载模型
    model = load_model()
    
    # 编码查询
    query_vector = model.encode([query])
    
    # 准备检索
    all_chunks = []
    chunk_to_file = {}
    combined_index = faiss.IndexFlatL2(384)  # 384是向量维度
    
    offset = 0
    for file_name, (chunks, index) in file_indices.items():
        all_chunks.extend(chunks)
        for i in range(len(chunks)):
            chunk_to_file[offset + i] = file_name
        if index.ntotal > 0:
            vectors = index.reconstruct_n(0, index.ntotal)
            combined_index.add(vectors.astype(np.float32))
        offset += len(chunks)
    
    # 执行检索
    if combined_index.ntotal > 0:
        D, I = combined_index.search(query_vector.astype(np.float32), k=top_k)
        results = []
        for i in I[0]:
            if 0 <= i < len(all_chunks):
                chunk = all_chunks[i]
                file_name = chunk_to_file.get(i, "未知文件")
                results.append((file_name, chunk))
        return results
    
    return [] 