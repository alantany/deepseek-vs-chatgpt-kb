"""
AI知识问答系统

使用方法：
1. 上传文档（支持PDF、DOCX、TXT格式）
2. 输入问题
3. 获取AI回答
"""

import streamlit as st
import sys
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import faiss
import tiktoken
import pickle
import numpy as np
from collections import Counter
import pandas as pd
import jieba
from dotenv import load_dotenv
import requests
import re

# 加载环境变量
load_dotenv()

def get_config(key: str, default: str = None) -> str:
    """
    获取配置值，优先从Streamlit secrets获取，如果不存在则从环境变量获取
    """
    try:
        return st.secrets[key]
    except KeyError:
        return os.getenv(key, default)

# 替换环境变量检查部分
required_configs = {
    'OPENAI_API_KEY': '用于ChatGPT的API密钥',
    'OPENAI_API_BASE': 'ChatGPT的API基础URL',
    'DEEPSEEK_API_KEY': '用于Deepseek的API密钥',
    'DEEPSEEK_API_BASE': 'Deepseek的API基础URL'
}

missing_configs = []
for config, description in required_configs.items():
    if not get_config(config):
        missing_configs.append(f"{config} ({description})")

if missing_configs:
    st.error("""
    ### 缺少必要的配置
    请在以下位置之一配置这些值：
    1. Streamlit Cloud部署：在项目设置中添加 secrets.toml
    2. 本地开发：在项目根目录创建 .env 文件
    
    缺少的配置：
    """ + "\n".join(missing_configs))
    st.stop()

# 设置页面配置
st.set_page_config(
    page_title="AI知识问答系统 - by Huaiyuan Tan",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# 添加开发者信息
st.markdown("<h6 style='text-align: right; color: gray;'>开发者: Huaiyuan Tan</h6>", unsafe_allow_html=True)

# 隐藏 Streamlit 默认的菜单、页脚和 Deploy 按钮
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 初始化OpenAI客户端
chatgpt_client = OpenAI(
    api_key=get_config("OPENAI_API_KEY"),
    base_url=get_config("OPENAI_API_BASE"),
)

# 对Deepseek的URL进行处理，确保不包含chat/completions
deepseek_base_url = get_config("DEEPSEEK_API_BASE")
if deepseek_base_url and deepseek_base_url.endswith("/chat/completions"):
    deepseek_base_url = deepseek_base_url.replace("/chat/completions", "")

# 初始化Deepseek客户端
deepseek_client = OpenAI(
    api_key=get_config("DEEPSEEK_API_KEY"),
    base_url=deepseek_base_url,
    default_headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_config('DEEPSEEK_API_KEY')}"
    }
)

def test_api_connection(client, model_name):
    """测试API连接状态"""
    try:
        st.write(f"正在测试 {model_name} API连接...")
        
        if model_name == "Deepseek":
            # 使用配置值而不是硬编码
            API_URL = f"{get_config('DEEPSEEK_API_BASE')}/chat/completions"
            API_KEY = get_config("DEEPSEEK_API_KEY")
            MODEL = "deepseek-ai/DeepSeek-R1"
            
            st.write(f"使用的API基础URL: {API_URL}")
            st.write(f"使用的模型: {MODEL}")
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            }
            data = {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
            response = requests.post(
                API_URL,
                headers=headers,
                json=data
            )
            if response.status_code != 200:
                raise Exception(f"API返回错误: {response.text}")
            response_data = response.json()
            model_info = f"{model_name} ({response_data['model']})" if 'model' in response_data else model_name
            return True, "连接正常", model_info
        else:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            model_info = f"{model_name} ({response.model})" if hasattr(response, 'model') else model_name
            return True, "连接正常", model_info
    except Exception as e:
        st.error(f"{model_name} 连接测试错误: {str(e)}")
        if model_name == "Deepseek":
            st.error(f"Deepseek配置信息：\nURL: {API_URL}\n模型: {MODEL}")
        return False, f"连接错误: {str(e)}", model_name

# 测试API连接状态
if 'api_status' not in st.session_state:
    st.session_state.api_status = {}

# 在页面顶部显示模型状态
st.markdown("## 🤖 模型状态")

# 创建两列布局显示模型状态
status_col1, status_col2 = st.columns(2)

# 测试并显示ChatGPT状态
with status_col1:
    st.markdown("### ChatGPT")
    chatgpt_ok, chatgpt_msg, chatgpt_model = test_api_connection(chatgpt_client, "ChatGPT")
    st.session_state.api_status['chatgpt'] = {
        'ok': chatgpt_ok,
        'message': chatgpt_msg,
        'model': chatgpt_model
    }
    if chatgpt_ok:
        st.success(f"✅ {chatgpt_model}\n\n状态：{chatgpt_msg}")
    else:
        st.error(f"❌ {chatgpt_model}\n\n状态：{chatgpt_msg}")

# 测试并显示Deepseek状态
with status_col2:
    st.markdown("### Deepseek")
    deepseek_ok, deepseek_msg, deepseek_model = test_api_connection(deepseek_client, "Deepseek")
    st.session_state.api_status['deepseek'] = {
        'ok': deepseek_ok,
        'message': deepseek_msg,
        'model': deepseek_model
    }
    if deepseek_ok:
        st.success(f"✅ {deepseek_model}\n\n状态：{deepseek_msg}")
    else:
        st.error(f"❌ {deepseek_model}\n\n状态：{deepseek_msg}")

st.markdown("---")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# 计算token数量
def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(encoding.encode(string))

# 文档向量化模块
def vectorize_document(file, max_tokens):
    text = ""
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        text = file.getvalue().decode("utf-8")
    
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
    
    vectors = model.encode(chunks)
    index = faiss.IndexFlatL2(384)  # 384是向量维度,根据实际模型调整
    index.add(vectors)
    return chunks, index

# 提取关键词
def extract_keywords(text, top_k=5):
    words = jieba.cut(text)
    word_count = Counter(words)
    # 过滤掉停用词和单个字符
    keywords = [word for word, count in word_count.most_common(top_k*2) if len(word) > 1]
    return keywords[:top_k]

# 基于关键词搜索文档
def search_documents(keywords, file_indices):
    relevant_docs = []
    for file_name, (chunks, _) in file_indices.items():
        doc_content = ' '.join(chunks)
        if any(keyword in doc_content for keyword in keywords):
            relevant_docs.append(file_name)
    return relevant_docs

# 知识问答模块
def rag_qa(query, file_indices, relevant_docs=None):
    try:
        st.write("开始处理查询...")
        keywords = extract_keywords(query)
        if relevant_docs is None:
            relevant_docs = search_documents(keywords, file_indices)
        
        st.write(f"找到相关文档数量: {len(relevant_docs)}")
        
        if not relevant_docs:
            return {
                'chatgpt': "没有找到相关文档。请尝试使用不同的关键词。",
                'deepseek': "没有找到相关文档。请尝试使用不同的关键词。"
            }, [], ""

        all_chunks = []
        chunk_to_file = {}
        combined_index = faiss.IndexFlatL2(384)
        
        offset = 0
        for file_name in relevant_docs:
            if file_name in file_indices:
                chunks, index = file_indices[file_name]
                all_chunks.extend(chunks)
                for i in range(len(chunks)):
                    chunk_to_file[offset + i] = file_name
                if index.ntotal > 0:
                    vectors = index.reconstruct_n(0, index.ntotal)
                    combined_index.add(vectors.astype(np.float32))
                offset += len(chunks)

        if not all_chunks:
            return {
                'chatgpt': "没有找到相关信息。请确保已上传文档。",
                'deepseek': "没有找到相关信息。请确保已上传文档。"
            }, [], ""

        query_vector = model.encode([query])
        D, I = combined_index.search(query_vector.astype(np.float32), k=3)
        context = []
        context_with_sources = []
        for i in I[0]:
            if 0 <= i < len(all_chunks):
                chunk = all_chunks[i]
                context.append(chunk)
                file_name = chunk_to_file.get(i, "未知文件")
                context_with_sources.append((file_name, chunk))

        context_text = "\n".join(context)
        
        # 确保总token数不超过4096
        max_context_tokens = 3000
        original_length = len(context_text)
        while num_tokens_from_string(context_text) > max_context_tokens:
            context_text = context_text[:int(len(context_text)*0.9)]
        if len(context_text) < original_length:
            st.write(f"截断上下文从 {original_length} 到 {len(context_text)} 字符")
        
        if not context_text:
            return {
                'chatgpt': "没有找到相关信息。",
                'deepseek': "没有找到相关信息。"
            }, [], ""

        # 创建两列布局
        left_col, right_col = st.columns(2)
        
        # 创建占位符
        with left_col:
            st.markdown("### ChatGPT回答")
            chatgpt_placeholder = st.empty()
            chatgpt_placeholder.markdown("""
            <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px;'>
                正在处理ChatGPT回答...
            </div>
            """, unsafe_allow_html=True)
        
        with right_col:
            st.markdown("### Deepseek回答")
            deepseek_placeholder = st.empty()
            deepseek_placeholder.markdown("""
            <div style='background-color: #e6f3ff; padding: 15px; border-radius: 5px;'>
                等待处理Deepseek回答...
            </div>
            """, unsafe_allow_html=True)

        responses = {'chatgpt': "", 'deepseek': ""}
        excerpts = {'chatgpt': "", 'deepseek': ""}

        # 先处理ChatGPT回答
        st.write("正在调用ChatGPT API...")
        if st.session_state.api_status['chatgpt']['ok']:
            try:
                chatgpt_response = chatgpt_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "你是一位有帮助的助手。请根据给定的上下文回答问题。始终使用中文回答，无论问题是什么语言。在回答之后，请务必提供一段最相关的原文摘录，以'相关原文：'为前缀。"},
                        {"role": "user", "content": f"上下文: {context_text}\n\n问题: {query}\n\n请提供你的回答然后在回答后面附上相关的原文摘录，以'相关原文：'为前缀。"}
                    ]
                )
                chatgpt_answer = chatgpt_response.choices[0].message.content
                st.write("ChatGPT API调用成功")
                
                if "相关原文：" in chatgpt_answer:
                    chatgpt_parts = chatgpt_answer.split("相关原文：", 1)
                    responses['chatgpt'] = chatgpt_parts[0].strip()
                    excerpts['chatgpt'] = chatgpt_parts[1].strip()
                else:
                    responses['chatgpt'] = chatgpt_answer.strip()
                
                # 立即更新ChatGPT回答
                chatgpt_placeholder.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px;'>
                    {responses['chatgpt']}
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"ChatGPT API调用出错: {str(e)}")
                responses['chatgpt'] = "API调用出错，请稍后重试"
                chatgpt_placeholder.error("获取ChatGPT回答失败")
        else:
            responses['chatgpt'] = "ChatGPT API 未连接"
            chatgpt_placeholder.error("ChatGPT API 未连接")

        # 再处理Deepseek回答
        st.write("正在调用Deepseek API...")
        if st.session_state.api_status['deepseek']['ok']:
            try:
                # 使用硬编码的配置
                API_URL = f"{get_config('DEEPSEEK_API_BASE')}/chat/completions"
                API_KEY = get_config("DEEPSEEK_API_KEY")
                MODEL = "deepseek-ai/DeepSeek-R1"
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}"
                }
                data = {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": "你是一位有帮助的助手。请根据给定的上下文回答问题。始终使用中文回答，无论问题是什么语言。在回答之后，请务必提供一段最相关的原文摘录，以'相关原文：'为前缀。"},
                        {"role": "user", "content": f"上下文: {context_text}\n\n问题: {query}\n\n请提供你的回答然后在回答后面附上相关的原文摘录，以'相关原文：'为前缀。"}
                    ],
                    "max_tokens": 1000
                }
                
                response = requests.post(
                    API_URL,
                    headers=headers,
                    json=data
                )
                
                if response.status_code != 200:
                    error_detail = response.json() if response.text else "无详细错误信息"
                    st.write(f"错误详情: {error_detail}")
                    raise Exception(f"API返回错误: {error_detail}")
                
                response_data = response.json()
                deepseek_answer = response_data['choices'][0]['message']['content']
                st.write("Deepseek API调用成功")
                
                # 解析推理过程和答案
                think_pattern = r'/think/(.*?)/think/'
                think_matches = re.findall(think_pattern, deepseek_answer, re.DOTALL)
                
                # 移除所有推理过程，得到最终答案
                final_answer = re.sub(think_pattern, '', deepseek_answer, flags=re.DOTALL)
                
                # 处理相关原文
                if "相关原文：" in final_answer:
                    answer_parts = final_answer.split("相关原文：", 1)
                    responses['deepseek'] = answer_parts[0].strip()
                    excerpts['deepseek'] = answer_parts[1].strip()
                else:
                    responses['deepseek'] = final_answer.strip()
                
                # 更新Deepseek回答显示
                deepseek_content = f"""
                <div style='background-color: #e6f3ff; padding: 15px; border-radius: 5px;'>
                    {responses['deepseek']}
                </div>
                """
                deepseek_placeholder.markdown(deepseek_content, unsafe_allow_html=True)
                
                # 如果有推理过程，显示在expander中
                if think_matches:
                    with deepseek_placeholder.expander("查看推理过程", expanded=False):
                        for i, think in enumerate(think_matches, 1):
                            st.markdown(f"""
                            <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 10px;'>
                                <strong>推理步骤 {i}:</strong><br>
                                {think.strip()}
                            </div>
                            """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Deepseek API调用出错: {str(e)}")
                st.error(f"Deepseek配置信息：\nURL: {API_URL}\n模型: {MODEL}")
                responses['deepseek'] = "API调用出错，请稍后重试"
                deepseek_placeholder.error("获取Deepseek回答失败")
        else:
            responses['deepseek'] = "Deepseek API 未连接"
            deepseek_placeholder.error("Deepseek API 未连接")

        # 显示来源文档
        if sources:
            st.markdown("### 来源文档")
            for file_name, context in sources:
                with st.expander(f"📄 {file_name}"):
                    st.write(context)
        
        # 显示相关原文
        excerpt = excerpts['chatgpt'] or excerpts['deepseek']
        if excerpt:
            st.markdown("### 相关原文")
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #dee2e6;'>
                {excerpt}
            </div>
            """, unsafe_allow_html=True)

        return responses, sources, excerpt

    except Exception as e:
        st.error(f"处理查询时发生错误: {str(e)}")
        import traceback
        st.error(f"错误详情:\n{traceback.format_exc()}")
        return {
            'chatgpt': "处理查询时发生错误，请稍后重试",
            'deepseek': "处理查询时发生错误，请稍后重试"
        }, [], ""

# 保存索引和chunks
def save_index(file_name, chunks, index):
    if not os.path.exists('indices'):
        os.makedirs('indices')
    with open(f'indices/{file_name}.pkl', 'wb') as f:
        pickle.dump((chunks, index), f)
    # 保存文件名到一个列表中
    file_list_path = 'indices/file_list.txt'
    if os.path.exists(file_list_path):
        with open(file_list_path, 'r') as f:
            file_list = f.read().splitlines()
    else:
        file_list = []
    if file_name not in file_list:
        file_list.append(file_name)
        with open(file_list_path, 'w') as f:
            f.write('\n'.join(file_list))

# 加载所有保存的索引
def load_all_indices():
    file_indices = {}
    file_list_path = 'indices/file_list.txt'
    if os.path.exists(file_list_path):
        with open(file_list_path, 'r') as f:
            file_list = f.read().splitlines()
        for file_name in file_list:
            file_path = f'indices/{file_name}.pkl'
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    chunks, index = pickle.load(f)
                file_indices[file_name] = (chunks, index)
    return file_indices

def delete_index(file_name):
    if os.path.exists(f'indices/{file_name}.pkl'):
        os.remove(f'indices/{file_name}.pkl')
    file_list_path = 'indices/file_list.txt'
    if os.path.exists(file_list_path):
        with open(file_list_path, 'r') as f:
            file_list = f.read().splitlines()
        if file_name in file_list:
            file_list.remove(file_name)
            with open(file_list_path, 'w') as f:
                f.write('\n'.join(file_list))

def main():
    st.markdown("""
    <style>
    .reportview-container .main .block-container{
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    .stColumn {
        padding: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("AI知识问答系统")

    # 初始化 session state
    if "file_indices" not in st.session_state:
        st.session_state.file_indices = load_all_indices()

    # 文档上传部分
    st.subheader("文档上传")
    
    max_tokens = 4096

    uploaded_files = st.file_uploader("上传文档", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"正在处理文档: {uploaded_file.name}..."):
                chunks, index = vectorize_document(uploaded_file, max_tokens)
                st.session_state.file_indices[uploaded_file.name] = (chunks, index)
                save_index(uploaded_file.name, chunks, index)
            st.success(f"文档 {uploaded_file.name} 向量化并添加到索引中！")

    # 显示已处理的文件并添加删除按钮
    st.subheader("已处理文档:")
    for file_name in list(st.session_state.file_indices.keys()):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"• {file_name}")
        with col2:
            if st.button("删除", key=f"delete_{file_name}"):
                del st.session_state.file_indices[file_name]
                delete_index(file_name)
                st.success(f"文档 {file_name} 已删除！")
                st.rerun()

    # 添加关键词搜索功能
    st.subheader("关键词搜索")
    search_keywords = st.text_input("输入关键词（用空格分隔）")
    if search_keywords:
        keywords = search_keywords.split()
        relevant_docs = search_documents(keywords, st.session_state.file_indices)
        if relevant_docs:
            st.write("相关文档：")
            for doc in relevant_docs:
                st.write(f"• {doc}")
            st.session_state.relevant_docs = relevant_docs
        else:
            st.write("没有找到相关文档。")
            st.session_state.relevant_docs = None

    # 问答部分
    st.subheader("问答")
    query = st.text_input("请输入您的问题")
    
    if query:
        try:
            with st.spinner("正在查找答案..."):
                responses, sources, excerpt = rag_qa(
                    query, 
                    st.session_state.file_indices,
                    st.session_state.get('relevant_docs')
                )
                
                # 创建两列布局
                left_col, right_col = st.columns(2)
                
                # 左侧显示ChatGPT回答
                with left_col:
                    st.markdown("### ChatGPT回答")
                    if responses and 'chatgpt' in responses and responses['chatgpt']:
                        st.markdown(f"""
                        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px;'>
                            {responses['chatgpt']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("获取ChatGPT回答失败")
                
                # 右侧显示Deepseek回答
                with right_col:
                    st.markdown("### Deepseek回答")
                    if responses and 'deepseek' in responses and responses['deepseek']:
                        st.markdown(f"""
                        <div style='background-color: #e6f3ff; padding: 15px; border-radius: 5px;'>
                            {responses['deepseek']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("获取Deepseek回答失败")
                
                # 显示来源文档
                if sources:
                    st.markdown("### 来源文档")
                    for file_name, context in sources:
                        with st.expander(f"📄 {file_name}"):
                            st.write(context)
                
                # 显示相关原文
                if excerpt:
                    st.markdown("### 相关原文")
                    st.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #dee2e6;'>
                        {excerpt}
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"处理问题时发生错误: {str(e)}")
            import traceback
            st.error(f"错误详情:\n{traceback.format_exc()}")
            st.info("请检查API配置是否正确，或稍后重试")

if __name__ == "__main__":
    main() 