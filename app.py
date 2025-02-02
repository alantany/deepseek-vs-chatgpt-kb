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
import time
import json

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

# 自定义样式，但保留Streamlit默认UI元素
custom_style = """
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
"""
st.markdown(custom_style, unsafe_allow_html=True)

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
        # 使用进度条替代普通文本提示
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1. 关键词提取和文档搜索 (10%)
        status_text.text("正在分析问题关键词...")
        keywords = extract_keywords(query)
        if relevant_docs is None:
            relevant_docs = search_documents(keywords, file_indices)
        progress_bar.progress(10)
        
        if not relevant_docs:
            status_text.error("未找到相关文档，请尝试使用不同的关键词。")
            return {
                'chatgpt': "没有找到相关文档。请尝试使用不同的关键词。",
                'deepseek': "没有找到相关文档。请尝试使用不同的关键词。"
            }, [], ""

        # 2. 向量检索准备 (20%)
        status_text.text("正在准备相关文档内容...")
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
        progress_bar.progress(20)

        if not all_chunks:
            status_text.error("无法从文档中提取内容，请确保文档已正确上传。")
            return {
                'chatgpt': "没有找到相关信息。请确保已上传文档。",
                'deepseek': "没有找到相关信息。请确保已上传文档。"
            }, [], ""

        # 3. 执行向量检索 (30%)
        status_text.text("正在检索最相关的内容片段...")
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
        progress_bar.progress(30)

        # 4. 准备上下文 (40%)
        status_text.text("正在整理上下文信息...")
        context_text = "\n".join(context)
        max_context_tokens = 3000
        original_length = len(context_text)
        while num_tokens_from_string(context_text) > max_context_tokens:
            context_text = context_text[:int(len(context_text)*0.9)]
        progress_bar.progress(40)
        
        if not context_text:
            status_text.error("无法生成有效的上下文内容。")
            return {
                'chatgpt': "没有找到相关信息。",
                'deepseek': "没有找到相关信息。"
            }, [], ""

        # 5. 创建UI布局 (45%)
        left_col, right_col = st.columns(2)
        with left_col:
            st.markdown("### ChatGPT回答")
            chatgpt_placeholder = st.empty()
            chatgpt_placeholder.markdown("""
            <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px;'>
                <div class="loading">正在等待ChatGPT回答...</div>
            </div>
            """, unsafe_allow_html=True)
        
        with right_col:
            st.markdown("### Deepseek回答")
            # 创建推理过程的expander
            thinking_expander = st.expander("🤔 查看Deepseek实时推理过程", expanded=True)
            with thinking_expander:
                st.markdown("""
                <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                    <p style='color: #666; margin-bottom: 10px;'>等待开始推理...</p>
                </div>
                """, unsafe_allow_html=True)
                thinking_placeholder = st.empty()
            
            deepseek_placeholder = st.empty()
            deepseek_placeholder.markdown("""
            <div style='background-color: #e6f3ff; padding: 15px; border-radius: 5px;'>
                <div class="loading">等待开始处理...</div>
            </div>
            """, unsafe_allow_html=True)

        responses = {'chatgpt': "", 'deepseek': ""}
        excerpts = {'chatgpt': "", 'deepseek': ""}
        progress_bar.progress(45)

        # 6. 处理ChatGPT回答 (45-70%)
        status_text.text("正在获取ChatGPT回答...")
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
        progress_bar.progress(70)

        # 7. 处理Deepseek回答 (70-95%)
        status_text.text("正在获取Deepseek回答...")
        deepseek_placeholder.markdown("""
        <div style='background-color: #e6f3ff; padding: 15px; border-radius: 5px;'>
            <div class="loading">正在处理Deepseek回答...</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.api_status['deepseek']['ok']:
            try:
                # 更新推理过程的状态提示
                with thinking_expander:
                    st.markdown("""
                    <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                        <p style='color: #666; margin-bottom: 10px;'>正在进行推理...</p>
                    </div>
                    """, unsafe_allow_html=True)
                
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
                        {"role": "system", "content": "你是一位有帮助的助手。请根据给定的上下文回答问题。始终使用中文回答，无论问题是什么语言。在回答之后，请务必提供一段最相关的原文摘录，以'相关原文：'为前缀。在回答过程中，请使用'/think/你的推理过程/think/'的格式来展示你的推理过程。"},
                        {"role": "user", "content": f"上下文: {context_text}\n\n问题: {query}\n\n请一步步思考并回答这个问题。在思考过程中，用'/think/你的推理过程/think/'格式来展示你的推理过程，最后提供完整答案和相关原文。"}
                    ],
                    "max_tokens": 1000,
                    "stream": True
                }
                
                # 使用stream模式发送请求
                response = requests.post(
                    API_URL,
                    headers=headers,
                    json=data,
                    stream=True
                )
                
                if response.status_code != 200:
                    error_detail = response.json() if response.text else "无详细错误信息"
                    raise Exception(f"API返回错误: {error_detail}")
                
                # 用于存储完整的响应
                full_response = ""
                current_think = ""
                current_answer = ""
                think_count = 0
                
                # 创建一个空的容器用于显示实时推理过程
                thinking_container = thinking_placeholder.container()
                
                # 处理流式响应
                for line in response.iter_lines():
                    if line:
                        # 移除"data: "前缀并解析JSON
                        json_str = line.decode('utf-8').replace('data: ', '')
                        if json_str.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(json_str)
                            if chunk.get('choices') and chunk['choices'][0].get('delta', {}).get('content'):
                                content = chunk['choices'][0]['delta']['content']
                                full_response += content
                                
                                # 检查是否在推理过程中
                                if '/think/' in content:
                                    if current_think:
                                        # 如果已经有一个推理过程在进行，先保存它
                                        think_count += 1
                                        thinking_container.markdown(f"""
                                        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; border: 1px solid #e9ecef;'>
                                            <div style='color: #495057; margin-bottom: 8px;'><strong>🔄 推理步骤 {think_count}</strong></div>
                                            <div style='color: #212529;'>{current_think.strip()}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    current_think = content.replace('/think/', '')
                                elif current_think is not None:
                                    if '/think/' in content:  # 结束当前推理
                                        current_think = current_think.replace('/think/', '')
                                        think_count += 1
                                        thinking_container.markdown(f"""
                                        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; border: 1px solid #e9ecef;'>
                                            <div style='color: #495057; margin-bottom: 8px;'><strong>🔄 推理步骤 {think_count}</strong></div>
                                            <div style='color: #212529;'>{current_think.strip()}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        current_think = None
                                    else:
                                        current_think += content
                                        # 实时更新当前推理步骤
                                        thinking_container.markdown(f"""
                                        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; border: 1px solid #e9ecef;'>
                                            <div style='color: #495057; margin-bottom: 8px;'><strong>🔄 推理步骤 {think_count + 1} (进行中...)</strong></div>
                                            <div style='color: #212529;'>{current_think.strip()}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    current_answer += content
                                    # 实时更新答案
                                    if not content.startswith('/think/'):
                                        deepseek_placeholder.markdown(f"""
                                        <div style='background-color: #e6f3ff; padding: 15px; border-radius: 5px;'>
                                            {current_answer}
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                        except json.JSONDecodeError:
                            continue
                
                # 解析最终答案
                deepseek_answer = full_response
                
                # 移除所有推理过程，得到最终答案
                final_answer = re.sub(r'/think/.*?/think/', '', deepseek_answer, flags=re.DOTALL)
                
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
                
            except Exception as e:
                responses['deepseek'] = "API调用出错，请稍后重试"
                deepseek_placeholder.error("获取Deepseek回答失败")
        else:
            responses['deepseek'] = "Deepseek API 未连接"
            deepseek_placeholder.error("Deepseek API 未连接")
        progress_bar.progress(95)

        # 8. 显示补充信息 (95-100%)
        status_text.text("正在整理补充信息...")
        
        # 显示来源文档
        if context_with_sources:
            st.markdown("### 来源文档")
            for file_name, context in context_with_sources:
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

        progress_bar.progress(100)
        status_text.text("处理完成！")
        
        # 清理临时UI元素
        time.sleep(0.5)  # 给用户一个短暂的时间看到完成状态
        progress_bar.empty()
        status_text.empty()

        return responses, context_with_sources, excerpt

    except Exception as e:
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()
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
                # 只调用rag_qa函数，不再重复显示结果
                rag_qa(
                    query, 
                    st.session_state.file_indices,
                    st.session_state.get('relevant_docs')
                )
        except Exception as e:
            st.error(f"处理问题时发生错误: {str(e)}")
            import traceback
            st.error(f"错误详情:\n{traceback.format_exc()}")
            st.info("请检查API配置是否正确，或稍后重试")

if __name__ == "__main__":
    main() 