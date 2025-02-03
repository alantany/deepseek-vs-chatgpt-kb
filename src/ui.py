"""
UI模块
"""

import streamlit as st
from .document_processor import vectorize_document
from .index_manager import save_index, delete_index
from .search import extract_keywords, search_documents
from .qa_system import process_chatgpt_response, process_deepseek_response, process_local_deepseek_response
from .config import get_config, OLLAMA_DEEPSEEK_MODEL
import requests

def setup_page():
    """
    设置页面基本配置
    """
    st.set_page_config(
        page_title="AI知识问答系统 - by Huaiyuan Tan",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items=None
    )
    
    # 添加开发者信息
    st.markdown("<h6 style='text-align: right; color: gray;'>开发者: Huaiyuan Tan</h6>", unsafe_allow_html=True)
    
    # 自定义样式
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

def show_model_status(chatgpt_status, deepseek_status):
    """
    显示模型状态
    """
    st.markdown("## 🤖 模型状态")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ChatGPT")
        if chatgpt_status['ok']:
            st.success(f"✅ {chatgpt_status['model']}\n\n状态：{chatgpt_status['message']}")
        else:
            st.error(f"❌ {chatgpt_status['model']}\n\n状态：{chatgpt_status['message']}")
    
    with col2:
        st.markdown("### Deepseek")
        if deepseek_status['ok']:
            st.success(f"✅ {deepseek_status['model']}\n\n状态：{deepseek_status['message']}")
        else:
            st.error(f"❌ {deepseek_status['model']}\n\n状态：{deepseek_status['message']}")
    
    with col3:
        st.markdown("### 本地DeepSeek")
        try:
            # 获取Ollama基础URL
            ollama_url = get_config('OLLAMA_BASE_URL')
            # 获取模型名称
            model_name = get_config('OLLAMA_DEEPSEEK_MODEL')
            
            try:
                response = requests.get(f"{ollama_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    if model_name in [model['name'] for model in models]:
                        st.success(f"✅ {model_name}\n\n状态：连接正常")
                    else:
                        st.warning(f"⚠️ {model_name}\n\n状态：模型未加载")
                else:
                    st.error(f"❌ 本地DeepSeek\n\n状态：连接失败 - HTTP {response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ 本地DeepSeek\n\n状态：连接错误 - {str(e)}")
        except Exception as e:
            st.error("❌ 本地DeepSeek\n\n状态：配置错误")
    
    st.markdown("---")

def handle_file_upload(max_tokens=4096):
    """
    处理文件上传
    """
    st.subheader("文档上传")
    uploaded_files = st.file_uploader("上传文档", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"正在处理文档: {uploaded_file.name}..."):
                chunks, index = vectorize_document(uploaded_file, max_tokens)
                st.session_state.file_indices[uploaded_file.name] = (chunks, index)
                save_index(uploaded_file.name, chunks, index)
            st.success(f"文档 {uploaded_file.name} 向量化并添加到索引中！")

def show_document_list():
    """
    显示已处理的文档列表
    """
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

def handle_keyword_search():
    """
    处理关键词搜索
    """
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

def handle_qa(query, chatgpt_client, context_text):
    """
    处理问答
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ChatGPT回答")
        chatgpt_placeholder = st.empty()
        chatgpt_placeholder.markdown("""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px;'>
            <div class="loading">正在处理中...</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### DeepSeek回答")
        
        # 创建一个默认收起的expander用于显示推理过程
        thinking_expander = st.expander("🤔 查看Deepseek推理过程", expanded=False)
        with thinking_expander:
            thinking_container = st.empty()
            thinking_container.markdown("""
            <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                <p style='color: #666; margin-bottom: 10px;'>等待处理中...</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 创建一个空的容器用于显示答案
        deepseek_answer_container = st.empty()
        deepseek_answer_container.markdown("""
        <div style='background-color: #e6f3ff; padding: 15px; border-radius: 5px;'>
            <div class="loading">等待处理中...</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("### 本地DeepSeek回答")
        
        # 创建一个默认收起的expander用于显示本地模型的推理过程
        local_thinking_expander = st.expander("🤔 查看本地Deepseek推理过程", expanded=False)
        with local_thinking_expander:
            local_thinking_container = st.empty()
            local_thinking_container.markdown("""
            <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                <p style='color: #666; margin-bottom: 10px;'>等待处理中...</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 创建一个空的容器用于显示本地模型的答案
        local_deepseek_answer_container = st.empty()
        local_deepseek_answer_container.markdown("""
        <div style='background-color: #e6f3ff; padding: 15px; border-radius: 5px;'>
            <div class="loading">等待处理中...</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 初始化返回值
    chatgpt_result = ("", "")
    deepseek_result = ("", "")
    local_deepseek_result = ("", "")
    
    # 处理ChatGPT回答
    try:
        chatgpt_answer, chatgpt_excerpt = process_chatgpt_response(chatgpt_client, context_text, query)
        chatgpt_placeholder.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px;'>
            {chatgpt_answer}
        </div>
        """, unsafe_allow_html=True)
        chatgpt_result = (chatgpt_answer, chatgpt_excerpt)
    except Exception as e:
        chatgpt_placeholder.markdown(f"""
        <div style='background-color: #ffe6e6; padding: 15px; border-radius: 5px;'>
            ChatGPT服务暂时不可用: {str(e)}
        </div>
        """, unsafe_allow_html=True)
    
    # 更新Deepseek状态为正在处理
    thinking_container.markdown("""
    <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
        <p style='color: #666; margin-bottom: 10px;'>正在推理中...</p>
    </div>
    """, unsafe_allow_html=True)
    deepseek_answer_container.markdown("""
    <div style='background-color: #e6f3ff; padding: 15px; border-radius: 5px;'>
        <div class="loading">正在处理中...</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 处理Deepseek回答
    try:
        deepseek_answer, deepseek_excerpt, thinking_steps = process_deepseek_response(
            context_text, 
            query, 
            deepseek_answer_container,
            thinking_container
        )
        deepseek_result = (deepseek_answer, deepseek_excerpt)
    except Exception as e:
        deepseek_answer_container.markdown(f"""
        <div style='background-color: #ffe6e6; padding: 15px; border-radius: 5px;'>
            Deepseek服务暂时不可用: {str(e)}
        </div>
        """, unsafe_allow_html=True)
        thinking_container.markdown("""
        <div style='background-color: #ffe6e6; padding: 10px; border-radius: 5px;'>
            <p style='color: #666;'>推理过程不可用</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 更新本地Deepseek状态为正在处理
    local_thinking_container.markdown("""
    <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
        <p style='color: #666; margin-bottom: 10px;'>正在推理中...</p>
    </div>
    """, unsafe_allow_html=True)
    local_deepseek_answer_container.markdown("""
    <div style='background-color: #e6f3ff; padding: 15px; border-radius: 5px;'>
        <div class="loading">正在处理中...</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 处理本地DeepSeek回答
    try:
        local_deepseek_answer, local_deepseek_excerpt, local_thinking_steps = process_local_deepseek_response(
            context_text,
            query,
            local_deepseek_answer_container,
            local_thinking_container
        )
        local_deepseek_result = (local_deepseek_answer, local_deepseek_excerpt)
    except Exception as e:
        local_deepseek_answer_container.markdown(f"""
        <div style='background-color: #ffe6e6; padding: 15px; border-radius: 5px;'>
            本地Deepseek服务暂时不可用: {str(e)}
        </div>
        """, unsafe_allow_html=True)
        local_thinking_container.markdown("""
        <div style='background-color: #ffe6e6; padding: 10px; border-radius: 5px;'>
            <p style='color: #666;'>推理过程不可用</p>
        </div>
        """, unsafe_allow_html=True)
    
    return chatgpt_result, deepseek_result 