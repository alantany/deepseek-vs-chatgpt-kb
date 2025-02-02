"""
UI模块
"""

import streamlit as st
from .document_processor import vectorize_document
from .index_manager import save_index, delete_index
from .search import extract_keywords, search_documents
from .qa_system import process_chatgpt_response, process_deepseek_response

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
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        st.markdown("### ChatGPT")
        if chatgpt_status['ok']:
            st.success(f"✅ {chatgpt_status['model']}\n\n状态：{chatgpt_status['message']}")
        else:
            st.error(f"❌ {chatgpt_status['model']}\n\n状态：{chatgpt_status['message']}")
    
    with status_col2:
        st.markdown("### DeepSeek")
        if deepseek_status['ok']:
            st.success(f"✅ {deepseek_status['model']}\n\n状态：{deepseek_status['message']}")
        else:
            st.error(f"❌ {deepseek_status['model']}\n\n状态：{deepseek_status['message']}")
    
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
        
        # 创建一个默认收起的expander用于显示推理过程
        thinking_expander = st.expander("🤔 查看Deepseek推理过程", expanded=False)
        with thinking_expander:
            thinking_container = st.empty()
            thinking_container.markdown("""
            <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                <p style='color: #666; margin-bottom: 10px;'>等待开始推理...</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 创建一个空的容器用于显示答案
        deepseek_answer_container = st.empty()
        deepseek_answer_container.markdown("""
        <div style='background-color: #e6f3ff; padding: 15px; border-radius: 5px;'>
            <div class="loading">等待开始处理...</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 处理ChatGPT回答
    chatgpt_answer, chatgpt_excerpt = process_chatgpt_response(chatgpt_client, context_text, query)
    chatgpt_placeholder.markdown(f"""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px;'>
        {chatgpt_answer}
    </div>
    """, unsafe_allow_html=True)
    
    # 处理Deepseek回答
    deepseek_answer, deepseek_excerpt, thinking_steps = process_deepseek_response(
        context_text, 
        query, 
        deepseek_answer_container,
        thinking_container
    )
    
    return (chatgpt_answer, chatgpt_excerpt), (deepseek_answer, deepseek_excerpt) 