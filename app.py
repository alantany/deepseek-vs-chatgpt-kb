"""
AI知识问答系统

使用方法：
1. 上传文档（支持PDF、DOCX、TXT格式）
2. 输入问题
3. 获取AI回答
"""

import streamlit as st
from src.config import check_configs
from src.api_clients import init_api_clients, test_api_connection
from src.index_manager import load_all_indices
from src.search import vector_search
from src.ui import (
    setup_page,
    show_model_status,
    handle_file_upload,
    show_document_list,
    handle_keyword_search,
    handle_qa
)

def main():
    # 设置页面
    setup_page()
    
    # 检查配置
    missing_configs = check_configs()
    if missing_configs:
        st.error("""
        ### 缺少必要的配置
        请在以下位置之一配置这些值：
        1. Streamlit Cloud部署：在项目设置中添加 secrets.toml
        2. 本地开发：在项目根目录创建 .env 文件
        
        缺少的配置：
        """ + "\n".join(missing_configs))
        st.stop()
    
    # 初始化API客户端
    chatgpt_client, deepseek_client = init_api_clients()
    
    # 测试API连接
    if 'api_status' not in st.session_state:
        st.session_state.api_status = {}
        
    chatgpt_ok, chatgpt_msg, chatgpt_model = test_api_connection(chatgpt_client, "ChatGPT")
    deepseek_ok, deepseek_msg, deepseek_model = test_api_connection(deepseek_client, "Deepseek")
    
    st.session_state.api_status['chatgpt'] = {
        'ok': chatgpt_ok,
        'message': chatgpt_msg,
        'model': chatgpt_model
    }
    st.session_state.api_status['deepseek'] = {
        'ok': deepseek_ok,
        'message': deepseek_msg,
        'model': deepseek_model
    }
    
    # 显示模型状态
    show_model_status(
        st.session_state.api_status['chatgpt'],
        st.session_state.api_status['deepseek']
    )
    
    # 初始化session state
    if "file_indices" not in st.session_state:
        st.session_state.file_indices = load_all_indices()
    
    # 处理文件上传
    handle_file_upload()
    
    # 显示文档列表
    show_document_list()
    
    # 处理关键词搜索
    handle_keyword_search()
    
    # 问答部分
    st.subheader("问答")
    query = st.text_input("请输入您的问题")
    
    if query:
        try:
            with st.spinner("正在查找答案..."):
                # 执行向量搜索
                search_results = vector_search(query, st.session_state.file_indices)
                if not search_results:
                    st.error("未找到相关内容，请尝试其他问题。")
                    st.stop()
                
                # 准备上下文
                context_text = "\n".join([chunk for _, chunk in search_results])
                
                # 处理问答
                (chatgpt_answer, chatgpt_excerpt), (deepseek_answer, deepseek_excerpt) = handle_qa(
                    query, chatgpt_client, context_text
                )
                
                # 显示来源文档
                if search_results:
                    st.markdown("### 来源文档")
                    for file_name, context in search_results:
                        with st.expander(f"📄 {file_name}"):
                            st.write(context)
                
                # 显示相关原文
                excerpt = chatgpt_excerpt or deepseek_excerpt
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