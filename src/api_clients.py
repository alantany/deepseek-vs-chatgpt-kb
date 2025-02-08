"""
API客户端初始化和连接测试模块
"""

import requests
from openai import OpenAI
import streamlit as st
from .config import get_config

def init_api_clients():
    """
    初始化ChatGPT和Deepseek API客户端
    """
    # 初始化ChatGPT客户端
    chatgpt_client = OpenAI(
        api_key=get_config("OPENAI_API_KEY"),
        base_url=get_config("OPENAI_API_BASE"),
    )

    # 处理Deepseek基础URL
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

    return chatgpt_client, deepseek_client

def test_api_connection(client, model_name):
    """
    测试API连接状态
    """
    try:
        st.write(f"正在测试 {model_name} API连接...")
        
        if model_name == "Deepseek":
            API_URL = f"{get_config('DEEPSEEK_API_BASE')}/chat/completions"
            API_KEY = get_config("DEEPSEEK_API_KEY")
            MODEL = "deepseek/deepseek-r1:free"
            
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