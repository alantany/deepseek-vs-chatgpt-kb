"""
问答系统模块
"""

import json
import time
import requests
import streamlit as st
from .search import extract_keywords, search_documents, vector_search
from .config import get_config, OLLAMA_DEEPSEEK_MODEL

def process_chatgpt_response(client, context_text, query):
    """
    处理ChatGPT的回答
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一位有帮助的助手。请根据给定的上下文回答问题。始终使用中文回答，无论问题是什么语言。在回答之后，请务必提供一段最相关的原文摘录，以'相关原文：'为前缀。"},
                {"role": "user", "content": f"上下文: {context_text}\n\n问题: {query}\n\n请提供你的回答然后在回答后面附上相关的原文摘录，以'相关原文：'为前缀。"}
            ]
        )
        answer = response.choices[0].message.content
        
        # 分离答案和原文
        if "相关原文：" in answer:
            answer_parts = answer.split("相关原文：", 1)
            return answer_parts[0].strip(), answer_parts[1].strip()
        return answer.strip(), ""
    except Exception as e:
        st.error(f"ChatGPT API调用出错: {str(e)}")
        return "API调用出错，请稍后重试", ""

def process_deepseek_response(context_text, query, answer_container, thinking_container):
    """
    处理Deepseek的回答
    """
    try:
        API_URL = f"{get_config('DEEPSEEK_API_BASE')}/chat/completions"
        API_KEY = get_config("DEEPSEEK_API_KEY")
        MODEL = "deepseek/deepseek-r1:free"
        
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
        
        response = requests.post(API_URL, headers=headers, json=data, stream=True)
        
        if response.status_code != 200:
            raise Exception(f"API返回错误: {response.text}")
        
        # 用于收集所有内容的缓冲区
        full_content = ""
        current_chunk = ""
        
        def extract_and_display_answer(content):
            """
            从完整内容中提取答案并显示
            """
            # 找到最后一个推理过程的结束位置
            last_think_end = content.rfind('/think/')
            if last_think_end != -1:
                # 找到这个位置之后的第一个换行符
                next_newline = content.find('\n', last_think_end)
                if next_newline != -1:
                    # 提取答案部分（推理过程之后的内容）
                    answer = content[next_newline:].strip()
                    if answer:
                        # 如果答案中包含相关原文，只显示答案部分
                        if "相关原文：" in answer:
                            answer = answer.split("相关原文：")[0].strip()
                        # 更新答案显示
                        answer_container.markdown(f"""
                        <div style='background-color: #e6f3ff; padding: 15px; border-radius: 5px;'>
                            {answer}
                        </div>
                        """, unsafe_allow_html=True)
                        return answer
            return ""
        
        for line in response.iter_lines():
            if line:
                json_str = line.decode('utf-8').replace('data: ', '')
                if json_str.strip() == '[DONE]':
                    break
                
                try:
                    chunk = json.loads(json_str)
                    if not chunk.get('choices') or not chunk['choices'][0].get('delta', {}).get('content'):
                        continue
                        
                    content = chunk['choices'][0]['delta']['content']
                    full_content += content
                    current_chunk += content
                    
                    # 实时更新推理区域显示
                    thinking_container.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px;'>
                        {full_content}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 当收集到一定量的内容时尝试提取答案
                    if any(current_chunk.strip().endswith(end) for end in ['。', '！', '？', '：', '；', '\n']):
                        extract_and_display_answer(full_content)
                        current_chunk = ""
                
                except json.JSONDecodeError:
                    continue
        
        # 最后一次尝试提取答案
        final_answer = extract_and_display_answer(full_content)
        
        # 从最终内容中提取答案和原文
        if "相关原文：" in final_answer:
            answer_parts = final_answer.split("相关原文：", 1)
            return answer_parts[0].strip(), answer_parts[1].strip(), []
        return final_answer, "", []
        
    except Exception as e:
        st.error(f"Deepseek API调用出错: {str(e)}")
        return "API调用出错，请稍后重试", "", []

def process_local_deepseek_response(context_text, query, answer_container, thinking_container):
    """
    处理本地部署的Deepseek模型的回答
    """
    try:
        API_URL = f"{get_config('OLLAMA_BASE_URL')}/api/generate"
        
        # 构建提示词
        prompt = f"""你是一位有帮助的助手。请根据给定的上下文回答问题。始终使用中文回答，无论问题是什么语言。
在回答过程中，请使用'/think/你的推理过程/think/'的格式来展示你的推理过程。

上下文: {context_text}

问题: {query}

请一步步思考并回答这个问题。在思考过程中，用'/think/你的推理过程/think/'格式来展示你的推理过程，最后提供完整答案。"""
        
        data = {
            "model": OLLAMA_DEEPSEEK_MODEL,
            "prompt": prompt,
            "stream": True
        }
        
        response = requests.post(API_URL, json=data, stream=True)
        
        if response.status_code != 200:
            raise Exception(f"API返回错误: {response.text}")
        
        # 用于收集所有内容的缓冲区
        full_content = ""
        current_chunk = ""
        
        def extract_and_display_answer(content):
            """
            从完整内容中提取答案并显示
            """
            # 找到最后一个推理过程的结束位置
            last_think_end = content.rfind('/think/')
            if last_think_end != -1:
                # 找到这个位置之后的第一个换行符
                next_newline = content.find('\n', last_think_end)
                if next_newline != -1:
                    # 提取答案部分（推理过程之后的内容）
                    answer = content[next_newline:].strip()
                    if answer:
                        # 更新答案显示
                        answer_container.markdown(f"""
                        <div style='background-color: #e6f3ff; padding: 15px; border-radius: 5px;'>
                            {answer}
                        </div>
                        """, unsafe_allow_html=True)
                        return answer
            return ""
        
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    if not chunk.get('response'):
                        continue
                        
                    content = chunk['response']
                    full_content += content
                    current_chunk += content
                    
                    # 实时更新推理区域显示
                    thinking_container.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px;'>
                        {full_content}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 当收集到一定量的内容时尝试提取答案
                    if any(current_chunk.strip().endswith(end) for end in ['。', '！', '？', '：', '；', '\n']):
                        extract_and_display_answer(full_content)
                        current_chunk = ""
                
                except json.JSONDecodeError:
                    continue
        
        # 最后一次尝试提取答案
        final_answer = extract_and_display_answer(full_content)
        return final_answer, "", []
        
    except Exception as e:
        st.error(f"本地DeepSeek API调用出错: {str(e)}")
        return "API调用出错，请稍后重试", "", [] 