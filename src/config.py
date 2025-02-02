"""
配置管理模块
"""

import os
import streamlit as st
from dotenv import load_dotenv

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

# 定义必需的配置项
REQUIRED_CONFIGS = {
    'OPENAI_API_KEY': '用于ChatGPT的API密钥',
    'OPENAI_API_BASE': 'ChatGPT的API基础URL',
    'DEEPSEEK_API_KEY': '用于Deepseek的API密钥',
    'DEEPSEEK_API_BASE': 'Deepseek的API基础URL'
}

def check_configs():
    """
    检查必需的配置是否都已设置
    返回缺失的配置列表
    """
    missing_configs = []
    for config, description in REQUIRED_CONFIGS.items():
        if not get_config(config):
            missing_configs.append(f"{config} ({description})")
    return missing_configs 