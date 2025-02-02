# AI知识问答系统

基于ChatGPT和Deepseek的本地知识库问答系统，支持PDF、Word和文本文件的智能问答。

## 功能特点

- 支持多种文档格式（PDF、DOCX、TXT）
- 双模型对比（ChatGPT和Deepseek）
- 实时显示处理进度
- 文档关键词搜索
- 支持查看AI推理过程
- 显示相关原文和来源

## 安装说明

1. 克隆项目：
```bash
git clone [项目地址]
cd [项目目录]
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 配置API密钥：
   - 复制 `.env.example` 为 `.env`
   - 在 `.env` 中填入你的API密钥
   - 创建 `.streamlit/secrets.toml` 并配置相同的API密钥

## 运行应用

```bash
streamlit run app.py
```

## 部署说明

### 本地运行
1. 确保已安装所有依赖
2. 配置好 `.env` 文件
3. 运行 `streamlit run app.py`

### Streamlit Cloud部署
1. 在Streamlit Cloud中导入项目
2. 在项目设置中配置以下secrets：
   ```toml
   OPENAI_API_KEY = "你的OpenAI API密钥"
   OPENAI_API_BASE = "你的OpenAI API基础URL"
   DEEPSEEK_API_KEY = "你的Deepseek API密钥"
   DEEPSEEK_API_BASE = "你的Deepseek API基础URL"
   ```

## 目录结构

```
.
├── .env                 # 环境配置文件
├── .env.example        # 环境配置示例
├── .streamlit/         # Streamlit配置目录
│   └── secrets.toml    # Streamlit密钥配置
├── app.py              # 主应用文件
├── indices/            # 文档索引目录（运行时生成）
├── requirements.txt    # 项目依赖
└── README.md          # 项目文档
```

## 注意事项

1. 不要将包含真实API密钥的文件提交到版本控制系统
2. 确保 `.env` 和 `.streamlit/secrets.toml` 已添加到 `.gitignore`
3. 首次运行时需要等待模型下载
4. 处理大型文件时可能需要较长时间 