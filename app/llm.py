"""共享 LLM 实例.

所有节点从本模块导入 llm,避免重复初始化.
"""
from . import config
from langchain_openai import ChatOpenAI

# DeepSeek-V3 通过 OpenAI 兼容端点调用
llm = ChatOpenAI(
    model=config.LLM_MODEL,
    base_url=config.LLM_BASE_URL,
    api_key=config.LLM_API_KEY,
    temperature=0.1,
    max_tokens=2048,
)
