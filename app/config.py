import os

# ===== LLM 配置:DeepSeek API (OpenAI 兼容协议) =====
# DeepSeek-V3 通过 OpenAI 兼容端点调用,速度远快于本地 7B 模型
# API Key 从环境变量读取,不要硬编码到代码里(安全 + 方便切换)
LLM_BASE_URL = os.getenv('LLM_BASE_URL', "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv('DEEPSEEK_API_KEY', "")
LLM_MODEL = os.getenv('LLM_MODEL', "deepseek-chat")  # deepseek-chat 即 V3

# ===== 向量嵌入模型(仍用本地 Ollama,不依赖 API) =====
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', "http://localhost:11434")
EMBED_MODEL = "bge-m3"  # 用于生成向量的嵌入模型

#Milvus向量数据库配置
MILVUS_HOST = os.getenv('MILVUS_HOST', "milvus")
MILVUS_PORT = os.getenv('MILVUS_PORT', "19530")
MILVUS_COLLECTION = "legal_documents" #集合名称,类似数据库表

#MySQL配置
MYSQL_HOST = os.getenv('MYSQL_HOST', "localhost")
MYSQL_PORT = int(os.getenv('MYSQL_PORT', "3306"))
MYSQL_USER = os.getenv('MYSQL_USER', "root")
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', "123456")
MYSQL_DB = os.getenv('MYSQL_DB', "legal_db")

#文档处理与检索参数
DOCUMENTS_DIR = "./data/documents"  #法律文档存放目录(挂载卷)
CHUNK_SIZE = 1200 #每个文本块的最大字符数(法律条文通常较长,800太小容易切断法条)
CHUNK_OVERLAP = 200 #相邻文本块重叠字符数,保证上下文连贯(200/1200≈17%重叠率)
RAG_TOP_K = 3 # 检索返回的最相关文档片段数量,3条×1200字≈3500tokens,与num_ctx=8192配合避免上下文溢出
HISTORY_LIMIT = 100 #会话历史保留的消息数量

#嵌入批处理大小
EMBED_BATCH_SIZE = 32 #每批发送判给Embedding模型的文本数量