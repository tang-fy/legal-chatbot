from datetime import datetime
import re
import requests
import pandas as pd
from langchain_classic.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_core.tools import tool
from .vector_store import similarity_search
from .config import (
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DB,
    RAG_TOP_K,
    OLLAMA_BASE_URL,
    LLM_MODEL
)


def _extract_sql(text: str) -> str:
    """
    从 LLM 输出中提取可直接执行的 SQL.
    本地小模型常见两种"不守规矩"的输出:
    1. 回显 prompt 前缀: "Question: ...\\nSQLQuery: SELECT ..."
    2. 用 markdown 代码块包裹: "```sql\\nSELECT ...\\n```"
    """
    sql = (text or "").strip()
    # 优先提取 markdown 代码块中的内容
    fence = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.DOTALL | re.IGNORECASE)
    if fence:
        sql = fence.group(1).strip()
    # 取最后一个 SQLQuery: 之后的部分,去掉回显的 Question 等前缀
    if "SQLQuery:" in sql:
        sql = sql.split("SQLQuery:")[-1].strip()
    # 去掉末尾分号后可能跟随的自然语言解释,只保留第一条语句
    sql = sql.split("\n\n")[0].strip()
    return sql.rstrip(";").strip()

#工具1:法律文档检索(核心RAG工具)
@tool
def search_legal_documents(query: str) -> str:
    """
    在本地法律文档库(Milvus)中检索相关内容.
    当用户询问法律问题,需要法律依据时,Agent应调用此工具.
    """
    docs = similarity_search(query, k=RAG_TOP_K)
    if not docs:
        return "没有找到相关的法律文档."
    #将所有检索到的片段拼接,并附上标记
    context = "\n\n".join([f"[文档片段]\n{doc.page_content}" for doc in docs])
    return context

#工具2:案例数据库查询
#创建SQLAlchemy引擎(连接MySQL)
_db_engine= None

#包装为LangChain的SQLDatabase对象
_db = None

def get_db():
    """惰性初始化数据库连接,避免模块导入时就连接MySQL."""
    global _db_engine, _db
    if _db is None:
        _db_engine = create_engine(
            f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
            "?charset=utf8mb4",
            pool_pre_ping=True,
        )
        _db = SQLDatabase(_db_engine)
    return _db

@tool
def query_case_database(question: str) -> str:
    """
    查询MySQL中的法律案例数据库.
    内部使用另一个LLM将自然语言问题转换为SQL,然后执行并返回结果.
    """
    from langchain_classic.chains import create_sql_query_chain
    from langchain_ollama import ChatOllama
    #惰性获取数据库连接(首次调用时才真正连接MySQL)
    db = get_db()
    #创建专用的SQL生成LLM(温度设为0,确保生成稳定)
    sql_llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )
    #使用LangChain的SQL查询链
    chain = create_sql_query_chain(sql_llm, db)
    try:
        raw_output = chain.invoke({"question": question})
        sql_query = _extract_sql(raw_output) #清洗模型输出,提取可执行SQL
        result = db.run(sql_query) #执行SQL
        return f"生成的SQL:\n{sql_query}\n\n查询结果:\n{result}"
    except Exception as e:
        return f"数据库查询失败: {e}"

#所有工具列表,将传递给Agent
all_tools = [
    search_legal_documents,
    query_case_database
]