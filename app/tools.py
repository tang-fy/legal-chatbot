"""SQL 工具函数:数据库连接、NL2SQL 模型与 SQL 清洗.

LangGraph 改造后,检索和 SQL 查询已分别由 research / sql_node 直接调用,
这里只保留被节点复用的底层函数(get_db / _get_sql_llm / _extract_sql),
不再提供 @tool 包装(原 ReAct 架构遗留的 search_legal_documents、
query_case_database、all_tools 已删除).
"""
import re

from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

from .config import (
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DB,
    LLM_BASE_URL,
    LLM_API_KEY,
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


# 创建SQLAlchemy引擎(连接MySQL)
_db_engine = None

# 包装为LangChain的SQLDatabase对象
_db = None

# 缓存 NL2SQL 专用 LLM 实例,避免每次查询都新建
_sql_llm = None


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


def _get_sql_llm():
    """惰性初始化 NL2SQL 专用 LLM,复用实例避免重复创建开销."""
    global _sql_llm
    if _sql_llm is None:
        from langchain_openai import ChatOpenAI
        _sql_llm = ChatOpenAI(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=0,  # 温度设为0,确保SQL生成稳定
        )
    return _sql_llm
